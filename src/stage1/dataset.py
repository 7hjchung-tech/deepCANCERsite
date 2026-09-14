"""Stage1Dataset: manifest + raw cache join, cohort validation, collate.

Row order between the manifest and the cache is never assumed to match --
everything is joined by var_id (see build_cohort). Unsupported variants are
reported (var_id + reason), never silently dropped, and the same supported/
excluded cohort is used for all three model modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

from .alignment import (
    AlignmentValidationError,
    UnsupportedVariantError,
    normalize_edit,
    validate_reconstruction,
)
from .cache import RawStage1Cache
from .metadata import MetaScaler, raw_meta_features
from .schema import SLOT_PAD
from .window import SampleTokens, attention_valid_for_mode, build_sample_tokens


@dataclass
class CohortReport:
    supported: list[dict]     # {"var_id", "row", "edit"}
    skipped: list[dict]       # {"var_id", "reason"}


def build_cohort(manifest_rows: list[dict], wt_seq: str) -> CohortReport:
    supported, skipped = [], []
    seen_ids: set[str] = set()
    for row in manifest_rows:
        var_id = str(row["var_id"])
        if var_id in seen_ids:
            skipped.append({"var_id": var_id, "reason": "duplicate var_id in manifest"})
            continue
        seen_ids.add(var_id)
        try:
            edit = normalize_edit(wt_seq, row)
            validate_reconstruction(wt_seq, row["mut_seq"], edit)
        except (UnsupportedVariantError, AlignmentValidationError) as e:
            skipped.append({"var_id": var_id, "reason": str(e)})
            continue
        supported.append({"var_id": var_id, "row": row, "edit": edit})
    return CohortReport(supported=supported, skipped=skipped)


def join_cohort_with_cache(cohort: CohortReport, cache: RawStage1Cache) -> CohortReport:
    """Further restrict `supported` to var_ids actually present in the cache,
    moving misses into `skipped` with an explicit reason (never a silent drop).
    """
    supported, skipped = [], list(cohort.skipped)
    for entry in cohort.supported:
        if entry["var_id"] not in cache.mut_entries:
            skipped.append({"var_id": entry["var_id"], "reason": "var_id missing from raw cache"})
            continue
        supported.append(entry)
    return CohortReport(supported=supported, skipped=skipped)


class Stage1Dataset(Dataset):
    def __init__(
        self,
        entries: list[dict],
        cache: RawStage1Cache,
        window_radius: int,
        layers: list[int],
        label_col: str = "z_score_D4_D14",
        meta_scaler: Optional[MetaScaler] = None,
    ) -> None:
        self.entries = entries
        self.cache = cache
        self.window_radius = window_radius
        self.layers = list(layers)
        self.label_col = label_col
        self.meta_scaler = meta_scaler

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        var_id, row, edit = entry["var_id"], entry["row"], entry["edit"]
        tokens = build_sample_tokens(var_id, edit, self.cache, self.window_radius, self.layers)
        meta_raw = raw_meta_features(edit)
        if self.meta_scaler is not None:
            meta_raw = self.meta_scaler.transform(meta_raw[None, :])[0]
        label = row.get(self.label_col)
        label = float(label) if label is not None and label == label else float("nan")
        return {"tokens": tokens, "meta_raw": meta_raw, "label": label, "var_id": var_id, "edit": edit}


def stage1_collate(batch: list[dict], model_mode: str) -> dict:
    """Pad variable-A samples to a common A_max and stack into batch tensors."""
    A_max = max(item["tokens"].n_slots for item in batch)
    layers = batch[0]["tokens"].layers
    L = len(layers)
    D = batch[0]["tokens"].H_wt.shape[-1]
    B = len(batch)

    H_wt = torch.zeros(B, L, A_max, D)
    H_mut = torch.zeros(B, L, A_max, D)
    delta = torch.zeros(B, L, A_max, D)
    wt_present = torch.zeros(B, A_max)
    mut_present = torch.zeros(B, A_max)
    delta_valid = torch.zeros(B, A_max)
    token_valid = torch.zeros(B, A_max)
    slot_kind = torch.full((B, A_max), SLOT_PAD, dtype=torch.long)
    wt_pos = torch.zeros(B, A_max, dtype=torch.long)
    mut_pos = torch.zeros(B, A_max, dtype=torch.long)
    anchor_rel_coord = torch.zeros(B, A_max)
    insertion_rank = torch.zeros(B, A_max)
    labels = torch.zeros(B)
    var_ids = []
    meta_raw = torch.zeros(B, batch[0]["meta_raw"].shape[0])

    for b, item in enumerate(batch):
        t: SampleTokens = item["tokens"]
        a = t.n_slots
        H_wt[b, :, :a] = t.H_wt
        H_mut[b, :, :a] = t.H_mut
        delta[b, :, :a] = t.delta
        wt_present[b, :a] = t.wt_present
        mut_present[b, :a] = t.mut_present
        delta_valid[b, :a] = t.delta_valid
        token_valid[b, :a] = t.token_valid
        slot_kind[b, :a] = t.slot_kind
        wt_pos[b, :a] = t.wt_pos
        mut_pos[b, :a] = t.mut_pos
        anchor_rel_coord[b, :a] = t.anchor_rel_coord
        insertion_rank[b, :a] = t.insertion_rank
        labels[b] = item["label"]
        var_ids.append(item["var_id"])
        meta_raw[b] = torch.as_tensor(item["meta_raw"], dtype=torch.float32)

    attention_valid = attention_valid_for_mode(token_valid, delta_valid, model_mode)

    return {
        "layers": layers,
        "H_wt": H_wt, "H_mut": H_mut, "delta": delta,
        "wt_present": wt_present, "mut_present": mut_present, "delta_valid": delta_valid,
        "token_valid": token_valid, "attention_valid": attention_valid,
        "slot_kind": slot_kind, "wt_pos": wt_pos, "mut_pos": mut_pos,
        "anchor_rel_coord": anchor_rel_coord, "insertion_rank": insertion_rank,
        "meta_raw": meta_raw, "label": labels, "var_id": var_ids,
    }


def make_collate_fn(model_mode: str):
    def _fn(batch: list[dict]) -> dict:
        return stage1_collate(batch, model_mode)
    return _fn
