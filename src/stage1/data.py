"""Bind the validated frozen representation cache to the manifest, for Stage 1.

RESPONSIBILITIES
----------------
1. Verify the cache's provenance BEFORE anything trains on it.
2. Map rows by `var_id`. Never by position -- the cache row order is the export
   order, which is not the manifest's row order and is not a contract.
3. Restrict to the validated `in_eval_scope=True` cohort, and prove that the
   cache contains exactly that cohort by recomputing it from Task B's own
   `build_variant_record`.
4. Preserve the shipped train/val/test assignment, and cross-check the split
   stored in the cache against the manifest's.
5. Keep targets OUT of the feature path: `Stage1Batch` never carries a label,
   and the target column is read from the manifest into a separate object.

WHAT IT DOES NOT DO
-------------------
It does not modify, re-window, re-align or re-pool anything. The cache's
alignment, window rule, layer set and slot semantics are Task C's and are used
verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..embeddings import representation_cache as rc
from ..embeddings.variant_map import SPLIT_RULE_VERSION, build_variant_record
from .interface import Stage1Batch, TargetBatch
from .targets import DEFAULT_TARGET_COLUMN, TargetScaler

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = _ROOT / "data" / "esm_repr_v1" / "frozen_repr_v1.pt"
DEFAULT_MANIFEST = _ROOT / "data" / "split_manifest.csv"
DEFAULT_WT = _ROOT / "data" / "wt_sequence.txt"

SPLITS: Tuple[str, ...] = ("train", "val", "test")


class ProvenanceError(rc.StaleCacheError):
    """The frozen cache does not match the code/manifest it claims."""


class CohortError(RuntimeError):
    """The cache's row set is not the validated in_eval_scope cohort."""


# ==========================================================================
# provenance
# ==========================================================================
def verify_cache_provenance(
    prov: Dict[str, Any],
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    n_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Refuse a cache that was not built by this code from this manifest.

    Checked here rather than trusted, because every one of these silently
    changes what the numbers mean:

      * the recorded provenance hash actually matches its own contents;
      * the manifest file and the split assignment are byte-identical to the
        ones the cache was exported from;
      * `variant_map.py` -- which decides the cohort and the alignment -- has
        not changed since the export;
      * the model, layers, precisions, adapter state, window rule and split
        rule are the ones this code expects;
      * the cache declares that it holds no targets.

    Returns the verified provenance. Raises ProvenanceError naming every field
    that disagrees.
    """
    bad: Dict[str, Dict[str, Any]] = {}

    def want(field: str, expected: Any) -> None:
        found = prov.get(field, "<missing>")
        if found != expected:
            bad[field] = {"cache": found, "expected": expected}

    recomputed = rc.provenance_hash(prov)
    if prov.get("provenance_hash") != recomputed:
        bad["provenance_hash"] = {"cache": prov.get("provenance_hash"), "expected": recomputed}

    manifest_path = Path(manifest_path)
    want("manifest_hash", rc.file_sha256(manifest_path))
    split_series = pd.read_csv(manifest_path, usecols=["var_id", "split"])
    import hashlib

    want(
        "split_schema_hash",
        hashlib.sha256(
            split_series.sort_values("var_id").to_csv(index=False).encode()
        ).hexdigest(),
    )
    want("variant_map_hash", rc.code_hash(["src/embeddings/variant_map.py"])[
        "src/embeddings/variant_map.py"
    ])

    want("backend", rc.BACKEND)
    want("model_name", rc.MODEL_NAME)
    want("repr_layers", list(rc.REPR_LAYERS))
    want("layer_convention", rc.LAYER_CONVENTION)
    want("adapter", "none")
    want("adapter_state", "frozen")
    want("forward_precision", "torch.float32")
    want("cache_precision", "torch.float32")
    want("delta_precision", "torch.float32")
    want("window_rule", rc.WINDOW_RULE)
    want("window_rule_version", rc.WINDOW_RULE_VERSION)
    want("alignment_version", rc.ALIGNMENT_VERSION)
    want("split_rule_version", SPLIT_RULE_VERSION)
    want("contains_targets", False)

    if prov.get("base_checkpoint_hash") in (None, "", "missing"):
        bad["base_checkpoint_hash"] = {
            "cache": prov.get("base_checkpoint_hash"), "expected": "a real sha256"
        }
    if n_rows is not None and int(prov.get("n_rows", -1)) != int(n_rows):
        bad["n_rows"] = {"cache": prov.get("n_rows"), "expected": n_rows}

    if bad:
        import json

        raise ProvenanceError(
            "frozen representation cache failed provenance verification in "
            f"{len(bad)} field(s): {sorted(bad)}\n"
            + json.dumps(bad, indent=2, default=str)
        )
    return prov


def in_eval_scope_var_ids(manifest_df: pd.DataFrame, wt_seq: str) -> List[str]:
    """Recompute the validated cohort from Task B's own record builder.

    This is the authority on scope. It is recomputed rather than read from a
    generated CSV so that a stale audit artifact cannot widen the cohort.
    """
    keep: List[str] = []
    for _, raw in manifest_df.iterrows():
        rec = build_variant_record(raw.to_dict(), wt_seq, manifest_df=manifest_df)
        if rec["in_eval_scope"]:
            keep.append(str(rec["var_id"]))
    return keep


# ==========================================================================
# cohort binding
# ==========================================================================
@dataclass
class CohortReport:
    n_cache_rows: int
    n_manifest_rows: int
    n_in_scope: int
    cohort_verified: bool
    rows_by_split: Dict[str, int]
    target_column: str
    n_targets_present: int


class FrozenReprData:
    """The frozen cache, bound to the manifest by var_id.

    `cache_index_of[var_id]` is the single source of row identity. Split
    membership, targets and variant type all come from the manifest keyed by
    the same id, so nothing anywhere depends on row order.
    """

    def __init__(
        self,
        cache: rc.RepresentationCache,
        manifest_df: pd.DataFrame,
        *,
        wt_seq: Optional[str] = None,
        target_column: str = DEFAULT_TARGET_COLUMN,
        verify_cohort: bool = True,
        manifest_path: Path = DEFAULT_MANIFEST,
        verify_provenance: bool = True,
    ) -> None:
        self.cache = cache
        self.manifest = manifest_df.reset_index(drop=True)
        self.target_column = target_column

        if verify_provenance:
            verify_cache_provenance(
                cache.provenance, manifest_path=manifest_path, n_rows=len(cache)
            )

        cache_ids = list(cache.var_id)
        if len(set(cache_ids)) != len(cache_ids):
            dupes = sorted({v for v in cache_ids if cache_ids.count(v) > 1})[:5]
            raise CohortError(f"cache contains duplicate var_ids, e.g. {dupes}")
        self.cache_index_of: Dict[str, int] = {v: i for i, v in enumerate(cache_ids)}

        man_ids = self.manifest["var_id"].astype(str).tolist()
        if len(set(man_ids)) != len(man_ids):
            raise CohortError("manifest contains duplicate var_ids")
        self._man_row: Dict[str, int] = {v: i for i, v in enumerate(man_ids)}

        missing = [v for v in cache_ids if v not in self._man_row]
        if missing:
            raise CohortError(
                f"{len(missing)} cache var_ids are absent from the manifest, "
                f"e.g. {missing[:5]}"
            )

        # split: manifest is authoritative, the cache's copy must agree
        man_split = self.manifest["split"].astype(str).tolist()
        mismatched = [
            (v, cache.split[i], man_split[self._man_row[v]])
            for i, v in enumerate(cache_ids)
            if str(cache.split[i]) != man_split[self._man_row[v]]
        ]
        if mismatched:
            raise CohortError(
                f"{len(mismatched)} rows disagree on split between cache and "
                f"manifest, e.g. {mismatched[:3]}"
            )
        self.split_of: Dict[str, str] = {
            v: man_split[self._man_row[v]] for v in cache_ids
        }

        self.cohort_verified = False
        if verify_cohort:
            if wt_seq is None:
                wt_seq = DEFAULT_WT.read_text(encoding="utf-8").strip()
            expected = set(in_eval_scope_var_ids(self.manifest, wt_seq))
            have = set(cache_ids)
            if expected != have:
                only_scope = sorted(expected - have)[:5]
                only_cache = sorted(have - expected)[:5]
                raise CohortError(
                    "the cache is not the validated in_eval_scope cohort: "
                    f"{len(expected - have)} in-scope rows missing (e.g. {only_scope}), "
                    f"{len(have - expected)} extra rows present (e.g. {only_cache})"
                )
            self.cohort_verified = True

        # targets, read from the manifest and kept out of the feature path
        if target_column not in self.manifest.columns:
            raise KeyError(f"target column {target_column!r} is not in the manifest")
        y = self.manifest[target_column].to_numpy(dtype=np.float64)
        self._y_by_row = np.asarray([y[self._man_row[v]] for v in cache_ids])

        self.variant_type: List[str] = [str(v) for v in cache.p["variant_type"]]
        self.edit_type: List[str] = [str(v) for v in cache.p["edit_type"]]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.cache)

    @property
    def var_id(self) -> List[str]:
        return list(self.cache.var_id)

    def indices_for_split(self, split: str) -> List[int]:
        """Cache row indices belonging to one split, in cache order."""
        if split not in SPLITS:
            raise KeyError(f"unknown split {split!r}; have {SPLITS}")
        return [i for i, v in enumerate(self.cache.var_id) if self.split_of[v] == split]

    def targets_raw(self, indices: Sequence[int]) -> np.ndarray:
        return self._y_by_row[np.asarray(list(indices), dtype=np.int64)]

    def fit_target_scaler(self, *, split: str = "train") -> TargetScaler:
        """Fit mu/sigma on ONE split's rows -- 'train' unless deliberately told
        otherwise, and the split used is recorded on the scaler itself."""
        idx = self.indices_for_split(split)
        return TargetScaler.fit_on_train(
            self.targets_raw(idx), target_column=self.target_column, fit_split=split
        )

    def report(self) -> CohortReport:
        return CohortReport(
            n_cache_rows=len(self.cache),
            n_manifest_rows=len(self.manifest),
            n_in_scope=len(self.cache),
            cohort_verified=self.cohort_verified,
            rows_by_split={s: len(self.indices_for_split(s)) for s in SPLITS},
            target_column=self.target_column,
            n_targets_present=int(np.isfinite(self._y_by_row).sum()),
        )

    # ------------------------------------------------------------------
    def make_batch(
        self, indices: Sequence[int], scaler: Optional[TargetScaler] = None
    ) -> Tuple[Stage1Batch, TargetBatch]:
        """Materialise one batch: (features, targets) as two separate objects."""
        idx = list(indices)
        b = self.cache.batch(idx)
        ids = [self.cache.var_id[i] for i in idx]

        features = Stage1Batch(
            H_WT=b["H_WT"],
            H_MUT=b["H_MUT"],
            delta_H=b["delta_H"],
            wt_pos=b["wt_pos"],
            mut_pos=b["mut_pos"],
            wt_present=b["wt_present"],
            mut_present=b["mut_present"],
            delta_valid=b["delta_valid"],
            token_valid=b["token_valid"],
            slot_kind=b["slot_kind"],
            var_id=ids,
            split=[self.split_of[v] for v in ids],
            variant_type=[self.variant_type[i] for i in idx],
            edit_type=[self.edit_type[i] for i in idx],
            pp=[self.cache.p["pp"][i] for i in idx],
            slot_kind_vocab=list(self.cache.slot_kind_vocab),
            layers=list(self.cache.layers),
        )

        y_raw = self.targets_raw(idx)
        targets = TargetBatch(
            var_id=list(ids),
            y_raw=torch.tensor(y_raw, dtype=torch.float32),
            y_std=(
                None if scaler is None
                else torch.tensor(scaler.transform(y_raw), dtype=torch.float32)
            ),
            target_column=self.target_column,
            scaler=None if scaler is None else scaler.to_dict(),
        )
        return features, targets


# ==========================================================================
# torch Dataset / DataLoader
# ==========================================================================
class Stage1Dataset(Dataset):
    """Indices into one split. The collate does the real work."""

    def __init__(self, data: FrozenReprData, split: str) -> None:
        self.data = data
        self.split = split
        self.indices = data.indices_for_split(split)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> int:
        return self.indices[i]


def make_collate(data: FrozenReprData, scaler: Optional[TargetScaler] = None):
    def collate(batch_indices: List[int]) -> Tuple[Stage1Batch, TargetBatch]:
        return data.make_batch(batch_indices, scaler=scaler)

    return collate


def make_loader(
    data: FrozenReprData,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    scaler: Optional[TargetScaler] = None,
    generator: Optional[torch.Generator] = None,
    drop_last: bool = False,
) -> DataLoader:
    """A DataLoader yielding (Stage1Batch, TargetBatch) pairs.

    num_workers is deliberately 0: the whole cache is one in-memory tensor set,
    so workers would fork ~1 GB per worker to do index_select.
    """
    return DataLoader(
        Stage1Dataset(data, split),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=drop_last,
        num_workers=0,
        collate_fn=make_collate(data, scaler),
    )


def load_frozen_data(
    *,
    cache_path: Path = DEFAULT_CACHE,
    manifest_path: Path = DEFAULT_MANIFEST,
    wt_path: Path = DEFAULT_WT,
    target_column: str = DEFAULT_TARGET_COLUMN,
    verify_cohort: bool = True,
) -> FrozenReprData:
    """Load + verify + bind, in one call."""
    cache_path, manifest_path = Path(cache_path), Path(manifest_path)
    prov_path = cache_path.with_suffix(".provenance.json")
    expected = None
    if prov_path.exists():
        import json

        expected = json.loads(prov_path.read_text())
    cache = rc.load_cache(cache_path, expected_provenance=expected, strict=True)
    manifest = pd.read_csv(manifest_path)
    return FrozenReprData(
        cache,
        manifest,
        wt_seq=Path(wt_path).read_text(encoding="utf-8").strip(),
        target_column=target_column,
        verify_cohort=verify_cohort,
        manifest_path=manifest_path,
    )
