"""DiffEmbedder -- WT-difference embedding, cached or end-to-end.

mode="cached":
    Looks up precomputed {"diff": (1280,), "mag": (4,)} per var_id from a
    .pt file produced offline by dump_diff_emb.py. ESM-2 is never loaded --
    this path is for a frozen backbone (M1/M2), where the embedding never
    changes during training, so precomputing once is strictly cheaper.

mode="e2e":
    Holds a live ESMEncoder (optionally LoRA-injected) and computes WT/MUT
    hidden states + diff/mag on every forward call. Required whenever the
    encoder has trainable parameters (M3/M4), since a cached lookup can't
    backprop into LoRA weights.

Both modes expose the same forward(var_ids) -> {"diff": (B,1280), "mag": (B,4)}
contract, so downstream code (BottleneckMLP + MagMLP) doesn't care which
path produced the tensors.

WT-broadcast optimisation (mode="e2e" only): WT is the SAME sequence for
every variant in the dataset, so it is forwarded through the encoder exactly
ONCE per batch and reused for every MUT in that batch, regardless of batch
size or how many distinct MUT lengths (indels) are present. A batch of 8
variants pushes 1 (WT) + 8 (MUT) = 9 sequences through the model, not 16
(the naive per-pair (WT, MUT) approach would forward WT redundantly 8 times).
MUT sequences are grouped by length before batching, since ESM-2 forward
passes in this codebase assume equal-length sequences within one call
(indels change sequence length; missense/synonymous never do).

diff/mag computation itself (window pooling + magnitude scalars) is shared
with the cached path via diff_compute.compute_diff_and_mag -- otherwise a
cached-encoder model and an end-to-end LoRA model would not be comparable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .diff_compute import MAG_DIM, compute_diff_and_mag
from .esm_encoder import ESMEncoder

DIFF_DIM = 1280


class DiffEmbedder(nn.Module):
    def __init__(
        self,
        mode: str,
        pooling_cfg: dict,
        cache_path: Optional[str] = None,
        manifest: Optional[dict] = None,
        wt_seq: Optional[str] = None,
        esm_cfg: Optional[dict] = None,
        lora_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if mode not in ("cached", "e2e"):
            raise ValueError(f"mode must be 'cached' or 'e2e', got {mode!r}")
        self.mode = mode
        self.pooling_cfg = pooling_cfg
        self._layer = int(esm_cfg["repr_layer"]) if esm_cfg else None

        if mode == "cached":
            assert cache_path is not None, "mode='cached' requires cache_path"
            self._cache: dict = torch.load(cache_path, weights_only=True)
            self.encoder = None
        else:  # e2e
            assert manifest is not None and wt_seq is not None and esm_cfg is not None, (
                "mode='e2e' requires manifest, wt_seq, esm_cfg"
            )
            self.manifest = manifest
            self.wt_seq = wt_seq
            # ESMEncoder is a plain object (not nn.Module) holding the ESM2
            # nn.Module in .model; register it so its parameters (incl. LoRA)
            # show up in this module's .parameters() / state_dict().
            self.encoder = ESMEncoder(esm_cfg, lora_cfg)
            self.add_module("_esm_model", self.encoder.model)

    def forward(self, var_ids: list[str]) -> dict[str, Tensor]:
        if self.mode == "cached":
            return self._forward_cached(var_ids)
        return self._forward_e2e(var_ids)

    # ------------------------------------------------------------------
    def _forward_cached(self, var_ids: list[str]) -> dict[str, Tensor]:
        diffs = torch.stack([self._cache[v]["diff"] for v in var_ids])
        mags = torch.stack([self._cache[v]["mag"] for v in var_ids])
        return {"diff": diffs, "mag": mags}

    # ------------------------------------------------------------------
    def _forward_e2e(self, var_ids: list[str]) -> dict[str, Tensor]:
        rows = [self.manifest[v] for v in var_ids]
        positions = [int(r["pp"]) - 1 for r in rows]   # 1-based pp -> 0-based residue index
        mut_seqs = [r["mut_seq"] for r in rows]

        # ---- WT forwarded exactly ONCE per batch, no matter the batch size ----
        H_wt = self.encoder.encode_grad([self.wt_seq], repr_layers=self._layer)[self._layer][0]

        # Group MUT sequences by length (indels change length; a single
        # forward call needs equal-length inputs). WT is NOT recomputed here.
        by_len: dict[int, list[int]] = {}
        for i, s in enumerate(mut_seqs):
            by_len.setdefault(len(s), []).append(i)

        diffs: list[Optional[Tensor]] = [None] * len(var_ids)
        mags: list[Optional[Tensor]] = [None] * len(var_ids)
        for _, idxs in by_len.items():
            seqs = [mut_seqs[i] for i in idxs]
            H_mut_batch = self.encoder.encode_grad(seqs, repr_layers=self._layer)[self._layer]
            for local_i, global_i in enumerate(idxs):
                out = compute_diff_and_mag(
                    H_wt, H_mut_batch[local_i], positions[global_i], self.pooling_cfg
                )
                diffs[global_i] = out["diff"]
                mags[global_i] = out["mag"]

        return {"diff": torch.stack(diffs), "mag": torch.stack(mags)}
