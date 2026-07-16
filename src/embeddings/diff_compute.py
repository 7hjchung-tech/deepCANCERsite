"""Shared WT-difference + magnitude computation.

Used by BOTH the cached path (dump_diff_emb.py, precomputes once offline)
and the end-to-end path (DiffEmbedder mode="e2e", computed every forward).
Keeping this in one place guarantees the two paths compute byte-for-byte
the same diff/magnitude definition -- otherwise a cached-encoder model
(M1/M2) and an end-to-end LoRA model (M3/M4) would not be comparable.

mag (4,) = [ ||diff||_2,                              overall magnitude
             1 - cosine(pooled_wt, pooled_mut),        directional shift
             diff.abs().max(),                         peak change
             diff.abs().mean() ]                       average change
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .diff_pooling import window_pool

MAG_DIM = 4


def compute_diff_and_mag(H_wt: Tensor, H_mut: Tensor, p: int, cfg: dict) -> dict[str, Tensor]:
    """Args:
        H_wt:  (L_wt, D) WT per-residue embeddings (BOS/EOS stripped).
        H_mut: (L_mut, D) MUT per-residue embeddings. L_mut may differ from
               L_wt for indels; window_pool clamps to each sequence's own length.
        p:     Mutation position, 0-indexed (same anchor used for both).
        cfg:   config["pooling"] dict (window_W, mode, tau).

    Returns {"diff": (D,), "mag": (MAG_DIM,)}.
    """
    W = int(cfg["window_W"])
    mode = str(cfg.get("mode", "uniform"))
    tau = float(cfg.get("tau", 1.0))

    pooled_wt = window_pool(H_wt, p, W, mode, tau)
    pooled_mut = window_pool(H_mut, p, W, mode, tau)
    diff = pooled_mut - pooled_wt

    l2 = diff.norm(p=2)
    cos = F.cosine_similarity(pooled_wt.unsqueeze(0), pooled_mut.unsqueeze(0)).squeeze(0)
    mag = torch.stack([l2, 1.0 - cos, diff.abs().max(), diff.abs().mean()])
    return {"diff": diff, "mag": mag}
