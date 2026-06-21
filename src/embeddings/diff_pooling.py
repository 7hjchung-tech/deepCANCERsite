"""Window pooling and WT-difference computation.

Computes a difference vector  diff = pooled_MUT − pooled_WT
from per-residue ESM-2 embeddings using a symmetric window around the
mutation site.

Two pooling modes:
  "uniform"   — equal-weight mean of residues in [p-W, p+W] ∩ [0, L).
  "exp_decay" — exponentially decaying weights w_i = exp(-|i-p|/τ),
                normalised to sum=1.  As τ→∞, converges to uniform.

Mutation position p is 0-indexed (Python residue index).
Window radius W is a config hyperparameter; sweep candidates: [5, 10, 20].
"""

from __future__ import annotations

import torch
from torch import Tensor


def window_pool(
    H: Tensor,
    p: int,
    W: int,
    mode: str,
    tau: float = 1.0,
) -> Tensor:
    """Pool residues in the window [p-W, p+W] ∩ [0, L).

    Args:
        H:    (L, D) per-residue embeddings (BOS/EOS already stripped).
        p:    Mutation position, 0-indexed.
        W:    Window half-width. Sweep candidates: [5, 10, 20].
        mode: "uniform" | "exp_decay".
        tau:  Decay constant (exp_decay only). Large τ (e.g. 1e6) → uniform.

    Returns:
        (D,) pooled vector.
    """
    L = H.shape[0]
    lo = max(0, p - W)        # clamp to sequence start
    hi = min(L, p + W + 1)   # clamp to sequence end (exclusive)

    H_win = H[lo:hi]          # (win_len, D),  win_len ≤ 2W+1

    if mode == "uniform":
        return H_win.mean(dim=0)

    if mode == "exp_decay":
        indices = torch.arange(lo, hi, device=H.device, dtype=torch.float32)
        dists   = torch.abs(indices - float(p))
        # When τ is very large, dists/τ → 0 for all i → exp(·) → 1 uniformly
        weights = torch.exp(-dists / tau)
        weights = weights / weights.sum()                    # normalise, sum=1
        return (H_win * weights.unsqueeze(1)).sum(dim=0)

    raise ValueError(
        f"Unknown pooling mode '{mode}'. Expected 'uniform' or 'exp_decay'."
    )


def diff_pool(
    H_wt:  Tensor,
    H_mut: Tensor,
    p:     int,
    cfg:   dict,
) -> Tensor:
    """Compute pooled_MUT − pooled_WT around the mutation site.

    Args:
        H_wt:  (L, D) WT per-residue embeddings (BOS/EOS stripped).
        H_mut: (L, D) MUT per-residue embeddings.
        p:     Mutation position, 0-indexed.
        cfg:   config["pooling"] dict.
               Required key : window_W (int)
               Optional keys: mode (str, default "uniform"),
                              tau  (float, default 1.0, exp_decay only)

    Returns:
        (D,) difference vector.
    """
    W    = int(cfg["window_W"])
    mode = str(cfg.get("mode", "uniform"))
    tau  = float(cfg.get("tau", 1.0))

    pooled_wt  = window_pool(H_wt,  p, W, mode, tau)
    pooled_mut = window_pool(H_mut, p, W, mode, tau)
    return pooled_mut - pooled_wt
