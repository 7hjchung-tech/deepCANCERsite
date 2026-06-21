"""Distance metric for layer-representation convergence analysis.

Only Mahalanobis distance is provided.  The covariance inverse (Sigma_inv)
must be pre-computed by the caller so it can be reused across many (h_l, h_norm)
pairs without repeated matrix inversions.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mahalanobis(h_l: Tensor, h_norm: Tensor, Sigma_inv: Tensor) -> float:
    """Mahalanobis distance between a layer-l pooled vector and the layer-33 reference.

    D = sqrt( max(0, diff^T @ Sigma_inv @ diff) )

    The radicand is clamped to zero before the square root to guard against
    small negative values caused by floating-point rounding in Sigma_inv.

    Args:
        h_l:       (D,) pooled MUT representation at layer l.
        h_norm:    (D,) pooled MUT representation at layer 33 (reference).
        Sigma_inv: (D, D) pre-computed inverse covariance matrix for layer l.
                   Compute once per layer and reuse across epochs / variants.

    Returns:
        Scalar float distance ≥ 0.
    """
    diff = h_l.float() - h_norm.float()
    val = diff @ Sigma_inv.float() @ diff
    return float(val.clamp(min=0.0).sqrt())
