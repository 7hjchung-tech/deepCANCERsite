"""Layer-wise covariance estimation with Ledoit-Wolf shrinkage.

Provides estimate_layer_covariance() for a single layer and CovarianceStore
for managing per-layer Sigma^{-1} caches across the full ESM-2 backbone.

Freeze policy
─────────────
freeze_at="pretrained" (default):
    Sigma_l is estimated once from activations of the untuned (LoRA B=0) model.
    All subsequent epochs share the same metric space — Mahalanobis distances
    are directly comparable across epochs.

freeze_at="per_epoch":
    Sigma_l is re-estimated at each snapshot() call.  Because the metric space
    shifts with every update, cross-epoch distance comparisons lose their
    absolute meaning; only relative within-epoch rankings are interpretable.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def estimate_layer_covariance(
    per_residue_reps: Tensor,
    estimator: str = "ledoit_wolf",
    pca_dim: Optional[int] = None,
    cond_warn_threshold: float = 1e8,
) -> tuple[Tensor, Any]:
    """Estimate Sigma_l from per-residue representations and return its inverse.

    Args:
        per_residue_reps:    (N, D) float tensor — all residue representations
                             collected from reference sequences at one layer.
        estimator:           Shrinkage method.  Only "ledoit_wolf" is supported.
        pca_dim:             If set, apply PCA to reduce to pca_dim dimensions
                             before covariance estimation.  The returned
                             Sigma_inv has shape (pca_dim, pca_dim).
                             None → full D-dimensional covariance.
        cond_warn_threshold: Emit a RuntimeWarning when cond(Sigma) exceeds
                             this value (ill-conditioned → unreliable distances).

    Returns:
        (Sigma_inv, pca_or_None)
        Sigma_inv: Tensor of shape (D', D') where D' = pca_dim or D.
        pca_or_None: fitted sklearn PCA object when pca_dim is set, else None.
    """
    from sklearn.covariance import LedoitWolf

    X = per_residue_reps.float().cpu().numpy()

    pca = None
    if pca_dim is not None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(X)

    if estimator == "ledoit_wolf":
        lw = LedoitWolf().fit(X)
        Sigma = lw.covariance_
    else:
        raise ValueError(f"Unknown estimator: {estimator!r}. Only 'ledoit_wolf' is supported.")

    cond = float(np.linalg.cond(Sigma))
    logger.debug("Layer covariance condition number: %.3e", cond)
    if cond > cond_warn_threshold:
        warnings.warn(
            f"Covariance condition number {cond:.2e} exceeds threshold "
            f"{cond_warn_threshold:.2e}. Mahalanobis distances may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    Sigma_inv = np.linalg.inv(Sigma)
    return torch.tensor(Sigma_inv, dtype=torch.float32), pca


class CovarianceStore:
    """Per-layer Sigma^{-1} cache with configurable freeze policy.

    Usage::

        store = CovarianceStore(cfg["cov"])
        store.fit({l: reps_l for l in range(34)})   # call once (pretrained) or per epoch
        Sigma_inv_5 = store.get_inv(5)               # retrieve layer 5 inverse

    With freeze_at="pretrained", the first fit() call is the only one that has
    any effect; subsequent calls are silently ignored so the same metric space
    is used across all epochs.
    """

    def __init__(self, cfg: dict) -> None:
        self._freeze_at: str = cfg.get("freeze_at", "pretrained")
        self._estimator: str = cfg.get("estimator", "ledoit_wolf")
        self._pca_dim: Optional[int] = cfg.get("pca_dim", None)
        self._cond_warn: float = float(cfg.get("cond_warn_threshold", 1e8))
        self._inv_cache: dict[int, Tensor] = {}
        self._fitted: bool = False

    def fit(self, per_layer_reps: dict[int, Tensor]) -> None:
        """Estimate and cache Sigma_l^{-1} for every layer key.

        With freeze_at="pretrained": no-op after the first successful fit.
        With freeze_at="per_epoch": replaces the cache on every call.
        """
        if self._fitted and self._freeze_at == "pretrained":
            return

        for layer_idx, reps in per_layer_reps.items():
            Sigma_inv, _ = estimate_layer_covariance(
                reps,
                estimator=self._estimator,
                pca_dim=self._pca_dim,
                cond_warn_threshold=self._cond_warn,
            )
            self._inv_cache[layer_idx] = Sigma_inv

        self._fitted = True

    def get_inv(self, layer_idx: int) -> Tensor:
        """Return the pre-computed Sigma^{-1} for the given layer index."""
        if not self._fitted:
            raise RuntimeError(
                "CovarianceStore has not been fitted. Call fit() before get_inv()."
            )
        return self._inv_cache[layer_idx]

    def is_fitted(self) -> bool:
        return self._fitted
