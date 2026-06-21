"""Tests for src/analysis/covariance.py.

Coverage:
  - N < D (underdetermined): Ledoit-Wolf still produces invertible, finite Sigma
  - Sigma_inv shape is correct (full-dim and PCA-reduced)
  - Sigma_inv is symmetric
  - freeze_at="pretrained": second fit() is a no-op (same Sigma_inv returned)
  - freeze_at="per_epoch": second fit() updates Sigma_inv
  - get_inv() before fit() raises RuntimeError
"""

import torch
import pytest

from src.analysis.covariance import estimate_layer_covariance, CovarianceStore


def test_underdetermined_invertible():
    """N=100 < D=1280: Ledoit-Wolf shrinkage should produce a finite inverse."""
    torch.manual_seed(42)
    X = torch.randn(100, 1280)
    Sigma_inv, pca = estimate_layer_covariance(X)
    assert pca is None
    assert Sigma_inv.shape == (1280, 1280)
    assert torch.isfinite(Sigma_inv).all(), "Sigma_inv contains non-finite values"


def test_sigma_inv_shape_full_dim():
    torch.manual_seed(0)
    X = torch.randn(50, 64)
    Sigma_inv, _ = estimate_layer_covariance(X)
    assert Sigma_inv.shape == (64, 64)


def test_sigma_inv_symmetric():
    torch.manual_seed(0)
    X = torch.randn(50, 64)
    Sigma_inv, _ = estimate_layer_covariance(X)
    assert torch.allclose(Sigma_inv, Sigma_inv.T, atol=1e-4)


def test_pca_dim_shape():
    """With pca_dim=32, Sigma_inv should be (32, 32) and pca object returned."""
    torch.manual_seed(0)
    X = torch.randn(50, 1280)
    Sigma_inv, pca = estimate_layer_covariance(X, pca_dim=32)
    assert Sigma_inv.shape == (32, 32)
    assert pca is not None


def test_unknown_estimator_raises():
    X = torch.randn(10, 8)
    with pytest.raises(ValueError, match="Unknown estimator"):
        estimate_layer_covariance(X, estimator="bogus")


def test_cov_store_pretrained_freeze():
    """Second fit() with freeze_at='pretrained' must return the same Sigma_inv."""
    torch.manual_seed(0)
    cfg = {"freeze_at": "pretrained", "estimator": "ledoit_wolf"}
    store = CovarianceStore(cfg)

    reps_1 = {0: torch.randn(50, 32)}
    reps_2 = {0: torch.randn(50, 32) * 100.0}   # very different

    store.fit(reps_1)
    inv_first = store.get_inv(0).clone()

    store.fit(reps_2)                             # must be no-op
    inv_second = store.get_inv(0)

    assert torch.allclose(inv_first, inv_second), (
        "pretrained freeze: second fit() should not change Sigma_inv"
    )


def test_cov_store_per_epoch_refits():
    """Second fit() with freeze_at='per_epoch' must update Sigma_inv."""
    torch.manual_seed(0)
    cfg = {"freeze_at": "per_epoch", "estimator": "ledoit_wolf"}
    store = CovarianceStore(cfg)

    store.fit({0: torch.randn(50, 32)})
    inv_1 = store.get_inv(0).clone()

    store.fit({0: torch.randn(50, 32) * 10.0})
    inv_2 = store.get_inv(0)

    assert not torch.allclose(inv_1, inv_2), (
        "per_epoch: second fit() should update Sigma_inv"
    )


def test_cov_store_not_fitted_raises():
    store = CovarianceStore({"freeze_at": "pretrained"})
    with pytest.raises(RuntimeError, match="not been fitted"):
        store.get_inv(0)


def test_cov_store_is_fitted_flag():
    store = CovarianceStore({"freeze_at": "pretrained"})
    assert not store.is_fitted()
    store.fit({0: torch.randn(20, 32)})
    assert store.is_fitted()
