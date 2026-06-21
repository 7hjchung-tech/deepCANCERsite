"""Tests for src/analysis/layer_metrics.py.

Coverage:
  - Same vector → distance = 0
  - Sigma_inv = I → equals Euclidean distance
  - Non-negative for arbitrary inputs
  - Scale invariance: Sigma_inv scaled by k → distance scaled by sqrt(k)
  - Negative radicand (pathological Sigma_inv) is clamped to 0.0
"""

import torch
import pytest

from src.analysis.layer_metrics import mahalanobis


D = 16


def test_same_vector_is_zero():
    h = torch.randn(D)
    Sigma_inv = torch.eye(D)
    assert mahalanobis(h, h, Sigma_inv) == pytest.approx(0.0, abs=1e-6)


def test_identity_sigma_equals_euclidean():
    torch.manual_seed(0)
    h_l    = torch.randn(D)
    h_norm = torch.randn(D)
    Sigma_inv = torch.eye(D)
    expected = float(torch.dist(h_l, h_norm))
    got = mahalanobis(h_l, h_norm, Sigma_inv)
    assert got == pytest.approx(expected, abs=1e-5)


def test_nonnegative_for_arbitrary_input():
    torch.manual_seed(1)
    h_l    = torch.randn(D)
    h_norm = torch.randn(D)
    A = torch.randn(D, D)
    Sigma_inv = A @ A.T + torch.eye(D) * 0.1   # PSD
    assert mahalanobis(h_l, h_norm, Sigma_inv) >= 0.0


def test_scale_invariance():
    """Scaling Sigma_inv by k should scale the distance by sqrt(k)."""
    torch.manual_seed(2)
    h_l    = torch.randn(D)
    h_norm = torch.zeros(D)
    Sigma_inv = torch.eye(D)
    d1 = mahalanobis(h_l, h_norm, Sigma_inv)
    d4 = mahalanobis(h_l, h_norm, 4.0 * Sigma_inv)
    assert d4 == pytest.approx(2.0 * d1, abs=1e-5)


def test_negative_radicand_clamped():
    """Pathological (negative-definite) Sigma_inv must not cause NaN or errors."""
    h_l    = torch.tensor([1.0, 0.0, 0.0, 0.0])
    h_norm = torch.zeros(4)
    # diff^T @ (-I) @ diff = -‖diff‖² < 0  → should clamp to 0
    Sigma_inv = -torch.eye(4)
    result = mahalanobis(h_l, h_norm, Sigma_inv)
    assert result == pytest.approx(0.0, abs=1e-6)
    assert result >= 0.0
