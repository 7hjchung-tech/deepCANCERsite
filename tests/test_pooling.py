"""Tests for diff_pooling.py.

All tests use synthetic torch.randn tensors — no ESM model loading required.

Coverage:
  (a) uniform mode equals manual slice mean
  (b) exp_decay with τ=1e6 converges to uniform (numerical stability)
  (c) boundary positions (p=0, p=L-1) do not raise and return correct shape
  (d) diff_pool output shape is (D,) for both modes
"""

import torch
import pytest

from src.embeddings.diff_pooling import window_pool, diff_pool

L, D = 50, 1280
torch.manual_seed(42)
H = torch.randn(L, D)   # fixed tensor reused across (a) and (b)


# ── (a) uniform equals manual mean ──────────────────────────────────────────

def test_uniform_equals_mean():
    p, W = 25, 5
    lo, hi = p - W, p + W + 1       # no boundary clamp needed here (25±5 in [0,50))
    expected = H[lo:hi].mean(dim=0)
    result   = window_pool(H, p, W, mode="uniform")
    assert torch.allclose(result, expected, atol=1e-6)


# ── (b) exp_decay(τ→∞) converges to uniform ─────────────────────────────────

def test_exp_decay_converges_to_uniform():
    p, W  = 25, 5
    uniform = window_pool(H, p, W, mode="uniform")
    decay   = window_pool(H, p, W, mode="exp_decay", tau=1e6)
    # With τ=1e6 and W=5, max weight deviation from 1/(2W+1) is < 1e-5
    assert torch.allclose(decay, uniform, atol=1e-4), (
        f"max abs diff: {(decay - uniform).abs().max().item():.2e}"
    )


# ── (c) boundary handling ────────────────────────────────────────────────────

@pytest.mark.parametrize("p,W", [
    (0,  5),     # left boundary — window clipped to [0, 6)
    (0,  10),    # left boundary, wider window
    (L-1, 5),    # right boundary — window clipped to [44, 50)
    (L-1, 10),   # right boundary, wider window
    (1,  5),     # near-left (lo clamps to 0)
    (L-2, 5),    # near-right (hi clamps to L)
])
def test_boundary_uniform(p, W):
    result = window_pool(H, p, W, mode="uniform")
    assert result.shape == (D,), f"shape {result.shape} for p={p}, W={W}"


@pytest.mark.parametrize("p,W", [
    (0,   5),
    (L-1, 5),
])
def test_boundary_exp_decay(p, W):
    result = window_pool(H, p, W, mode="exp_decay", tau=3.0)
    assert result.shape == (D,)


# ── (d) diff_pool output shape ───────────────────────────────────────────────

def test_diff_shape_uniform():
    torch.manual_seed(0)
    H_wt  = torch.randn(L, D)
    H_mut = torch.randn(L, D)
    cfg   = {"window_W": 5, "mode": "uniform"}
    diff  = diff_pool(H_wt, H_mut, p=20, cfg=cfg)
    assert diff.shape == (D,), f"expected ({D},), got {diff.shape}"


def test_diff_shape_exp_decay():
    torch.manual_seed(1)
    H_wt  = torch.randn(L, D)
    H_mut = torch.randn(L, D)
    cfg   = {"window_W": 5, "mode": "exp_decay", "tau": 3.0}
    diff  = diff_pool(H_wt, H_mut, p=20, cfg=cfg)
    assert diff.shape == (D,), f"expected ({D},), got {diff.shape}"


# ── extra: invalid mode raises ───────────────────────────────────────────────

def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown pooling mode"):
        window_pool(H, p=10, W=5, mode="gaussian")
