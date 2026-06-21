"""Tests for bottleneck.py.

No ESM model loading required — tests only the MLP module itself.
"""

import torch
import pytest

from src.embeddings.bottleneck import BottleneckMLP

CFG = {"in_dim": 1280, "out_dim": 256}


def test_output_shape_1d():
    model = BottleneckMLP(CFG)
    x     = torch.randn(1280)
    out   = model(x)
    assert out.shape == (256,), f"expected (256,), got {out.shape}"


def test_output_shape_batched():
    model = BottleneckMLP(CFG)
    x     = torch.randn(4, 1280)
    out   = model(x)
    assert out.shape == (4, 256), f"expected (4, 256), got {out.shape}"


def test_output_dtype():
    model = BottleneckMLP(CFG)
    x     = torch.randn(1280)
    out   = model(x)
    assert out.dtype == torch.float32


def test_custom_dims():
    model = BottleneckMLP({"in_dim": 512, "out_dim": 64})
    x     = torch.randn(512)
    out   = model(x)
    assert out.shape == (64,)


def test_gradient_flows():
    """Sanity check: gradients reach the linear weight in a training scenario."""
    model = BottleneckMLP(CFG)
    x     = torch.randn(1280)
    loss  = model(x).sum()
    loss.backward()
    assert model.net[0].weight.grad is not None
