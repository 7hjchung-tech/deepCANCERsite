"""Tests for LoRA injection (1b).

Uses a small mock model — no ESM-2 weight download required.
The mock reproduces the attribute structure that inject_lora depends on:
    model.layers[i].self_attn.{q_proj, k_proj, v_proj}

Coverage:
  LoRALinear:
    - output shape is unchanged
    - at init (B=0) output equals base linear (no change to activations)
    - only lora_A / lora_B have requires_grad=True; base params are frozen
    - gradients reach lora_A and lora_B; base.weight.grad stays None
    - 1-D input works (diff_pool output → bottleneck scenario)
  inject_lora:
    - replaces q_proj and v_proj; k_proj is untouched
    - target_layers restricts injection to specified indices
    - base weights remain frozen after injection
    - trainable param count matches analytical expectation
"""

import torch
import torch.nn as nn
import pytest

from src.embeddings.lora import LoRALinear, inject_lora, lora_param_count


# ── helpers ──────────────────────────────────────────────────────────────────

DIM  = 1280
RANK = 8


def frozen_linear(in_f: int = DIM, out_f: int = DIM) -> nn.Linear:
    lin = nn.Linear(in_f, out_f)
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


class _FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DIM, DIM)
        self.k_proj = nn.Linear(DIM, DIM)
        self.v_proj = nn.Linear(DIM, DIM)


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttn()


class _FakeESM(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(num_layers)])


def freeze(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── LoRALinear unit tests ─────────────────────────────────────────────────────

def test_lora_output_shape_3d():
    """(T, B, D) → same shape; mirrors ESM-2 attention input."""
    lora = LoRALinear(frozen_linear(), rank=RANK, alpha=16.0)
    x    = torch.randn(10, 2, DIM)
    assert lora(x).shape == (10, 2, DIM)


def test_lora_output_shape_2d():
    lora = LoRALinear(frozen_linear(), rank=RANK, alpha=16.0)
    x    = torch.randn(4, DIM)
    assert lora(x).shape == (4, DIM)


def test_lora_output_shape_1d():
    """1-D input (diff_pool output) must work."""
    lora = LoRALinear(frozen_linear(), rank=RANK, alpha=16.0)
    x    = torch.randn(DIM)
    assert lora(x).shape == (DIM,)


def test_lora_equals_base_at_init():
    """B is zeroed at init → LoRA term is zero → output must match base."""
    lin  = frozen_linear()
    lora = LoRALinear(lin, rank=RANK, alpha=16.0)

    x        = torch.randn(5, 3, DIM)
    base_out = lin(x)
    lora_out = lora(x)

    assert torch.allclose(lora_out, base_out, atol=1e-6), (
        f"max diff at init: {(lora_out - base_out).abs().max().item():.2e}"
    )


def test_only_ab_trainable():
    lora      = LoRALinear(frozen_linear(), rank=RANK, alpha=16.0)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    frozen    = {n for n, p in lora.named_parameters() if not p.requires_grad}

    assert trainable == {"lora_A", "lora_B"}, f"unexpected trainable: {trainable}"
    assert "base.weight" in frozen
    assert "base.bias"   in frozen


def test_gradient_flows_to_ab():
    lin  = frozen_linear()
    lora = LoRALinear(lin, rank=RANK, alpha=16.0)

    # Non-zero B so the LoRA term produces a non-trivial gradient path
    with torch.no_grad():
        lora.lora_B.fill_(0.01)

    x = torch.randn(3, DIM)
    lora(x).sum().backward()

    assert lora.lora_A.grad is not None,    "lora_A should receive gradient"
    assert lora.lora_B.grad is not None,    "lora_B should receive gradient"
    assert lin.weight.grad is None,          "frozen base must not receive gradient"


# ── inject_lora tests ─────────────────────────────────────────────────────────

def test_inject_replaces_qv_not_k():
    model = freeze(_FakeESM(num_layers=3))
    inject_lora(model, rank=RANK, alpha=16.0)

    for layer in model.layers:
        assert isinstance(layer.self_attn.q_proj, LoRALinear), "q_proj not replaced"
        assert isinstance(layer.self_attn.v_proj, LoRALinear), "v_proj not replaced"
        assert isinstance(layer.self_attn.k_proj, nn.Linear),  "k_proj should be unchanged"


def test_inject_target_layers():
    model = freeze(_FakeESM(num_layers=4))
    inject_lora(model, rank=RANK, alpha=16.0, target_layers=[0, 2])

    assert isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
    assert isinstance(model.layers[2].self_attn.q_proj, LoRALinear)
    assert isinstance(model.layers[1].self_attn.q_proj, nn.Linear), "layer 1 untouched"
    assert isinstance(model.layers[3].self_attn.q_proj, nn.Linear), "layer 3 untouched"


def test_base_frozen_after_inject():
    model = freeze(_FakeESM(num_layers=2))
    inject_lora(model, rank=RANK, alpha=16.0)

    for layer in model.layers:
        assert not layer.self_attn.q_proj.base.weight.requires_grad
        assert not layer.self_attn.v_proj.base.weight.requires_grad


def test_param_count_analytical():
    """Trainable params = num_layers × 2 projections × 2 matrices × rank × dim."""
    num_layers = 4
    model = freeze(_FakeESM(num_layers=num_layers))
    inject_lora(model, rank=RANK, alpha=16.0)

    # Per LoRALinear: lora_A (RANK×DIM) + lora_B (DIM×RANK) = 2·RANK·DIM
    # Per layer: Q + V = 2 LoRALinears
    expected = num_layers * 2 * 2 * RANK * DIM
    got      = lora_param_count(model)
    assert got == expected, f"expected {expected:,}, got {got:,}"


def test_inject_output_unchanged_at_init():
    """End-to-end: after injection with B=0, forward output equals pre-injection."""
    torch.manual_seed(0)
    model = freeze(_FakeESM(num_layers=2))

    # Reference output before injection
    x_q = torch.randn(5, 2, DIM)
    ref = {
        i: model.layers[i].self_attn.q_proj(x_q).detach().clone()
        for i in range(2)
    }

    inject_lora(model, rank=RANK, alpha=16.0)

    for i in range(2):
        post = model.layers[i].self_attn.q_proj(x_q)
        assert torch.allclose(post, ref[i], atol=1e-6), (
            f"layer {i} output changed after LoRA injection (B should be 0)"
        )
