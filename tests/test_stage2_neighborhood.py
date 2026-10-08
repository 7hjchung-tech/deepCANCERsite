"""3D-neighborhood FiLM model: synthetic checks (no GPU, no real PDB/cache needed).

Covers §13 of the task spec that can be checked without real data: residue/field axis
separation, padding attention, epoch-0 == y_base, E2 gradient flow, E1's zero Q/K gradient
being the expected (not buggy) outcome, equal E1/E2 parameter counts, and identical-init-state
copying. WT<->PDB mapping and the real neighbor cache's self-exclusion/no-duplicate/tie-break
properties are checked directly against the real cache in test_stage2_neighborhood_real.py
(skipped if the cache has not been built).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage2.neighbor_model import NeighborhoodFiLMModel  # noqa: E402
from src.stage2.neighbor_structure import ResidueStructureTokenizer  # noqa: E402

D, B = 128, 3


def _tokenizer():
    import numpy as np
    tok = ResidueStructureTokenizer(d_s=32, n_bins=4)
    rng = np.random.default_rng(0)
    cont = rng.normal(size=(50, 8)).astype("float32")
    ss = rng.integers(0, 3, size=50)
    tok.fit_preprocessing({"continuous": torch.as_tensor(cont), "ss": torch.as_tensor(ss)})
    return tok


def _batch(mode: str, seed=0):
    g = torch.Generator().manual_seed(seed)
    h_base = torch.randn(B, D, generator=g)
    cont = torch.randn(B, 9, 8, generator=g)
    ss = torch.randint(0, 3, (B, 9), generator=g)
    distance = torch.rand(B, 9, generator=g) * 20
    distance[:, 0] = 0.0
    offset = torch.randint(-50, 50, (B, 9), generator=g)
    offset[:, 0] = 0
    is_anchor = torch.zeros(B, 9, dtype=torch.bool)
    is_anchor[:, 0] = True
    valid = torch.ones(B, 9, dtype=torch.bool)
    if mode == "e1":
        valid[:, 1:] = False
    y_base = torch.randn(B, generator=g)
    return {"continuous": cont, "ss": ss}, distance, offset, is_anchor, valid, h_base, y_base


def test_residue_and_field_axis_shapes():
    tok = _tokenizer()
    raw, *_ = _batch("e2")
    s = tok(raw)
    assert s.shape == (B, 9, 32)                 # residue axis kept, field axis reduced away


def test_padding_attention_zero_and_valid_weights_sum_to_one():
    tok = _tokenizer()
    raw, distance, offset, is_anchor, valid, h_base, y_base = _batch("e2")
    valid = valid.clone(); valid[1, 4:] = False   # extra padding on one sample
    s = tok(raw)
    m = NeighborhoodFiLMModel(d=D)
    out = m(h_base, s, distance, offset, is_anchor, valid, y_base)
    w = out["weights"]
    assert torch.allclose(w[~valid], torch.zeros_like(w[~valid]))
    assert torch.allclose(w.sum(-1), torch.ones(B), atol=1e-6)


def test_epoch0_prediction_equals_y_base_for_e1_and_e2():
    tok = _tokenizer()
    m = NeighborhoodFiLMModel(d=D)
    for mode in ("e1", "e2"):
        raw, distance, offset, is_anchor, valid, h_base, y_base = _batch(mode)
        s = tok(raw)
        out = m(h_base, s, distance, offset, is_anchor, valid, y_base)
        assert torch.equal(out["pred"], y_base)
        assert torch.equal(out["delta"], torch.zeros_like(y_base))


def test_all_masked_input_is_a_hard_error():
    tok = _tokenizer()
    raw, distance, offset, is_anchor, valid, h_base, y_base = _batch("e2")
    valid = torch.zeros_like(valid)
    m = NeighborhoodFiLMModel(d=D)
    import pytest
    with pytest.raises(ValueError, match="all-masked"):
        m(h_base, tok(raw), distance, offset, is_anchor, valid, y_base)


def test_e2_updates_qkv_and_structure_encoder_e1_qk_gradient_is_zero_not_buggy():
    for mode, expect_qk_grad in (("e1", False), ("e2", True)):
        torch.manual_seed(0)
        tok = _tokenizer()
        m = NeighborhoodFiLMModel(d=D)
        for p in list(m.parameters()) + list(tok.parameters()):
            torch.nn.init.normal_(p, std=0.1)
        raw, distance, offset, is_anchor, valid, h_base, y_base = _batch(mode)
        y = y_base + torch.randn(B)
        opt = torch.optim.AdamW(list(m.parameters()) + list(tok.parameters()), lr=1e-2)
        s = tok(raw)
        out = m(h_base, s, distance, offset, is_anchor, valid, y_base)
        loss = torch.nn.functional.huber_loss(out["pred"], y)
        opt.zero_grad(); loss.backward()
        qk_grad = (m.w_q.weight.grad.abs().sum() + m.w_k.weight.grad.abs().sum()
                  + m.dist_proj.weight.grad.abs().sum() + m.anchor_emb.weight.grad.abs().sum())
        v_grad = m.w_v.weight.grad.abs().sum()
        if expect_qk_grad:
            assert qk_grad > 0, "E2 should have gradient through Q/K/metadata (multiple valid keys to choose among)"
        else:
            assert qk_grad == 0, ("E1 has exactly one valid key, so its softmax weight is always 1 "
                                  "regardless of the logit -- zero Q/K/metadata gradient is expected, not a bug")
        assert v_grad > 0, f"{mode}: W_V should always receive gradient (c_struct depends on it even with 1 valid key)"
        assert tok.a_res.weight.grad is not None and tok.a_res.weight.grad.abs().sum() > 0


def test_e1_and_e2_have_identical_parameter_count():
    m1, m2 = NeighborhoodFiLMModel(d=D), NeighborhoodFiLMModel(d=D)
    assert m1.num_trainable_params() == m2.num_trainable_params()
    t1, t2 = _tokenizer(), _tokenizer()
    assert sum(p.numel() for p in t1.parameters()) == sum(p.numel() for p in t2.parameters())


def test_copying_init_state_gives_bit_identical_weights_across_separate_instances():
    torch.manual_seed(1)
    m1 = NeighborhoodFiLMModel(d=D)
    state = {k: v.clone() for k, v in m1.state_dict().items()}
    torch.manual_seed(999)              # different seed -> would normally diverge
    m2 = NeighborhoodFiLMModel(d=D)
    assert not all(torch.equal(state[k], v) for k, v in m2.state_dict().items())
    m2.load_state_dict(state)
    assert all(torch.equal(state[k], v) for k, v in m2.state_dict().items())
    assert m1 is not m2                 # separate instances, not a shared reference
