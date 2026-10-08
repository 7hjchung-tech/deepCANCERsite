"""R0 / R1 reverse-direction Stage 2 checks on synthetic tensors (no GPU, no real data)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage2.reverse import ReverseFiLMModel, ReverseMultiHeadFiLMModel, build_reverse_model  # noqa: E402

D, B, N = 128, 3, 11   # N includes the Stage 1 metadata token in the last slot


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, 9, 32, generator=g)
    K = torch.randn(B, N, D, generator=g)
    V = torch.randn(B, N, D, generator=g)
    valid = torch.ones(B, N)
    valid[1, 6:-1] = 0         # residue padding in the middle, metadata token stays valid
    type_id = torch.tensor([0, 1, 2])
    y1 = torch.randn(B, generator=g)
    return S, K, V, valid, type_id, y1


def test_param_counts_exact_and_r1_difference():
    r0, r1 = ReverseFiLMModel("r0_struct_mean"), ReverseFiLMModel("r1_seq_query")
    shared = (32 * D + D) + 3 * D + (2 * D * 32 + 32) + (32 * 2 * D + 2 * D) + (D * 32 + 32) + (32 + 1)
    assert r0.num_trainable_params() == shared
    assert r1.num_trainable_params() - r0.num_trainable_params() == (D * 32 + 32) + (32 * 32 + 32)


def test_initial_prediction_equals_stage1_and_shapes():
    S, K, V, valid, tid, y1 = _batch()
    for mode in ("r0_struct_mean", "r1_seq_query"):
        out = ReverseFiLMModel(mode)(S, K, V, valid, tid, y1)
        assert torch.equal(out["pred"], y1)
        assert out["z"].shape == (B, D)
    a = ReverseFiLMModel("r1_seq_query")(S, K, V, valid, tid, y1)["attn"]
    assert a.shape == (B, N - 1, 9)
    assert torch.allclose(a.sum(-1), torch.ones(B, N - 1), atol=1e-6)     # softmax over the 9 structure tokens


def test_metadata_token_and_padding_do_not_affect_output():
    S, K, V, valid, tid, y1 = _batch()
    for mode in ("r0_struct_mean", "r1_seq_query"):
        torch.manual_seed(0)
        m = ReverseFiLMModel(mode)
        for p in m.parameters():                     # non-zero head so delta depends on z
            torch.nn.init.normal_(p, std=0.1)
        V2 = V.clone()
        V2[:, -1] = 100.0                            # metadata token
        V2[1, 6:-1] = -50.0                          # padded residue slots
        assert torch.allclose(m(S, K, V, valid, tid, y1)["pred"], m(S, K, V2, valid, tid, y1)["pred"], atol=1e-5)


def test_r0_context_is_identical_across_tokens():
    S, K, V, valid, tid, y1 = _batch()
    c = ReverseFiLMModel("r0_struct_mean")(S, K, V, valid, tid, y1)["c"]
    assert torch.allclose(c, c[:, :1].expand_as(c))


def test_gradients_reach_every_module_after_a_few_steps():
    S, K, V, valid, tid, y1 = _batch()
    y = y1 + torch.randn(B)
    for mode in ("r0_struct_mean", "r1_seq_query"):
        torch.manual_seed(0)
        m = ReverseFiLMModel(mode)
        S_ = S.clone().requires_grad_(True)          # stands in for the tokenizer output
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        init = {n: p.detach().clone() for n, p in m.named_parameters()}
        for _ in range(4):
            opt.zero_grad()
            loss = torch.nn.functional.huber_loss(m(S_, K, V, valid, tid, y1)["pred"], y)
            loss.backward()
            opt.step()
        assert S_.grad is not None and S_.grad.abs().sum() > 0
        for n, p in m.named_parameters():
            assert not torch.equal(p.detach(), init[n]), f"{mode}: {n} never updated"


# ---- R2 (multi-head, learnable temperature) ----------------------------------------------

def test_r2_param_count_and_more_capacity_than_r1():
    r1 = ReverseFiLMModel("r1_seq_query")
    r2 = ReverseMultiHeadFiLMModel()                         # defaults: n_heads=4, d_head=16
    dh = 4 * 16
    expected = ((D * dh + dh) + (32 * dh + dh) + (32 * dh + dh) + (dh * D + D)     # W_Q, W_K, W_V, W_O
                + 1                                                                # log_tau
                + 3 * D                                                            # type_emb
                + (2 * D * 32 + 32) + (32 * 2 * D + 2 * D)                        # film
                + (D * 32 + 32) + (32 + 1))                                        # head
    assert r2.num_trainable_params() == expected
    assert r2.num_trainable_params() > r1.num_trainable_params()


def test_r2_initial_prediction_equals_stage1_and_shapes():
    S, K, V, valid, tid, y1 = _batch()
    out = ReverseMultiHeadFiLMModel()(S, K, V, valid, tid, y1)
    assert torch.equal(out["pred"], y1)                       # film/head last layers zero-init
    assert out["z"].shape == (B, D)
    assert out["attn"].shape == (B, N - 1, 4, 9)               # [B, N_res, n_heads, 9]
    assert torch.allclose(out["attn"].sum(-1), torch.ones(B, N - 1, 4), atol=1e-6)
    assert out["entropy"].shape == (B, 4)                      # one entropy per head


def test_r2_tau_init_changes_tau_but_not_at_init_output():
    sharp = ReverseMultiHeadFiLMModel(tau_init=0.1)
    flat = ReverseMultiHeadFiLMModel(tau_init=2.0)
    assert float(sharp.tau()) < float(flat.tau())
    S, K, V, valid, tid, y1 = _batch()
    # delta is 0 at init regardless of tau (film/head are zero-init), but attn differs with tau
    assert torch.equal(sharp(S, K, V, valid, tid, y1)["pred"], y1)
    assert not torch.allclose(sharp(S, K, V, valid, tid, y1)["attn"], flat(S, K, V, valid, tid, y1)["attn"])


def test_r2_metadata_token_and_padding_do_not_affect_output():
    S, K, V, valid, tid, y1 = _batch()
    torch.manual_seed(0)
    m = ReverseMultiHeadFiLMModel()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    V2 = V.clone()
    V2[:, -1] = 100.0
    V2[1, 6:-1] = -50.0
    assert torch.allclose(m(S, K, V, valid, tid, y1)["pred"], m(S, K, V2, valid, tid, y1)["pred"], atol=1e-5)


def test_r2_gradients_reach_every_module_after_a_few_steps():
    S, K, V, valid, tid, y1 = _batch()
    y = y1 + torch.randn(B)
    torch.manual_seed(0)
    m = ReverseMultiHeadFiLMModel()
    S_ = S.clone().requires_grad_(True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    init = {n: p.detach().clone() for n, p in m.named_parameters()}
    for _ in range(4):
        opt.zero_grad()
        loss = torch.nn.functional.huber_loss(m(S_, K, V, valid, tid, y1)["pred"], y)
        loss.backward()
        opt.step()
    assert S_.grad is not None and S_.grad.abs().sum() > 0
    for n, p in m.named_parameters():
        assert not torch.equal(p.detach(), init[n]), f"r2: {n} never updated"


def test_build_reverse_model_dispatches_by_mode():
    assert isinstance(build_reverse_model("r0_struct_mean"), ReverseFiLMModel)
    assert isinstance(build_reverse_model("r1_seq_query"), ReverseFiLMModel)
    r2 = build_reverse_model("r2_multihead_query", tau_init=0.1)
    assert isinstance(r2, ReverseMultiHeadFiLMModel)
    assert abs(float(r2.tau()) - 0.1) < 1e-5
