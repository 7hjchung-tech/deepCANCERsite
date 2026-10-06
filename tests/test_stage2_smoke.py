"""Stage 2 smoke checks on synthetic tensors (no ESM, no GPU, no real data)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage1.config import resolve_config  # noqa: E402
from src.stage1.model import build_stage1_model  # noqa: E402
from src.stage1.modules import ConstantQueryPooling  # noqa: E402
from src.stage2.attention import masked_cosine_attention  # noqa: E402
from src.stage2.engine import l2sp_penalty, unfreeze_stage1  # noqa: E402
from src.stage2.model import Stage2Model  # noqa: E402
from src.stage2.stage1_adapter import Stage1Handle, stage1_outputs  # noqa: E402
from src.stage2.structure import StructureTokenizerMissing, load_structure_store, load_tokenizer  # noqa: E402
from src.stage2.synthetic import SyntheticSmokeTokenizer  # noqa: E402
from src.stage1.dataset import Stage1Dataset, stage1_collate  # noqa: E402
from src.stage1.synthetic import make_synthetic_fixture  # noqa: E402

D, B, N = 128, 3, 11


def _synthetic_stage1_batch(hidden: int, window: int):
    fx = make_synthetic_fixture(hidden_dim=hidden, layers=[33])
    ds = Stage1Dataset(fx.cohort_entries[:4], fx.cache, window, [33])
    return stage1_collate([ds[i] for i in range(len(ds))], "unified_reference_delta")


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, 9, 32, generator=g)
    K = torch.randn(B, N, D, generator=g)
    V = torch.randn(B, N, D, generator=g)
    valid = torch.ones(B, N)
    valid[1, 7:] = 0           # padding on sample 1
    valid[2, :] = 0            # sample 2 has no valid token at all
    valid[2, 0] = 1
    type_id = torch.tensor([0, 1, 2])
    y1 = torch.randn(B, generator=g)
    return S, K, V, valid, type_id, y1


def test_forward_shapes_and_finite_for_both_modes():
    S, K, V, valid, tid, y1 = _batch()
    for mode, Q in (("single_query", 1), ("nine_query", 9)):
        out = Stage2Model(mode)(S, K, V, valid, tid, y1)
        assert out["weights"].shape == (B, Q, N)
        assert out["z"].shape == (B, D)
        assert out["pred"].shape == (B,)
        assert torch.isfinite(out["pred"]).all() and torch.isfinite(out["weights"]).all()


def test_padding_gets_zero_attention_and_valid_sums_to_one():
    S, K, V, valid, tid, y1 = _batch()
    valid[2, :] = 0                       # sample 2 becomes all-masked for this check
    out = Stage2Model("nine_query")(S, K, V, valid, tid, y1)
    w = out["weights"]
    pad = valid.bool().logical_not().unsqueeze(1).expand_as(w)
    assert torch.all(w[pad] == 0)
    sums = w.sum(-1)
    assert torch.allclose(sums[0], torch.ones(9), atol=1e-5)
    assert torch.allclose(sums[1], torch.ones(9), atol=1e-5)
    assert torch.all(sums[2] == 0)     # all-masked sample: explicit zeros, not fake uniform mass


def test_initial_prediction_equals_stage1_prediction():
    S, K, V, valid, tid, y1 = _batch()
    for mode in ("single_query", "nine_query"):
        out = Stage2Model(mode)(S, K, V, valid, tid, y1)
        assert torch.allclose(out["pred"], y1, atol=1e-7)
        assert torch.all(out["delta"] == 0)


def test_both_modes_have_identical_parameter_structure_and_count():
    a, b = Stage2Model("single_query"), Stage2Model("nine_query")
    assert [k for k, _ in a.named_parameters()] == [k for k, _ in b.named_parameters()]
    assert a.num_trainable_params() == b.num_trainable_params()
    adapter = 32 * 128 + 128
    type_emb = 3 * 128
    tau = 1
    film = (128 * 32 + 32) + (32 * 256 + 256)
    head = (128 * 32 + 32) + (32 * 1 + 1)
    expected = adapter + type_emb + tau + film + head
    assert a.num_trainable_params() == expected


def test_nine_query_equals_single_query_when_structure_tokens_are_identical():
    S, K, V, valid, tid, y1 = _batch()
    S_same = S[:, :1, :].expand(-1, 9, -1).contiguous()
    single = Stage2Model("single_query")
    # give the residual path a nonzero output so the pooled vector actually matters,
    # then copy the SAME weights into the nine-query model
    with torch.no_grad():
        single.head[-1].weight.normal_(0, 0.1)
        single.film[-1].weight.normal_(0, 0.1)
    nine = Stage2Model("nine_query")
    nine.load_state_dict(single.state_dict())
    o1 = single(S_same, K, V, valid, tid, y1)
    o9 = nine(S_same, K, V, valid, tid, y1)
    assert torch.allclose(o1["z"], o9["z"], atol=1e-5)
    assert torch.allclose(o1["pred"], o9["pred"], atol=1e-5)


def test_masked_attention_matches_stage1_constant_query_pooling():
    torch.manual_seed(0)
    pool = ConstantQueryPooling(D)
    with torch.no_grad():
        pool.log_tau.fill_(0.3)
    S, K, V, valid, _, _ = _batch(1)
    z_ref, w_ref = pool(K, V, valid)
    tau = torch.nn.functional.softplus(pool.log_tau) + 1e-4
    out, w = masked_cosine_attention(pool.q0.view(1, 1, D).expand(B, 1, D), K, V, valid, tau)
    assert torch.allclose(out[:, 0], z_ref, atol=1e-5)
    assert torch.allclose(w[:, 0], w_ref, atol=1e-6)


def test_temperature_stays_positive_for_extreme_parameter():
    m = Stage2Model("single_query")
    with torch.no_grad():
        m.log_tau.fill_(-50.0)
    assert float(m.tau()) > 0


def test_gradient_reaches_upstream_after_a_few_steps():
    torch.manual_seed(0)
    S, K, V, valid, tid, y1 = _batch(2)
    tok = SyntheticSmokeTokenizer()
    m = Stage2Model("nine_query")
    opt = torch.optim.AdamW(list(m.parameters()) + list(tok.parameters()), lr=1e-2)
    raw = {"continuous": torch.randn(B, 8), "ss": torch.tensor([0, 1, 2])}
    target = torch.randn(B)
    grads_seen = []
    for _ in range(4):
        out = m(tok(raw), K, V, valid, tid, y1)
        loss = torch.nn.functional.huber_loss(out["pred"], target)
        opt.zero_grad()
        loss.backward()
        grads_seen.append(float(m.adapter.weight.grad.abs().sum()))
        opt.step()
    assert max(grads_seen) > 0.0      # zero-init head: upstream grads appear after the first update
    assert float(m.adapter.weight.abs().sum()) > 0


def test_stage1_adapter_outputs_feed_stage2_and_stage1_stays_frozen():
    cfg = resolve_config("unified_reference_delta", window_radius=2, layers=[33],
                         overrides={"d_esm": 16, "batch_size": 2})
    model = build_stage1_model("unified_reference_delta", cfg, init_seed=3)
    model.freeze_for_stage2()
    handle = Stage1Handle(model=model, y_mean=0.0, y_std=1.0, meta_scaler=None, cfg=cfg, reference={},
                          ckpt_path="synthetic")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    batch = _synthetic_stage1_batch(hidden=16, window=2)
    s1 = stage1_outputs(handle, batch, grad=False)
    assert s1["K"].shape[-1] == D and s1["valid"].shape == s1["K"].shape[:2]
    stage2 = Stage2Model("single_query")
    S = torch.randn(s1["K"].shape[0], 9, 32)
    out = stage2(S, s1["K"], s1["V"], s1["valid"], torch.zeros(s1["K"].shape[0], dtype=torch.long), s1["y1"])
    assert torch.allclose(out["pred"], s1["y1"], atol=1e-6)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(not p.requires_grad for p in model.parameters())


def test_joint_l2sp_zero_at_reference_positive_after_perturbation_and_gradients_flow():
    cfg = resolve_config("unified_reference_delta", window_radius=2, layers=[33],
                         overrides={"d_esm": 16, "batch_size": 2})
    model = build_stage1_model("unified_reference_delta", cfg, init_seed=5)
    model.freeze_for_stage2()
    handle = Stage1Handle(model=model, y_mean=0.0, y_std=1.0, meta_scaler=None, cfg=cfg, reference={},
                          ckpt_path="synthetic")
    named = unfreeze_stage1(handle, ["pooling", "head"])
    ref = {n: p.detach().clone() for n, p in named}
    assert float(l2sp_penalty(named, ref)) == 0.0
    with torch.no_grad():
        for _, p in named:
            p.add_(0.01)
    assert float(l2sp_penalty(named, ref)) > 0.0
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert len(frozen_names) > 0 and all(n.startswith(("content_builder", "layer_embedding", "metadata_encoder"))
                                         for n in frozen_names)


def test_structure_store_rejects_bad_tables_and_aligns_by_id(tmp_path):
    cols = ["var_id", "split", "A_plddt", "A_rsasa", "A_dist_walker_a", "A_dist_walker_b", "A_dist_atp_contact",
            "A_dist_ssdna_binding", "A_dist_bcdx2_interface", "A_dist_cx3_interface",
            "A_ss_helix", "A_ss_sheet", "A_ss_loop"]
    rows = []
    for i, v in enumerate(["v1", "v2", "v3"]):
        rows.append([v, "train", 50.0 + i, 0.5, 1, 2, 3, 4, 5, 6, 1, 0, 0])
    good = pd.DataFrame(rows, columns=cols)
    split = {"v1": "train", "v2": "train", "v3": "train"}

    p = tmp_path / "good.csv"
    good.iloc[::-1].to_csv(p, index=False)     # reversed row order must still align by id
    store = load_structure_store(p, ["v1", "v2", "v3"], split)
    assert store.raw(["v2"])["continuous"][0, 0].item() == pytest.approx(51.0)

    dup = pd.concat([good, good.iloc[:1]])
    p2 = tmp_path / "dup.csv"
    dup.to_csv(p2, index=False)
    with pytest.raises(ValueError, match="duplicated"):
        load_structure_store(p2, ["v1", "v2", "v3"], split)

    p3 = tmp_path / "missing.csv"
    good.iloc[:2].to_csv(p3, index=False)
    with pytest.raises(ValueError, match="no structure row"):
        load_structure_store(p3, ["v1", "v2", "v3"], split)

    nan = good.copy()
    nan.loc[0, "A_rsasa"] = np.nan
    p4 = tmp_path / "nan.csv"
    nan.to_csv(p4, index=False)
    with pytest.raises(ValueError, match="NaN"):
        load_structure_store(p4, ["v1", "v2", "v3"], split)

    p5 = tmp_path / "split.csv"
    good.to_csv(p5, index=False)
    with pytest.raises(ValueError, match="split disagrees"):
        load_structure_store(p5, ["v1", "v2", "v3"], {"v1": "val", "v2": "train", "v3": "train"})


def test_missing_tokenizer_fails_with_clear_error():
    with pytest.raises(StructureTokenizerMissing, match="No StructureTokenizer"):
        load_tokenizer(None, {})
    with pytest.raises(StructureTokenizerMissing, match="was not found"):
        load_tokenizer("not_a_real_module_xyz:make", {})
