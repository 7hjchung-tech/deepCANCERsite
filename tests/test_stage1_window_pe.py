"""Window/attention/positional-encoding tests (task spec section 12, "Window/attention").

All CPU, no real ESM -- built on the synthetic fixture from src/stage1/synthetic.py.
"""

from __future__ import annotations

import torch

from src.stage1.alignment import normalize_edit
from src.stage1.cache import FakeFrozenEncoder, build_cache_from_manifest
from src.stage1.dataset import Stage1Dataset, make_collate_fn
from src.stage1.positional import build_position_encoding
from src.stage1.schema import SLOT_PAD
from src.stage1.synthetic import SYNTHETIC_WT, make_synthetic_fixture
from src.stage1.window import build_sample_tokens


def test_pe_same_wt_position_identical_encoding_across_window_radii():
    row = {"var_id": "v1", "slim_consequence": "clinical_inframe_insertion", "pp": 25,
           "HGVSp": "NP_TEST:p.Xaa25dup"}
    edit = normalize_edit(SYNTHETIC_WT, row)
    encoder = FakeFrozenEncoder(hidden_dim=4, seed=0)
    mut_seq = SYNTHETIC_WT[:25] + SYNTHETIC_WT[24:25] + SYNTHETIC_WT[25:]
    cache = build_cache_from_manifest([{"var_id": "v1", "mut_seq": mut_seq}], SYNTHETIC_WT, [33], encoder)

    small = build_sample_tokens("v1", edit, cache, window_radius=3, layers=[33])
    big = build_sample_tokens("v1", edit, cache, window_radius=15, layers=[33])

    def pe_at_wt_pos(t, wt_pos):
        idx = (t.wt_pos == wt_pos).nonzero(as_tuple=True)[0].item()
        pe = build_position_encoding(
            t.anchor_rel_coord.unsqueeze(0), t.insertion_rank.unsqueeze(0),
            (t.slot_kind == 2).to(torch.float32).unsqueeze(0),
        )
        return pe[0, idx]

    assert torch.allclose(pe_at_wt_pos(small, 24), pe_at_wt_pos(big, 24))
    assert torch.allclose(pe_at_wt_pos(small, 25), pe_at_wt_pos(big, 25))


def test_pe_insertion_rank_zeroed_for_non_mut_only():
    anchor = torch.tensor([[0.0, 1.0, -1.0]])
    rank = torch.tensor([[5.0, 0.0, 0.0]])   # only slot 0 pretends to carry a rank
    is_mut_only = torch.tensor([[1.0, 0.0, 0.0]])
    pe = build_position_encoding(anchor, rank, is_mut_only)
    assert pe.shape == (1, 3, 128)
    # slots 1,2 are not mut_only -> their rank half must be exactly zero
    assert torch.allclose(pe[0, 1, 64:], torch.zeros(64))
    assert torch.allclose(pe[0, 2, 64:], torch.zeros(64))
    assert not torch.allclose(pe[0, 0, 64:], torch.zeros(64))


def test_window_shapes_across_radii_and_layer_counts():
    fixture = make_synthetic_fixture(hidden_dim=8, layers=[10, 20, 33])
    entry = next(e for e in fixture.cohort_entries if e["edit"].edit_type == "delins")
    for W in (0, 3, 10, 25):
        for layers in ([33], [10, 33], [10, 20, 33]):
            t = build_sample_tokens(entry["var_id"], entry["edit"], fixture.cache, W, layers)
            L = len(layers)
            assert t.H_wt.shape == (L, t.n_slots, 8)
            assert t.delta.shape == (L, t.n_slots, 8)
            assert t.slot_kind.shape == (t.n_slots,)
            assert not torch.isnan(t.H_wt).any()
            assert not torch.isnan(t.delta).any()


def test_batch_padding_marks_pad_slots_and_zero_validity():
    fixture = make_synthetic_fixture(hidden_dim=8, layers=[33])
    ds = Stage1Dataset(fixture.cohort_entries, fixture.cache, window_radius=10, layers=[33])
    collate = make_collate_fn("branched_projection")
    batch = collate([ds[i] for i in range(len(ds))])

    A_max = batch["slot_kind"].shape[1]
    for b in range(len(ds)):
        n = ds[b]["tokens"].n_slots
        if n < A_max:
            assert (batch["slot_kind"][b, n:] == SLOT_PAD).all()
            assert (batch["token_valid"][b, n:] == 0).all()
            assert (batch["attention_valid"][b, n:] == 0).all()
            assert (batch["wt_pos"][b, n:] == 0).all()
            assert (batch["mut_pos"][b, n:] == 0).all()


def test_model_a_attention_valid_excludes_one_sided_slots():
    fixture = make_synthetic_fixture(hidden_dim=8, layers=[33])
    ds = Stage1Dataset(fixture.cohort_entries, fixture.cache, window_radius=10, layers=[33])
    collate = make_collate_fn("paired_delta")
    batch = collate([ds[i] for i in range(len(ds))])
    from src.stage1.schema import SLOT_PAIRED
    is_paired = batch["slot_kind"] == SLOT_PAIRED
    assert torch.equal(batch["attention_valid"].bool(), is_paired & batch["token_valid"].bool())


def test_model_bc_attention_valid_includes_one_sided_slots():
    fixture = make_synthetic_fixture(hidden_dim=8, layers=[33])
    ds = Stage1Dataset(fixture.cohort_entries, fixture.cache, window_radius=10, layers=[33])
    for mode in ("branched_projection", "unified_reference_delta"):
        collate = make_collate_fn(mode)
        batch = collate([ds[i] for i in range(len(ds))])
        assert torch.equal(batch["attention_valid"], batch["token_valid"])


def test_metadata_only_edge_case_window_radius_zero():
    """A pure deletion at W=0 has zero PAIRED slots -> model A's residue
    attention is empty; only the metadata token remains valid. This must not
    crash and must not be confused with an all-invalid batch."""
    fixture = make_synthetic_fixture(hidden_dim=8, layers=[33])
    entry = next(e for e in fixture.cohort_entries if e["edit"].edit_type == "deletion" and e["edit"].d == 4)
    ds = Stage1Dataset([entry], fixture.cache, window_radius=0, layers=[33])
    collate = make_collate_fn("paired_delta")
    batch = collate([ds[0]])
    assert batch["attention_valid"].sum() == 0    # zero residue tokens valid for model A

    from src.stage1.model import build_stage1_model
    model = build_stage1_model("paired_delta", {"d_esm": 8, "bottleneck_dim": 4, "layers": [33]})
    out = model(batch, return_extras=True)
    assert torch.isfinite(out["pred"]).all()
    # only the metadata token (last column) can carry weight
    assert torch.allclose(out["attn_weights"][:, -1], torch.ones(1))
    assert torch.allclose(out["attn_weights"][:, :-1], torch.zeros_like(out["attn_weights"][:, :-1]))


def test_single_vs_padded_batch_eval_agree():
    fixture = make_synthetic_fixture(hidden_dim=8, layers=[33])
    ds = Stage1Dataset(fixture.cohort_entries, fixture.cache, window_radius=10, layers=[33])
    collate = make_collate_fn("unified_reference_delta")

    from src.stage1.model import build_stage1_model
    model = build_stage1_model("unified_reference_delta", {"d_esm": 8, "bottleneck_dim": 4, "layers": [33]})
    model.eval()

    single = collate([ds[0]])
    with torch.no_grad():
        pred_single = model(single)["pred"]

    padded = collate([ds[0], ds[1], ds[2]])
    with torch.no_grad():
        pred_padded = model(padded)["pred"]

    assert torch.allclose(pred_single[0], pred_padded[0], atol=1e-5)


def test_pooling_no_nan_on_zero_content():
    from src.stage1.modules import ConstantQueryPooling
    pool = ConstantQueryPooling(dim=8)
    K = torch.zeros(2, 5, 8)
    V = torch.zeros(2, 5, 8)
    valid = torch.ones(2, 5)
    z, w = pool(K, V, valid)
    assert not torch.isnan(z).any()
    assert not torch.isnan(w).any()
    assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-5)


def test_attention_weights_sum_to_one_for_every_model():
    fixture = make_synthetic_fixture(hidden_dim=8, layers=[33])
    ds = Stage1Dataset(fixture.cohort_entries, fixture.cache, window_radius=10, layers=[33])
    from src.stage1.model import build_stage1_model
    for mode in ("paired_delta", "branched_projection", "unified_reference_delta"):
        collate = make_collate_fn(mode)
        batch = collate([ds[i] for i in range(len(ds))])
        model = build_stage1_model(mode, {"d_esm": 8, "bottleneck_dim": 4, "layers": [33]})
        out = model(batch, return_extras=True)
        assert torch.allclose(out["attn_weights"].sum(-1), torch.ones(len(ds)), atol=1e-5)
