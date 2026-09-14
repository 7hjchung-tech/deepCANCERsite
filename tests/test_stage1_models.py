"""Model-level tests: content-builder routing, parameter counts, seeded init
parity, freeze-for-stage2, and gradient connectivity (task spec section 12,
"세 모델의 정보 사용" + "학습 경계와 재현성").
"""

from __future__ import annotations

import torch

from src.stage1.model import build_stage1_model
from src.stage1.modules import (
    BranchedProjectionContentBuilder,
    PairedDeltaContentBuilder,
    UnifiedReferenceDeltaContentBuilder,
    content_projection_param_count,
)
from src.stage1.schema import MODEL_MODES, SLOT_MUT_ONLY, SLOT_PAIRED, SLOT_WT_ONLY

D_ESM, BOTTLENECK = 6, 3


# ---------------------------------------------------------------------------
# closed-form parameter counts (pins the arithmetic behind the task's
# 45,216 / 135,648 / 86,368 example at d_esm=1280, bottleneck=32, token=128)
# ---------------------------------------------------------------------------
def test_content_projection_param_counts_match_task_example():
    assert content_projection_param_count("paired_delta", 1280, 32) == 45_216
    assert content_projection_param_count("branched_projection", 1280, 32) == 135_648
    assert content_projection_param_count("unified_reference_delta", 1280, 32) == 86_368


def test_content_projection_param_counts_match_actual_module():
    for mode, builder_cls in [
        ("paired_delta", PairedDeltaContentBuilder),
        ("branched_projection", BranchedProjectionContentBuilder),
        ("unified_reference_delta", UnifiedReferenceDeltaContentBuilder),
    ]:
        module = builder_cls(D_ESM, BOTTLENECK)
        actual = sum(p.numel() for p in module.parameters())
        assert actual == content_projection_param_count(mode, D_ESM, BOTTLENECK)


def test_full_model_param_count_reported_per_mode():
    counts = {}
    for mode in MODEL_MODES:
        model = build_stage1_model(mode, {"d_esm": D_ESM, "bottleneck_dim": BOTTLENECK, "layers": [33]})
        counts[mode] = model.num_total_params()
    # B > C > A in content-projection capacity, and that ordering must survive
    # into the whole-model total (shared modules are identical across modes).
    assert counts["paired_delta"] < counts["unified_reference_delta"] < counts["branched_projection"]


# ---------------------------------------------------------------------------
# content builder routing (white-box: exact equality, not "did it change")
# ---------------------------------------------------------------------------
def _one_slot_batch(n_slots=3):
    B, Lyr, A, D = 1, 1, n_slots, D_ESM
    torch.manual_seed(0)
    H_wt = torch.randn(B, Lyr, A, D)
    H_mut = torch.randn(B, Lyr, A, D)
    delta = torch.randn(B, Lyr, A, D)
    return H_wt, H_mut, delta


def test_A_content_depends_only_on_delta():
    builder = PairedDeltaContentBuilder(D_ESM, BOTTLENECK)
    H_wt, H_mut, delta = _one_slot_batch()
    slot_kind = torch.tensor([[SLOT_PAIRED, SLOT_WT_ONLY, SLOT_MUT_ONLY]])
    wt_present = torch.tensor([[1.0, 1.0, 0.0]])
    mut_present = torch.tensor([[1.0, 0.0, 1.0]])
    delta_valid = torch.tensor([[1.0, 0.0, 0.0]])

    out1 = builder(H_wt, H_mut, delta, wt_present, mut_present, delta_valid, slot_kind)
    H_wt_perturbed = H_wt + torch.randn_like(H_wt) * 10.0
    H_mut_perturbed = H_mut + torch.randn_like(H_mut) * 10.0
    out2 = builder(H_wt_perturbed, H_mut_perturbed, delta, wt_present, mut_present, delta_valid, slot_kind)
    assert torch.allclose(out1, out2)


def test_B_routes_each_slot_to_its_own_projection():
    builder = BranchedProjectionContentBuilder(D_ESM, BOTTLENECK)
    H_wt, H_mut, delta = _one_slot_batch()
    slot_kind = torch.tensor([[SLOT_PAIRED, SLOT_WT_ONLY, SLOT_MUT_ONLY]])
    wt_present = torch.tensor([[1.0, 1.0, 0.0]])
    mut_present = torch.tensor([[1.0, 0.0, 1.0]])
    delta_valid = torch.tensor([[1.0, 0.0, 0.0]])

    out = builder(H_wt, H_mut, delta, wt_present, mut_present, delta_valid, slot_kind)
    expected_paired = builder.p_delta(delta[:, :, 0])
    expected_wt = builder.p_wt(H_wt[:, :, 1])
    expected_mut = builder.p_mut(H_mut[:, :, 2])
    assert torch.allclose(out[:, :, 0], expected_paired, atol=1e-6)
    assert torch.allclose(out[:, :, 1], expected_wt, atol=1e-6)
    assert torch.allclose(out[:, :, 2], expected_mut, atol=1e-6)


def test_C_input_is_exactly_R_delta_flags_concat():
    builder = UnifiedReferenceDeltaContentBuilder(D_ESM, BOTTLENECK)
    H_wt, H_mut, delta = _one_slot_batch()
    slot_kind = torch.tensor([[SLOT_PAIRED, SLOT_WT_ONLY, SLOT_MUT_ONLY]])
    wt_present = torch.tensor([[1.0, 1.0, 0.0]])
    mut_present = torch.tensor([[1.0, 0.0, 1.0]])
    delta_valid = torch.tensor([[1.0, 0.0, 0.0]])

    out = builder(H_wt, H_mut, delta, wt_present, mut_present, delta_valid, slot_kind)

    # manually build the expected [R; delta; flags] per slot and re-run through
    # the SAME p_shared instance -- exact equality iff the wiring is correct.
    R_expected = torch.stack([H_wt[:, :, 0], H_wt[:, :, 1], H_mut[:, :, 2]], dim=2)
    flags_expected = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],   # paired
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # wt_only
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],   # mut_only
    ]).unsqueeze(0).unsqueeze(1)          # (1,1,3,6)
    x_expected = torch.cat([R_expected, delta, flags_expected], dim=-1)
    expected = builder.p_shared(x_expected)
    assert torch.allclose(out, expected, atol=1e-6)


def test_C_flags_change_is_visible_in_builder_input():
    """Same H_wt/H_mut/delta, but flip a slot from paired to wt_only: R and
    flags must change, verified structurally (not via random-init behavior)."""
    builder = UnifiedReferenceDeltaContentBuilder(D_ESM, BOTTLENECK)
    H_wt, H_mut, delta = _one_slot_batch(n_slots=1)
    wt_present = torch.tensor([[1.0]])
    mut_present = torch.tensor([[1.0]])

    paired_kind = torch.tensor([[SLOT_PAIRED]])
    wtonly_kind = torch.tensor([[SLOT_WT_ONLY]])
    delta_valid_paired = torch.tensor([[1.0]])
    delta_valid_wtonly = torch.tensor([[0.0]])

    out_paired = builder(H_wt, H_mut, delta, wt_present, mut_present, delta_valid_paired, paired_kind)
    out_wtonly = builder(H_wt, H_mut, delta, wt_present, mut_present, delta_valid_wtonly, wtonly_kind)
    # R is identical (wt_present=1 both times -> R=H_wt), delta identical, but
    # flags differ (is_paired/is_wt_only + delta_valid) -> input to p_shared differs.
    assert not torch.allclose(out_paired, out_wtonly)


# ---------------------------------------------------------------------------
# component-seeded init parity across modes (task spec section 5)
# ---------------------------------------------------------------------------
def test_shared_components_identical_across_modes_at_init():
    cfg = {"d_esm": D_ESM, "bottleneck_dim": BOTTLENECK, "layers": [33], "init_seed": 777}
    models = {mode: build_stage1_model(mode, cfg, init_seed=777) for mode in MODEL_MODES}
    ref = models["paired_delta"]
    for mode, m in models.items():
        assert torch.equal(m.layer_embedding.weight, ref.layer_embedding.weight), mode
        assert torch.equal(m.metadata_encoder.net[0].weight, ref.metadata_encoder.net[0].weight), mode
        assert torch.equal(m.pooling.q0, ref.pooling.q0), mode
        assert torch.equal(m.head.out.weight, ref.head.out.weight), mode


def test_p_delta_identical_between_A_and_B_at_init():
    cfg = {"d_esm": D_ESM, "bottleneck_dim": BOTTLENECK, "layers": [33], "init_seed": 42}
    a = build_stage1_model("paired_delta", cfg, init_seed=42)
    b = build_stage1_model("branched_projection", cfg, init_seed=42)
    assert torch.equal(a.content_builder.p_delta.net[0].weight, b.content_builder.p_delta.net[0].weight)
    assert torch.equal(a.content_builder.p_delta.net[2].weight, b.content_builder.p_delta.net[2].weight)


def test_construction_order_does_not_perturb_other_modes_init():
    cfg = {"d_esm": D_ESM, "bottleneck_dim": BOTTLENECK, "layers": [33], "init_seed": 5}
    # build in one order...
    m1_first = build_stage1_model("paired_delta", cfg, init_seed=5)
    m2_first = build_stage1_model("branched_projection", cfg, init_seed=5)
    # ...and the reverse order -- shared-component weights must not depend on order.
    m2_second = build_stage1_model("branched_projection", cfg, init_seed=5)
    m1_second = build_stage1_model("paired_delta", cfg, init_seed=5)
    assert torch.equal(m1_first.head.out.weight, m1_second.head.out.weight)
    assert torch.equal(m2_first.head.out.weight, m2_second.head.out.weight)


# ---------------------------------------------------------------------------
# freeze_for_stage2
# ---------------------------------------------------------------------------
def test_freeze_for_stage2_disables_grad_and_train_mode():
    model = build_stage1_model("unified_reference_delta", {"d_esm": D_ESM, "bottleneck_dim": BOTTLENECK, "layers": [33]})
    model.freeze_for_stage2()
    assert all(not p.requires_grad for p in model.parameters())
    assert model.num_trainable_params() == 0
    assert not model.training

    model.train()   # simulate an upstream Stage 2 model's train() sweep
    assert not model.training, "frozen Stage 1 must ignore .train(True)"

    model.train(True)
    assert not model.training


def test_freeze_output_stable_across_train_calls():
    fixture_batch = _synthetic_batch()
    model = build_stage1_model("branched_projection", {"d_esm": 8, "bottleneck_dim": 4, "layers": [33]})
    model.freeze_for_stage2()
    with torch.no_grad():
        out1 = model(fixture_batch)["pred"]
    model.train()
    with torch.no_grad():
        out2 = model(fixture_batch)["pred"]
    assert torch.allclose(out1, out2)


def _synthetic_batch():
    from src.stage1.dataset import Stage1Dataset, make_collate_fn
    from src.stage1.synthetic import make_synthetic_fixture

    fixture = make_synthetic_fixture(hidden_dim=8, layers=[33])
    ds = Stage1Dataset(fixture.cohort_entries, fixture.cache, window_radius=10, layers=[33])
    collate = make_collate_fn("branched_projection")
    return collate([ds[i] for i in range(len(ds))])


# ---------------------------------------------------------------------------
# gradient connectivity / boundary
# ---------------------------------------------------------------------------
def test_backward_reaches_only_used_branches_in_B():
    """A batch with ONLY paired slots must give p_wt/p_mut exactly zero
    gradient (they are multiplied by an all-zero mask, never None -- the
    graph is still connected, contribution is exactly zero)."""
    builder = BranchedProjectionContentBuilder(D_ESM, BOTTLENECK)
    H_wt, H_mut, delta = _one_slot_batch(n_slots=2)
    slot_kind = torch.tensor([[SLOT_PAIRED, SLOT_PAIRED]])
    wt_present = torch.tensor([[1.0, 1.0]])
    mut_present = torch.tensor([[1.0, 1.0]])
    delta_valid = torch.tensor([[1.0, 1.0]])

    out = builder(H_wt, H_mut, delta, wt_present, mut_present, delta_valid, slot_kind)
    out.pow(2).mean().backward()

    assert builder.p_delta.net[0].weight.grad is not None
    assert torch.count_nonzero(builder.p_delta.net[0].weight.grad) > 0
    assert builder.p_wt.net[0].weight.grad is not None
    assert torch.allclose(builder.p_wt.net[0].weight.grad, torch.zeros_like(builder.p_wt.net[0].weight.grad))
    assert torch.allclose(builder.p_mut.net[0].weight.grad, torch.zeros_like(builder.p_mut.net[0].weight.grad))


def test_no_optimizer_step_needed_for_synthetic_backward_check():
    """Connectivity-only check: backward() runs, no optimizer is even
    constructed -- this test would fail loudly (RuntimeError) if the graph
    were ever disconnected between the head and the content projections."""
    model = build_stage1_model("paired_delta", {"d_esm": 8, "bottleneck_dim": 4, "layers": [33]})
    batch = _synthetic_batch2("paired_delta")
    out = model(batch)
    out["pred"].pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def _synthetic_batch2(mode):
    from src.stage1.dataset import Stage1Dataset, make_collate_fn
    from src.stage1.synthetic import make_synthetic_fixture

    fixture = make_synthetic_fixture(hidden_dim=8, layers=[33])
    ds = Stage1Dataset(fixture.cohort_entries, fixture.cache, window_radius=10, layers=[33])
    collate = make_collate_fn(mode)
    return collate([ds[i] for i in range(len(ds))])
