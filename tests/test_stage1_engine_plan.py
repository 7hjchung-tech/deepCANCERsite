"""Engine helpers (train-only preprocessing, optimizer scope), experiment
plan generation, split-overlap audit, and the "import/--help never trains"
guarantee (task spec section 10 "이번 실행 제한").
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.stage1.engine import fit_meta_scaler_from_entries, fit_target_scaling, make_optimizer
from src.stage1.experiment_plan import generate_experiment_plan, write_plan
from src.stage1.metadata import raw_meta_features
from src.stage1.model import build_stage1_model
from src.stage1.schema import MODEL_MODES
from src.stage1.split_audit import audit_split_overlap, build_position_to_split, edited_span_positions
from src.stage1.synthetic import make_synthetic_fixture

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# train-only preprocessing
# ---------------------------------------------------------------------------
def test_target_scaling_uses_only_given_labels():
    train_labels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mean, std = fit_target_scaling(train_labels)
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(np.std(train_labels))


def test_meta_scaler_fit_from_train_entries_only():
    fixture = make_synthetic_fixture(hidden_dim=4, layers=[33])
    train_entries = fixture.cohort_entries[:3]
    scaler = fit_meta_scaler_from_entries(train_entries)
    raw = raw_meta_features(train_entries[0]["edit"])
    transformed = scaler.transform(raw[None, :])[0]
    assert transformed.shape == raw.shape
    # one-hot half must be untouched by standardization
    assert np.allclose(transformed[:5], raw[:5])


# ---------------------------------------------------------------------------
# optimizer scope
# ---------------------------------------------------------------------------
def test_optimizer_only_covers_trainable_params():
    model = build_stage1_model("branched_projection", {"d_esm": 8, "bottleneck_dim": 4, "layers": [33]})
    opt = make_optimizer(model, {"lr": 1e-4, "weight_decay": 0.0})
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == model.num_trainable_params()


def test_frozen_model_has_no_trainable_params_for_an_optimizer():
    """freeze_for_stage2() leaves nothing for an optimizer to update -- Stage 1
    training happens strictly BEFORE freezing, so make_optimizer is not called
    afterwards; this only pins that the trainable-param count is truly zero."""
    model = build_stage1_model("paired_delta", {"d_esm": 8, "bottleneck_dim": 4, "layers": [33]})
    model.freeze_for_stage2()
    assert model.num_trainable_params() == 0
    assert [p for p in model.parameters() if p.requires_grad] == []


# ---------------------------------------------------------------------------
# experiment plan generation (never trains)
# ---------------------------------------------------------------------------
def test_plan_generates_full_cross_product():
    windows, seeds = [5, 10], [42, 43]
    plan = generate_experiment_plan(model_modes=list(MODEL_MODES), windows=windows, folds=[None], seeds=seeds)
    assert len(plan) == len(MODEL_MODES) * len(windows) * 1 * len(seeds)
    run_ids = {p["run_id"] for p in plan}
    assert len(run_ids) == len(plan)   # all unique


def test_plan_marks_existing_results_for_skip(tmp_path):
    out_root = tmp_path / "runs"
    run_dir = out_root / "paired_delta" / "W5" / "shipped_split" / "seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}")

    plan = generate_experiment_plan(model_modes=["paired_delta"], windows=[5], folds=[None], seeds=[42],
                                     out_root=out_root)
    assert len(plan) == 1
    assert plan[0]["skip_existing"] is True


def test_write_plan_is_valid_json(tmp_path):
    plan = generate_experiment_plan(model_modes=["paired_delta"], windows=[5], folds=[None], seeds=[42],
                                     out_root=tmp_path / "runs")
    path = tmp_path / "plan.json"
    write_plan(plan, path)
    import json
    loaded = json.loads(path.read_text())
    assert len(loaded) == 1
    assert loaded[0]["command"][0] == "python"


# ---------------------------------------------------------------------------
# split / edited-span overlap audit
# ---------------------------------------------------------------------------
def test_edited_span_overlap_detected_when_deliberately_constructed():
    fixture = make_synthetic_fixture(hidden_dim=4, layers=[33])
    delins_entry = next(e for e in fixture.cohort_entries if e["edit"].edit_type == "delins")
    delins_entry["row"]["split"] = "train"

    pos_to_split = {p: "train" for p in range(1, 60)}
    touched = sorted(edited_span_positions(delins_entry["edit"]))
    assert len(touched) == 5   # 5 deleted WT residues
    pos_to_split[touched[-1]] = "val"   # force a conflicting split on one touched position

    result = audit_split_overlap([delins_entry], pos_to_split)
    assert result.n_indels_audited == 1
    assert len(result.edited_span_overlaps) == 1
    assert result.edited_span_overlaps[0]["position"] == touched[-1]


def test_no_overlap_when_span_uniformly_assigned():
    fixture = make_synthetic_fixture(hidden_dim=4, layers=[33])
    delins_entry = next(e for e in fixture.cohort_entries if e["edit"].edit_type == "delins")
    delins_entry["row"]["split"] = "train"
    pos_to_split = {p: "train" for p in range(1, 60)}
    result = audit_split_overlap([delins_entry], pos_to_split)
    assert result.is_clean


def test_missense_and_synonymous_are_not_audited_as_indels():
    fixture = make_synthetic_fixture(hidden_dim=4, layers=[33])
    entries = [e for e in fixture.cohort_entries if e["edit"].edit_type in ("missense", "synonymous")]
    for e in entries:
        e["row"]["split"] = "train"
    pos_to_split = build_position_to_split([e["row"] for e in fixture.cohort_entries])
    result = audit_split_overlap(entries, pos_to_split)
    assert result.n_indels_audited == 0


# ---------------------------------------------------------------------------
# "import/--help never trains" (task spec section 10)
# ---------------------------------------------------------------------------
def test_import_train_stage1_has_no_side_effects():
    result = subprocess.run(
        [sys.executable, "-c", "import train_stage1"],
        cwd=_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_help_does_not_train_or_download():
    result = subprocess.run(
        [sys.executable, "train_stage1.py", "--help"],
        cwd=_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_no_args_prints_help_and_does_nothing():
    result = subprocess.run(
        [sys.executable, "train_stage1.py"],
        cwd=_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_execute_plan_flag_is_refused_this_session():
    import shutil

    out_dir = _ROOT / "runs" / "_test_plan_refusal"
    try:
        result = subprocess.run(
            [sys.executable, "train_stage1.py", "--plan", "--windows", "5", "--execute-plan",
             "--out", str(out_dir)],
            cwd=_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode != 0
        assert "out of scope" in (result.stdout + result.stderr).lower()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
