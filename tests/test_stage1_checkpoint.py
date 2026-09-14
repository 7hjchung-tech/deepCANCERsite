"""Stage 1 -> Stage 2 checkpoint contract: round trip, mode-mismatch refusal,
reference-mismatch refusal, and freeze-on-load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.stage1.checkpoint import (
    CheckpointModeMismatchError,
    CheckpointReferenceMismatchError,
    build_and_load_stage1,
    default_reference,
    save_checkpoint,
)
from src.stage1.model import build_stage1_model

CFG = {"d_esm": 8, "bottleneck_dim": 4, "layers": [33]}


def _save(tmp_path, mode="paired_delta", wt_hash="hashA"):
    model = build_stage1_model(mode, CFG, init_seed=1)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path, model, {**CFG, "model_mode": mode}, {"mean": [0.0], "std": [1.0]},
        y_mean=1.0, y_std=2.0, reference=default_reference("esm2_t33_650M_UR50D", wt_hash),
    )
    return path


def test_checkpoint_round_trip_preserves_weights(tmp_path: Path):
    path = _save(tmp_path)
    model, ckpt = build_and_load_stage1(path, expected_mode="paired_delta")
    original = build_stage1_model("paired_delta", CFG, init_seed=1)
    assert torch.equal(model.head.out.weight, original.head.out.weight)
    assert ckpt["y_mean"] == 1.0 and ckpt["y_std"] == 2.0


def test_checkpoint_freezes_on_load_by_default(tmp_path: Path):
    path = _save(tmp_path)
    model, _ = build_and_load_stage1(path, expected_mode="paired_delta")
    assert model.num_trainable_params() == 0
    assert not model.training


def test_wrong_mode_load_is_rejected(tmp_path: Path):
    path = _save(tmp_path, mode="paired_delta")
    with pytest.raises(CheckpointModeMismatchError):
        build_and_load_stage1(path, expected_mode="branched_projection")


def test_incompatible_reference_is_rejected(tmp_path: Path):
    path = _save(tmp_path, wt_hash="hashA")
    with pytest.raises(CheckpointReferenceMismatchError):
        build_and_load_stage1(
            path, expected_mode="paired_delta",
            expected_reference=default_reference("esm2_t33_650M_UR50D", "hashB"),
        )


def test_compatible_reference_is_accepted(tmp_path: Path):
    path = _save(tmp_path, wt_hash="hashA")
    model, _ = build_and_load_stage1(
        path, expected_mode="paired_delta",
        expected_reference=default_reference("esm2_t33_650M_UR50D", "hashA"),
    )
    assert model is not None
