"""Tests for src/analysis/lens_tracker.py.

No ESM-2 model loading: encoder and model are mocked.
DIM=32 keeps Ledoit-Wolf covariance fitting fast (avoids 1280×1280 matrix ops).

Coverage:
  - snapshot() returns exactly 33 records (layer_index 0..32)
  - each record has the required 7 keys
  - layer_index values cover exactly {0, ..., 32}
  - output file is created and non-empty
  - all numeric values are in valid ranges (maha ≥ 0, eff_rank ≥ 1)
  - epoch field is written correctly
  - JSONL output is line-parseable
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from src.analysis.lens_tracker import LensTracker

# ── constants ─────────────────────────────────────────────────────────────────

SEQ_LEN = 12
DIM = 32
_ALL_LAYERS = list(range(34))


# ── mock objects ──────────────────────────────────────────────────────────────

class _MockModel(nn.Module):
    """Returns layer-specific random reps; each layer gets a different scale."""

    def forward(self, tokens, repr_layers=None, **kwargs):
        N, L_tok = tokens.shape[0], tokens.shape[1]
        reps: dict[int, torch.Tensor] = {}
        if repr_layers is not None:
            for l in repr_layers:
                # Different scale per layer so h_l != h_33 in general
                torch.manual_seed(l)
                reps[l] = torch.randn(N, L_tok, DIM) * float(l + 1)
        return {"representations": reps}


class _MockEncoder:
    """Duck-typed ESMEncoder: exposes batch_converter, device, and model."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.model  = _MockModel()

    def batch_converter(self, data):
        # Returns (labels, strs, tokens) — only tokens are used
        N       = len(data)
        seq_len = len(data[0][1])
        tokens  = torch.zeros(N, seq_len + 2, dtype=torch.long)
        return None, None, tokens


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def encoder():
    return _MockEncoder()


@pytest.fixture
def reference_variants():
    wt  = "A" * SEQ_LEN
    pos = SEQ_LEN // 2
    mut = wt[:pos] + "C" + wt[pos + 1:]
    return [(wt, mut, pos), (wt, mut, pos)]   # K=2 variants


@pytest.fixture
def analysis_cfg(tmp_path):
    return {
        "output_dir": str(tmp_path / "layer_analysis"),
        "output_format": "jsonl",
        "pooling_W": 3,
        "pooling_mode": "uniform",
        "pooling_tau": 5.0,
        "cov": {
            "freeze_at": "pretrained",
            "estimator": "ledoit_wolf",
            "pca_dim": None,
            "cond_warn_threshold": 1e8,
        },
    }


@pytest.fixture
def tracker(encoder, reference_variants, analysis_cfg):
    return LensTracker(encoder, reference_variants, analysis_cfg)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_record_count(tracker, encoder):
    records = tracker.snapshot(encoder.model, epoch=0)
    assert len(records) == 33, f"Expected 33 records, got {len(records)}"


def test_record_keys(tracker, encoder):
    records = tracker.snapshot(encoder.model, epoch=0)
    expected = {"epoch", "layer_index", "maha_dist", "mean_norm", "variance", "eff_rank"}
    for r in records:
        assert set(r.keys()) == expected, f"Key mismatch: {set(r.keys())} != {expected}"


def test_layer_indices_complete(tracker, encoder):
    records = tracker.snapshot(encoder.model, epoch=0)
    indices = {r["layer_index"] for r in records}
    assert indices == set(range(33)), f"layer_index set mismatch: {indices}"


def test_output_file_created(tracker, encoder, analysis_cfg):
    tracker.snapshot(encoder.model, epoch=0)
    out = Path(analysis_cfg["output_dir"]) / "layer_metrics.jsonl"
    assert out.exists(), "Output file was not created"
    assert out.stat().st_size > 0, "Output file is empty"


def test_nonnegative_values(tracker, encoder):
    records = tracker.snapshot(encoder.model, epoch=0)
    for r in records:
        assert r["maha_dist"]  >= 0.0, f"maha_dist negative at layer {r['layer_index']}"
        assert r["mean_norm"]  >= 0.0, f"mean_norm negative at layer {r['layer_index']}"
        assert r["variance"]   >= 0.0, f"variance negative at layer {r['layer_index']}"
        assert r["eff_rank"]   >= 1.0, f"eff_rank < 1 at layer {r['layer_index']}"


def test_epoch_field(tracker, encoder):
    records = tracker.snapshot(encoder.model, epoch=7)
    assert all(r["epoch"] == 7 for r in records), "epoch field not written correctly"


def test_jsonl_parseable(tracker, encoder, analysis_cfg):
    tracker.snapshot(encoder.model, epoch=0)
    out = Path(analysis_cfg["output_dir"]) / "layer_metrics.jsonl"
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    parsed = [json.loads(ln) for ln in lines]
    assert len(parsed) == 33


def test_two_snapshots_append(tracker, encoder, analysis_cfg):
    """Two snapshot calls should append — 66 records total in the file."""
    tracker.snapshot(encoder.model, epoch=0)
    tracker.snapshot(encoder.model, epoch=1)
    out = Path(analysis_cfg["output_dir"]) / "layer_metrics.jsonl"
    lines = [l for l in out.read_text(encoding="utf-8").strip().split("\n") if l]
    assert len(lines) == 66, f"Expected 66 lines (2 × 33), got {len(lines)}"
