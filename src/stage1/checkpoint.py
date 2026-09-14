"""Stage 1 checkpoint save/load and the freeze/hand-off contract for Stage 2.

A checkpoint bundles everything needed to reproduce Stage 1's OUTPUT exactly
(not the frozen ESM backbone -- that is identified, not duplicated):
  * model_mode + resolved config (window radius, layers, dims, seeds)
  * trainable state_dict (content projections, layer embedding, metadata
    encoder, query/temperature, sequence head)
  * MetaScaler state (train-only fitted numeric standardization)
  * target (y) mean/std for inverse-transforming predictions
  * reference identity: ESM checkpoint name, cache schema version, WT hash,
    alignment version -- so a Stage 2 load can refuse an incompatible cache.

Loading with the wrong model_mode is a hard error, never a partial/silent
state_dict load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .model import Stage1Model, build_stage1_model
from .schema import ALIGNMENT_VERSION, CACHE_SCHEMA_VERSION, CHECKPOINT_SCHEMA_VERSION


class CheckpointModeMismatchError(Exception):
    pass


class CheckpointReferenceMismatchError(Exception):
    pass


def save_checkpoint(
    path: str | Path,
    model: Stage1Model,
    cfg: dict,
    meta_scaler_state: dict,
    y_mean: float,
    y_std: float,
    reference: dict,
    extra: Optional[dict] = None,
) -> None:
    """reference must contain at least:
        {"esm_checkpoint", "cache_schema_version", "wt_hash", "alignment_version"}
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_mode": model.mode,
        "cfg": cfg,
        "state_dict": model.state_dict(),
        "meta_scaler": meta_scaler_state,
        "y_mean": y_mean,
        "y_std": y_std,
        "reference": reference,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: checkpoint schema_version {ckpt.get('schema_version')!r} != "
            f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    return ckpt


def build_and_load_stage1(
    path: str | Path,
    expected_mode: Optional[str] = None,
    expected_reference: Optional[dict] = None,
    map_location: str = "cpu",
    freeze: bool = True,
) -> tuple[Stage1Model, dict]:
    """Load a Stage 1 checkpoint for Stage 2 consumption.

    Raises CheckpointModeMismatchError if `expected_mode` is given and
    differs from the checkpoint's model_mode -- never partially loads a
    state_dict for the wrong architecture.

    Raises CheckpointReferenceMismatchError if `expected_reference` (e.g. the
    Stage 2 run's own cache identity) disagrees with what this checkpoint was
    trained against (ESM checkpoint id / cache schema / WT hash / alignment
    version) -- an incompatible cache must be a clear error, not silent reuse.
    """
    ckpt = load_checkpoint(path, map_location=map_location)
    mode = ckpt["model_mode"]
    if expected_mode is not None and mode != expected_mode:
        raise CheckpointModeMismatchError(
            f"{path}: checkpoint model_mode={mode!r} != expected {expected_mode!r}"
        )
    if expected_reference is not None:
        ref = ckpt.get("reference", {})
        mismatches = {
            k: (ref.get(k), v) for k, v in expected_reference.items() if ref.get(k) != v
        }
        if mismatches:
            raise CheckpointReferenceMismatchError(
                f"{path}: incompatible reference identity: {mismatches}"
            )

    model = build_stage1_model(mode, ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    if freeze:
        model.freeze_for_stage2()
    return model, ckpt


def inverse_transform_y(pred_scaled: torch.Tensor, y_mean: float, y_std: float) -> torch.Tensor:
    return pred_scaled * y_std + y_mean


def default_reference(esm_checkpoint: str, wt_hash: str) -> dict:
    return {
        "esm_checkpoint": esm_checkpoint,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "wt_hash": wt_hash,
        "alignment_version": ALIGNMENT_VERSION,
    }
