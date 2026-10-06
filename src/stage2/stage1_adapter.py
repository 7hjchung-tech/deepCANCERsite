"""Adapter that exposes the Stage 1 unified_reference_delta outputs Stage 2 needs.

Stage 1's own prediction path is left untouched: y1 is the original Stage 1
head output mapped back to z-score units; K/V/valid are the attention inputs
the pooling already uses. Stage 2 reads the same K/V through its own query.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.stage1.checkpoint import build_and_load_stage1, load_checkpoint
from src.stage1.metadata import MetaScaler
from src.stage1.model import Stage1Model

from .schema import STAGE1_MODE


@dataclass
class Stage1Handle:
    model: Stage1Model
    y_mean: float
    y_std: float
    meta_scaler: MetaScaler
    cfg: dict
    reference: dict
    ckpt_path: str


def load_stage1_handle(ckpt_path: str, expected_reference: dict, device: str) -> Stage1Handle:
    """Loads a Stage 1 checkpoint. Wrong model_mode or incompatible reference raises.

    freeze=True leaves every Stage 1 parameter frozen and forces eval mode
    permanently (Stage1Model.train() becomes a no-op). Joint training later
    re-enables only the modules it is allowed to update.
    """
    model, ckpt = build_and_load_stage1(
        ckpt_path, expected_mode=STAGE1_MODE, expected_reference=expected_reference,
        map_location=device, freeze=True,
    )
    model.to(device)
    return Stage1Handle(
        model=model,
        y_mean=float(ckpt["y_mean"]),
        y_std=float(ckpt["y_std"]),
        meta_scaler=MetaScaler.from_state_dict(ckpt["meta_scaler"]),
        cfg=ckpt["cfg"],
        reference=ckpt["reference"],
        ckpt_path=str(ckpt_path),
    )


def stage1_outputs(handle: Stage1Handle, batch: dict, grad: bool) -> dict:
    """Run Stage 1 once. grad=False for the frozen mode (no graph is built)."""
    device = next(handle.model.parameters()).device
    batch = {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        out = handle.model(batch, return_extras=True)
        y1 = out["pred"] * handle.y_std + handle.y_mean
    return {"y1": y1, "K": out["K"], "V": out["V"], "valid": out["attention_valid"]}


def read_reference(ckpt_path: str) -> dict:
    return load_checkpoint(ckpt_path)["reference"]
