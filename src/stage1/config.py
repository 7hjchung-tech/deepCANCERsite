"""Stage 1 config loading: reuses src/config_loader.py's `extends:` deep-merge
(the same mechanism M1-M4 use) so configs/stage1/*.yaml follow one convention
across the whole repo instead of Stage 1 inventing a second config format.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import load_model_config  # noqa: E402
from .schema import DEFAULT_BOTTLENECK_DIM, DEFAULT_D_ESM, MODEL_MODES, RAW_META_DIM  # noqa: E402

STAGE1_CONFIG_DIR = _ROOT / "configs" / "stage1"

DEFAULTS: dict = {
    "d_esm": DEFAULT_D_ESM,
    "bottleneck_dim": DEFAULT_BOTTLENECK_DIM,
    "layers": [33],
    "window_radius": 10,
    "head_hidden": 256,
    "dropout": 0.1,
    "meta_raw_dim": RAW_META_DIM,
    "init_seed": 1234,
    "lr": 1.0e-4,
    "weight_decay": 0.01,
    "batch_size": 32,
    "loss": "huber",
    "huber_delta": 1.0,
    "select_on": "subset",
    "patience": 10,
    "seeds": [42, 43, 44],
    "split_policy": "shipped",   # "shipped" | "position_cv"
}


def load_stage1_config(model_mode: str, path: str | Path | None = None) -> dict:
    if model_mode not in MODEL_MODES:
        raise ValueError(f"unknown model mode {model_mode!r}, expected one of {MODEL_MODES}")
    if path is None:
        path = STAGE1_CONFIG_DIR / f"{model_mode}.yaml"
    cfg = {**DEFAULTS, **load_model_config(path)}
    cfg["model_mode"] = model_mode
    return cfg


def resolve_config(model_mode: str, window_radius: int | None = None,
                    layers: list[int] | None = None, path: str | Path | None = None,
                    overrides: dict | None = None) -> dict:
    cfg = load_stage1_config(model_mode, path)
    if window_radius is not None:
        cfg["window_radius"] = window_radius
    if layers is not None:
        cfg["layers"] = list(layers)
    if overrides:
        cfg.update(overrides)
    return cfg
