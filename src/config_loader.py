"""Tiny config loader supporting a single-level `extends: base.yaml` key.

configs/m1.yaml ~ m4.yaml each set `extends: base.yaml` and override only
what differs; this recursively deep-merges the override onto the base so
untouched nested keys (e.g. pooling.window_W) are inherited rather than lost.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_model_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    extends = cfg.pop("extends", None)
    if extends is None:
        return cfg
    base = load_model_config(path.parent / extends)
    return _deep_merge(base, cfg)
