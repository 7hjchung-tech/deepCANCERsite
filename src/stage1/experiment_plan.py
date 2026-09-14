"""Generate the 3-model x window x fold x seed experiment plan.

Generating the plan NEVER starts training -- it only resolves configs and
command lines and writes a JSON manifest. Actually running any of it needs
train_stage1.py --train (a single run) or an explicit --sweep, neither of
which this task invokes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import resolve_config
from .schema import MODEL_MODES


def _fold_tag(fold: Optional[tuple[int, int, int]]) -> str:
    if fold is None:
        return "shipped_split"
    n_folds, fold_idx, cv_seed = fold
    return f"cv{n_folds}_fold{fold_idx}_seed{cv_seed}"


def generate_experiment_plan(
    model_modes: list[str] = MODEL_MODES,
    windows: list[int] = (5, 10, 20),
    folds: list[Optional[tuple[int, int, int]]] = (None,),
    seeds: list[int] = (42, 43, 44),
    out_root: str | Path = "runs/stage1",
    layers: Optional[list[int]] = None,
) -> list[dict]:
    """Example `windows`/`seeds` values are illustrative sweep candidates,
    NOT claimed optimal settings (per task spec §10).
    """
    for m in model_modes:
        if m not in MODEL_MODES:
            raise ValueError(f"unknown model mode {m!r}")

    out_root = Path(out_root)
    plan: list[dict] = []
    for mode in model_modes:
        for w in windows:
            for fold in folds:
                for seed in seeds:
                    ftag = _fold_tag(fold)
                    run_id = f"{mode}__W{w}__{ftag}__seed{seed}"
                    out_dir = out_root / mode / f"W{w}" / ftag / f"seed{seed}"
                    cfg = resolve_config(mode, window_radius=w, layers=layers)
                    cmd = [
                        "python", "train_stage1.py",
                        "--model", mode, "--window", str(w), "--seed", str(seed),
                        "--out", str(out_dir), "--train",
                    ]
                    if fold is not None:
                        n_folds, fold_idx, cv_seed = fold
                        cmd += ["--cv-folds", str(n_folds), "--cv-fold", str(fold_idx), "--cv-seed", str(cv_seed)]
                    already_done = (out_dir / "metrics.json").exists()
                    plan.append({
                        "run_id": run_id, "model_mode": mode, "window_radius": w,
                        "fold": ftag, "seed": seed, "out_dir": str(out_dir),
                        "command": cmd, "resolved_config": cfg,
                        "skip_existing": already_done,
                    })
    return plan


def write_plan(plan: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, default=str)
