"""Training/evaluation engine for Stage 1 -- ready to run, but never invoked
by this task (no --train call is made in this session; see train_stage1.py).

Reuses train.py's Spearman/Pearson/RMSE implementation (compute_metrics) so
Stage 1 numbers are computed the exact same way as the legacy M1-M4 numbers.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train import compute_metrics  # noqa: E402  (reuse: same Spearman/Pearson/RMSE as M1-M4)

from .dataset import Stage1Dataset, make_collate_fn  # noqa: E402
from .metadata import MetaScaler, raw_meta_features  # noqa: E402
from .model import Stage1Model  # noqa: E402


def set_all_seeds(seed: int) -> None:
    """Seed every stochastic-training source this engine actually touches.

    Model *initialization* already has its own deterministic, isolated seeding
    (see model.py's _build_seeded, which saves/restores the global RNG state
    around each submodule build) -- that path is untouched by this function.
    What was NOT seeded anywhere before this: DataLoader shuffling (no
    `generator=` is passed to the train DataLoader, so it draws from whatever
    the ambient global torch RNG state happens to be) and dropout (drawn from
    the same global state during forward). Calling this once, before the
    DataLoader/optimizer are built and before the first forward pass, makes
    both of those reproducible per seed. No DataLoader workers are used here
    (num_workers defaults to 0), so there is no separate worker-seed to set.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state_snapshot() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _rng_state_restore(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def fit_target_scaling(train_labels: np.ndarray) -> tuple[float, float]:
    y_mean = float(train_labels.mean())
    y_std = float(train_labels.std()) or 1.0
    return y_mean, y_std


def fit_meta_scaler_from_entries(entries: list[dict]) -> MetaScaler:
    """Train-only fit: `entries` must be the TRAIN split's cohort entries only."""
    raw = np.stack([raw_meta_features(e["edit"]) for e in entries])
    return MetaScaler.fit(raw)


def make_optimizer(model: Stage1Model, cfg: dict) -> torch.optim.Optimizer:
    """Trainable parameters only -- a frozen backbone has none to begin with
    here (ESM never enters this module), but this also protects against a
    future accidental non-trainable buffer leaking into the optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    print(f"[stage1.engine] optimizer over {n:,} trainable params")
    return torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))


def group_of(edit_type: str) -> str:
    if edit_type == "missense":
        return "missense"
    if edit_type == "synonymous":
        return "synonymous"
    return "indel"


@torch.no_grad()
def evaluate(model: Stage1Model, loader: DataLoader, y_mean: float, y_std: float, device: str,
             edit_types: Optional[list[str]] = None) -> tuple[dict, np.ndarray]:
    model.eval()
    preds_all, labels_all = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        preds_all.append(out["pred"].detach().cpu().numpy())
        labels_all.append(batch["label"].detach().cpu().numpy())
    preds = np.concatenate(preds_all) * y_std + y_mean
    labels = np.concatenate(labels_all)

    m = compute_metrics(labels, preds)
    m["loss"] = float((m["rmse"] / y_std) ** 2)

    if edit_types is not None:
        groups = np.array([group_of(t) for t in edit_types])
        per_group = {}
        per_group_n = {}
        for g in ("missense", "synonymous", "indel"):
            mask = groups == g
            per_group_n[g] = int(mask.sum())
            if mask.sum() >= 2:
                per_group[g] = compute_metrics(labels[mask], preds[mask])["spearman"]
        m["by_group"] = per_group
        # sample count per group even when the group is too small to score
        # (never fabricate a 0 spearman -- the group is just absent from
        # by_group above; by_group_n tells the analysis script why).
        m["by_group_n"] = per_group_n
        scored = [per_group[g] for g in ("missense", "indel") if g in per_group]
        m["subset"] = float(np.mean(scored)) if scored else float("nan")
    return m, preds


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _clone_state_dict(model: Stage1Model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _save_resume_bundle(path: str | Path, payload: dict) -> None:
    """Write atomically (tmp file + rename) so a crash mid-write never leaves
    a truncated/corrupt resume file that a later run would try to load."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_resume_bundle(path: str | Path, map_location: str = "cpu") -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def train_one_run(
    model: Stage1Model,
    train_ds: Stage1Dataset,
    val_ds: Stage1Dataset,
    cfg: dict,
    device: str = "cpu",
    max_epochs: int = 100,
    patience: int = 10,
    select_on: str = "subset",
    seed: Optional[int] = None,
    min_delta: float = 0.0,
    checkpoint_epochs: Optional[set] = None,
    resume_path: Optional[str | Path] = None,
) -> dict:
    """Full train/val loop.

    New (additive, all default to the exact prior behavior):
      seed              -- if given, seeds python/numpy/torch(+cuda) once
                            before the DataLoader/optimizer are built, so
                            shuffle order and dropout become reproducible per
                            seed (see set_all_seeds' docstring for why this
                            was previously NOT the case).
      min_delta         -- an epoch only counts as "improved" if
                            score > best_score + min_delta. 0.0 (default) is
                            bit-identical to the original `score > best_score`
                            comparison.
      checkpoint_epochs -- if given, a state_dict snapshot is kept (in memory,
                            returned as "fixed_budget_states") for every epoch
                            number in this set that the run actually reaches --
                            for the fixed-training-budget comparison, never
                            interpolated for an epoch a run stopped short of.
      resume_path       -- if given: (a) if the file exists, training resumes
                            from it instead of starting fresh (model/optimizer/
                            RNG/history/best-so-far all restored); (b) after
                            every epoch, the current full state is written
                            there. Caller is responsible for removing the file
                            once a run completes successfully.
    """
    model = model.to(device)
    optimizer = make_optimizer(model, cfg)
    loss_fn = nn.HuberLoss(delta=float(cfg.get("huber_delta", 1.0))) if cfg.get("loss") == "huber" else nn.MSELoss()

    collate = make_collate_fn(model.mode)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 32)), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 32)), shuffle=False, collate_fn=collate)

    train_labels = np.array([e["row"][cfg.get("label_col", "z_score_D4_D14")] for e in train_ds.entries], dtype=np.float32)
    y_mean, y_std = fit_target_scaling(train_labels)
    val_edit_types = [e["edit"].edit_type for e in val_ds.entries]

    history: list[dict] = []
    sign = -1.0 if select_on == "loss" else 1.0
    best = {"score": -np.inf, "epoch": -1}
    best_state = None
    fixed_budget_states: dict[int, dict] = {}
    checkpoint_epochs = checkpoint_epochs or set()
    start_epoch = 1
    stop_reason = "max_epochs"

    resumed = load_resume_bundle(resume_path) if resume_path is not None else None
    if resumed is not None:
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        _rng_state_restore(resumed["rng_state"])
        history = resumed["history"]
        best = resumed["best"]
        best_state = resumed["best_state"]
        fixed_budget_states = resumed.get("fixed_budget_states", {})
        start_epoch = resumed["epoch"] + 1
        print(f"[stage1.engine] resumed from {resume_path} at epoch {start_epoch} "
              f"(best so far: epoch {best['epoch']}, score {best['score']:.4f})")
    elif seed is not None:
        # Only seed a fresh run -- a resumed run restores the exact RNG state
        # it had left off at instead (re-seeding here would replay the same
        # shuffle order it already consumed).
        set_all_seeds(seed)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            batch = _to_device(batch, device)
            target = (batch["label"] - y_mean) / y_std
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = loss_fn(out["pred"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            running += float(loss.detach())
            n_batches += 1
        epoch_seconds = round(time.time() - t0, 1)

        val_m, _ = evaluate(model, val_loader, y_mean, y_std, device, edit_types=val_edit_types)
        score = sign * val_m[select_on]
        improved = score > best["score"] + min_delta
        if improved:
            best = {"score": score, "epoch": epoch, "val": val_m}
            best_state = _clone_state_dict(model)
        patience_counter = epoch - best["epoch"]

        if epoch in checkpoint_epochs:
            fixed_budget_states[epoch] = _clone_state_dict(model)

        history.append({
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_loss": val_m["loss"],
            "monitor_metric": select_on,
            "monitor_value": float(val_m[select_on]),
            "val_spearman": val_m["spearman"],
            "val_pearson": val_m["pearson"],
            "val_rmse": val_m["rmse"],
            "val_subset_spearman": val_m.get("subset"),
            "val_by_group_spearman": {
                g: (val_m["by_group"][g] if g in val_m.get("by_group", {}) else None)
                for g in ("missense", "synonymous", "indel")
            },
            "val_by_group_n": val_m.get("by_group_n", {}),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": epoch_seconds,
            "improved": bool(improved),
            "best_epoch_so_far": best["epoch"],
            "best_score_so_far": float(best["score"]),
            "patience_counter": patience_counter,
            # kept for backward compatibility with code/tests that read the
            # old nested "val" blob (e.g. eval_stage1_test.py-style consumers)
            "val": val_m,
        })

        if resume_path is not None:
            _save_resume_bundle(resume_path, {
                "epoch": epoch,
                "history": history,
                "best": best,
                "best_state": best_state,
                "fixed_budget_states": fixed_budget_states,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": _rng_state_snapshot(),
                "y_mean": y_mean,
                "y_std": y_std,
                "seed": seed,
            })

        if not improved and patience_counter >= patience:
            stop_reason = "patience"
            break

    last_state = _clone_state_dict(model)
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history, "best": best, "y_mean": y_mean, "y_std": y_std, "model": model,
        "last_state": last_state, "fixed_budget_states": fixed_budget_states,
        "stop_reason": stop_reason, "seed": seed,
    }
