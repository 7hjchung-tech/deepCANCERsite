"""Training/evaluation engine for Stage 1 -- ready to run, but never invoked
by this task (no --train call is made in this session; see train_stage1.py).

Reuses train.py's Spearman/Pearson/RMSE implementation (compute_metrics) so
Stage 1 numbers are computed the exact same way as the legacy M1-M4 numbers.
"""

from __future__ import annotations

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
        for g in ("missense", "synonymous", "indel"):
            mask = groups == g
            if mask.sum() >= 2:
                per_group[g] = compute_metrics(labels[mask], preds[mask])["spearman"]
        m["by_group"] = per_group
        scored = [per_group[g] for g in ("missense", "indel") if g in per_group]
        m["subset"] = float(np.mean(scored)) if scored else float("nan")
    return m, preds


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def train_one_run(
    model: Stage1Model,
    train_ds: Stage1Dataset,
    val_ds: Stage1Dataset,
    cfg: dict,
    device: str = "cpu",
    max_epochs: int = 100,
    patience: int = 10,
    select_on: str = "subset",
) -> dict:
    """Full train/val loop. NOT called anywhere in this task's execution --
    provided so a future explicit `--train` run has a ready implementation.
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

    for epoch in range(1, max_epochs + 1):
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

        val_m, _ = evaluate(model, val_loader, y_mean, y_std, device, edit_types=val_edit_types)
        history.append({"epoch": epoch, "train_loss": running / max(n_batches, 1), "val": val_m,
                         "seconds": round(time.time() - t0, 1)})
        score = sign * val_m[select_on]
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "val": val_m}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best["epoch"] >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best": best, "y_mean": y_mean, "y_std": y_std, "model": model}
