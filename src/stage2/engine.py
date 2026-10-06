"""Stage 2 data assembly, training (frozen_stage1 / joint_l2sp) and evaluation.

Metrics reuse the Stage 1 conventions: compute_metrics from train.py (the same
Spearman/Pearson/RMSE used for M1-M4 and Stage 1), group_of() for the
missense / synonymous / indel split, and subset = mean(Spearman_missense,
Spearman_indel) as the early-stopping monitor. Everything is in z-score units.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.stage1.dataset import Stage1Dataset, stage1_collate
from src.stage1.engine import group_of, set_all_seeds
from train import compute_metrics

from .model import Stage2Model
from .schema import VARIANT_TYPE_ID, VARIANT_TYPES
from .stage1_adapter import Stage1Handle, stage1_outputs
from .structure import StructureStore, StructureTokenizer

UNFREEZE_CHOICES = ("pooling", "head", "metadata_encoder", "content_builder")


def variant_type_id(edit) -> int:
    return VARIANT_TYPE_ID[group_of(edit.edit_type)]


def make_collate(model_mode: str, store: StructureStore):
    def _collate(items: list[dict]) -> dict:
        batch = stage1_collate(items, model_mode)
        batch["struct_raw"] = store.raw(batch["var_id"])
        batch["type_id"] = torch.tensor([variant_type_id(it["edit"]) for it in items], dtype=torch.long)
        return batch
    return _collate


def make_loader(entries: list[dict], cache, window: int, layers: list[int], handle: Stage1Handle,
                store: StructureStore, model_mode: str, batch_size: int, shuffle: bool) -> DataLoader:
    ds = Stage1Dataset(entries, cache, window, layers, meta_scaler=handle.meta_scaler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(model_mode, store))


def _move(d: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def forward_batch(stage2: Stage2Model, tokenizer: StructureTokenizer, handle: Stage1Handle,
                  batch: dict, device: str, grad: bool) -> tuple[dict, dict]:
    s1 = stage1_outputs(handle, batch, grad=grad)
    S = tokenizer(_move(batch["struct_raw"], device))
    out = stage2(S, s1["K"], s1["V"], s1["valid"], batch["type_id"].to(device), s1["y1"])
    return out, s1


def _select_monitor(metrics: dict) -> float:
    return float(metrics["subset"]) if metrics["subset"] == metrics["subset"] else float("-inf")


@torch.no_grad()
def evaluate(stage2: Stage2Model, tokenizer: StructureTokenizer, handle: Stage1Handle, loader: DataLoader,
             device: str) -> dict:
    stage2.eval()
    tokenizer.eval()
    y, p2, p1, groups, ent, gam, bet, dlt = [], [], [], [], [], [], [], []
    for batch in loader:
        out, s1 = forward_batch(stage2, tokenizer, handle, batch, device, grad=False)
        y.append(batch["label"].numpy())
        p2.append(out["pred"].cpu().numpy())
        p1.append(s1["y1"].cpu().numpy())
        groups += [VARIANT_TYPES[t] for t in batch["type_id"].tolist()]
        ent.append(out["entropy"].cpu().numpy())
        gam.append(out["gamma"].cpu().numpy())
        bet.append(out["beta"].cpu().numpy())
        dlt.append(out["delta"].cpu().numpy())
    y = np.concatenate(y)
    p2 = np.concatenate(p2)
    p1 = np.concatenate(p1)
    groups = np.array(groups)
    ent = np.concatenate(ent)
    m2 = _summarise(y, p2, groups)
    m1 = _summarise(y, p1, groups)
    return {
        "stage2": m2, "stage1": m1,
        "stage2_minus_stage1_subset": (m2["subset"] - m1["subset"]) if m2["subset"] == m2["subset"] else None,
        "attention_entropy_mean": ent.mean(axis=0).tolist(),        # per query (1 for single, 9 for nine)
        "gamma_mean": float(np.concatenate(gam).mean()), "gamma_std": float(np.concatenate(gam).std()),
        "beta_mean": float(np.concatenate(bet).mean()), "beta_std": float(np.concatenate(bet).std()),
        "delta_abs_mean": float(np.abs(np.concatenate(dlt)).mean()),
        "preds_stage2": p2.tolist(), "labels": y.tolist(), "groups": groups.tolist(),
    }


def _summarise(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict:
    m = compute_metrics(y, pred)
    by_group, by_group_n = {}, {}
    for g in ("missense", "synonymous", "indel"):
        mask = groups == g
        by_group_n[g] = int(mask.sum())
        if mask.sum() >= 2:
            by_group[g] = compute_metrics(y[mask], pred[mask])["spearman"]
    scored = [by_group[g] for g in ("missense", "indel") if g in by_group]
    m["by_group"] = by_group
    m["by_group_n"] = by_group_n
    m["subset"] = float(np.mean(scored)) if scored else float("nan")
    m["mae"] = float(np.mean(np.abs(y - pred)))
    return m


def unfreeze_stage1(handle: Stage1Handle, modules: list[str]) -> list[tuple[str, nn.Parameter]]:
    """Enable gradients for the named Stage 1 submodules only. Returns (name, param) pairs.

    Stage 1 stays in eval mode (no dropout) even while these parameters train.
    """
    for p in handle.model.parameters():
        p.requires_grad_(False)
    named = []
    for name in modules:
        if name not in UNFREEZE_CHOICES:
            raise ValueError(f"unknown Stage 1 module {name!r}; choose from {UNFREEZE_CHOICES}")
        sub = getattr(handle.model, name)
        for pname, p in sub.named_parameters():
            p.requires_grad_(True)
            named.append((f"{name}.{pname}", p))
    handle.model.eval()
    return named


def l2sp_penalty(named: list[tuple[str, nn.Parameter]], ref: dict[str, torch.Tensor]) -> torch.Tensor:
    """R_SP with reduction='sum': sum over unfrozen Stage 1 parameters of ||theta - theta_ref||^2."""
    total = torch.zeros((), device=named[0][1].device) if named else torch.zeros(())
    for name, p in named:
        total = total + (p - ref[name]).pow(2).sum()
    return total


def train_stage2(
    *, stage2: Stage2Model, tokenizer: StructureTokenizer, handle: Stage1Handle,
    loaders: dict[str, DataLoader], store: StructureStore, cfg: dict, device: str, out_dir: Path,
    seed: int, train_mode: str, unfreeze: list[str] | None = None, init_state: dict | None = None,
) -> dict:
    """frozen_stage1: only Stage 2 + tokenizer train.
    joint_l2sp: additionally the chosen Stage 1 submodules, with a lower lr and an L2-SP anchor.
    Selection and early stopping use validation only; test is evaluated once at the end."""
    set_all_seeds(seed)
    stage2.to(device)
    tokenizer.to(device)

    param_groups = [
        {"params": list(stage2.parameters()) + list(tokenizer.parameters()), "lr": cfg["lr"], "name": "stage2"},
    ]
    named_s1: list[tuple[str, nn.Parameter]] = []
    ref: dict[str, torch.Tensor] = {}
    if train_mode == "joint_l2sp":
        named_s1 = unfreeze_stage1(handle, unfreeze or cfg["unfreeze"])
        ref = {n: p.detach().clone() for n, p in named_s1}
        param_groups.append({"params": [p for _, p in named_s1],
                              "lr": cfg["lr"] * float(cfg["stage1_lr_ratio"]), "name": "stage1"})
    if init_state is not None:
        stage2.load_state_dict(init_state["stage2"])
        tokenizer.load_state_dict(init_state["tokenizer"])

    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.HuberLoss(delta=float(cfg["huber_delta"]))
    lam = float(cfg.get("lambda_sp", 0.0)) if train_mode == "joint_l2sp" else 0.0

    history, best, best_state, stop_reason = [], {"score": float("-inf"), "epoch": -1}, None, "max_epochs"
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        stage2.train()
        tokenizer.train()
        t0 = time.time()
        task_sum, pen_sum, n = 0.0, 0.0, 0
        for batch in loaders["train"]:
            out, _ = forward_batch(stage2, tokenizer, handle, batch, device, grad=(train_mode == "joint_l2sp"))
            task = loss_fn(out["pred"], batch["label"].to(device))
            pen = l2sp_penalty(named_s1, ref) if named_s1 else torch.zeros((), device=device)
            loss = task + lam * pen
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"] if p.requires_grad], float(cfg["grad_clip"]))
            optimizer.step()
            task_sum += float(task.detach())
            pen_sum += float(pen.detach())
            n += 1
        seconds = round(time.time() - t0, 1)
        val = evaluate(stage2, tokenizer, handle, loaders["val"], device)
        score = _select_monitor(val["stage2"])
        improved = score > best["score"] + float(cfg["min_delta"])
        if improved:
            best = {"score": score, "epoch": epoch}
            best_state = {
                "stage2": copy.deepcopy(stage2.state_dict()),
                "tokenizer": copy.deepcopy(tokenizer.state_dict()),
                "stage1_unfrozen": {n_: p.detach().clone() for n_, p in named_s1},
            }
        patience_counter = epoch - best["epoch"]
        history.append({
            "epoch": epoch, "train_task_loss": task_sum / max(n, 1),
            "train_l2sp_penalty": pen_sum / max(n, 1), "lambda_sp": lam,
            "val_stage2_subset": score, "val_stage2_spearman": val["stage2"]["spearman"],
            "val_stage2_rmse": val["stage2"]["rmse"], "val_stage2_mae": val["stage2"]["mae"],
            "val_stage1_subset": val["stage1"]["subset"], "val_stage1_spearman": val["stage1"]["spearman"],
            "val_stage2_minus_stage1_subset": val["stage2_minus_stage1_subset"],
            "val_by_group_spearman": val["stage2"]["by_group"], "val_by_group_n": val["stage2"]["by_group_n"],
            "attention_entropy_mean": val["attention_entropy_mean"],
            "gamma_std": val["gamma_std"], "beta_std": val["beta_std"], "delta_abs_mean": val["delta_abs_mean"],
            "lr_stage2": optimizer.param_groups[0]["lr"],
            "seconds": seconds, "improved": bool(improved),
            "best_epoch_so_far": best["epoch"], "best_score_so_far": best["score"],
            "patience_counter": patience_counter,
        })
        if not improved and patience_counter >= int(cfg["patience"]):
            stop_reason = "patience"
            break

    if best_state is None:
        raise RuntimeError("no epoch produced a finite validation monitor; check the inputs")
    stage2.load_state_dict(best_state["stage2"])
    tokenizer.load_state_dict(best_state["tokenizer"])
    for n_, p in named_s1:
        p.data.copy_(best_state["stage1_unfrozen"][n_])
    test = evaluate(stage2, tokenizer, handle, loaders["test"], device)
    return {"history": history, "best": best, "stop_reason": stop_reason,
            "best_state": best_state, "test": test, "stage1_named": named_s1, "ref": ref}
