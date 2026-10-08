"""Data assembly, training and evaluation for the 3D-neighborhood FiLM model (E0/E1/E2).

ESM and Stage1 (content_builder/pooling/head, everything) stay frozen/eval for every
condition -- this experiment never unfreezes Stage1 (unlike src/stage2/engine.py's
joint_l2sp path, which this module does not use). Only ResidueStructureTokenizer.a_res,
the tokenizer's Qk inner module, and NeighborhoodFiLMModel's own parameters train.
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

from .neighbor_model import NeighborhoodFiLMModel
from .neighbor_structure import ResidueStructureTokenizer, WTNeighborStore
from .stage1_adapter import Stage1Handle, stage1_outputs


def make_collate(store: WTNeighborStore, mode: str, var_id_to_pp: dict[str, int]):
    def _collate(items: list[dict]) -> dict:
        # items come from Stage1Dataset.__getitem__ (tokens/meta_raw/label/var_id/edit only --
        # no "row"), so the anchor position is looked up by var_id from the cohort-level map
        # built once in make_loader, not read off `it` directly.
        batch = stage1_collate(items, "unified_reference_delta")
        anchors = [var_id_to_pp[it["var_id"]] for it in items]
        batch["struct_raw"] = store.raw_for_anchors(anchors, mode)
        batch["anchor_pos"] = torch.tensor(anchors, dtype=torch.long)
        batch["edit_type"] = [it["edit"].edit_type for it in items]   # stage1_collate drops this; needed for grouping
        return batch
    return _collate


def make_loader(entries: list[dict], cache, window: int, layers: list[int], handle: Stage1Handle,
                store: WTNeighborStore, mode: str, batch_size: int, shuffle: bool) -> DataLoader:
    var_id_to_pp = {e["var_id"]: int(e["row"]["pp"]) for e in entries}
    ds = Stage1Dataset(entries, cache, window, layers, meta_scaler=handle.meta_scaler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(store, mode, var_id_to_pp))


def _move(d: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def forward_batch(model: NeighborhoodFiLMModel, tokenizer: ResidueStructureTokenizer,
                  handle: Stage1Handle, batch: dict, device: str) -> tuple[dict, dict]:
    s1 = stage1_outputs(handle, batch, grad=False)     # Stage1 is ALWAYS frozen in this experiment
    raw = _move(batch["struct_raw"], device)
    s = tokenizer(raw)
    out = model(s1["h_base"], s, raw["distance"].float(), raw["offset"], raw["is_anchor"], raw["valid"], s1["y1"])
    return out, s1


def _select_monitor(metrics: dict) -> float:
    return float(metrics["subset"]) if metrics["subset"] == metrics["subset"] else float("-inf")


def _summarise(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict:
    m = compute_metrics(y, pred)
    by_group, by_group_n = {}, {}
    for g in ("missense", "synonymous", "indel"):
        mask = groups == g
        by_group_n[g] = int(mask.sum())
        if mask.sum() >= 2:
            by_group[g] = compute_metrics(y[mask], pred[mask])["spearman"]
    scored = [by_group[g] for g in ("missense", "indel") if g in by_group]
    m["by_group"], m["by_group_n"] = by_group, by_group_n
    m["subset"] = float(np.mean(scored)) if scored else float("nan")
    m["mae"] = float(np.mean(np.abs(y - pred)))
    return m


@torch.no_grad()
def evaluate(model: NeighborhoodFiLMModel, tokenizer: ResidueStructureTokenizer, handle: Stage1Handle,
            loader: DataLoader, device: str) -> dict:
    model.eval(); tokenizer.eval()
    y, p2, p1, groups, var_ids = [], [], [], [], []
    ent, ent_norm, anc_m, loc_m, nonloc_m, delta_l, gamma_l, beta_l = [], [], [], [], [], [], [], []
    for batch in loader:
        out, s1 = forward_batch(model, tokenizer, handle, batch, device)
        y.append(batch["label"].numpy()); p2.append(out["pred"].cpu().numpy()); p1.append(s1["y1"].cpu().numpy())
        groups += [group_of(t) for t in batch["edit_type"]]
        var_ids += batch["var_id"]
        ent.append(out["entropy"].cpu().numpy()); ent_norm.append(out["entropy_norm"].cpu().numpy())
        anc_m.append(out["anchor_mass"].cpu().numpy()); loc_m.append(out["local_mass"].cpu().numpy())
        nonloc_m.append(out["nonlocal_mass"].cpu().numpy()); delta_l.append(out["delta"].cpu().numpy())
        gamma_l.append(out["gamma"].cpu().numpy()); beta_l.append(out["beta"].cpu().numpy())
    y, p2, p1 = np.concatenate(y), np.concatenate(p2), np.concatenate(p1)
    groups = np.array(groups) if groups else np.array(["missense"] * len(y))  # batch always carries edit via caller
    m2, m1 = _summarise(y, p2, groups), _summarise(y, p1, groups)
    return {
        "stage2": m2, "stage1": m1, "var_ids": var_ids, "labels": y.tolist(), "preds_stage2": p2.tolist(),
        "preds_stage1": p1.tolist(), "groups": groups.tolist(),
        "entropy_mean": float(np.nanmean(np.concatenate(ent))), "entropy_norm_mean": float(np.nanmean(np.concatenate(ent_norm))),
        "anchor_mass_mean": float(np.mean(np.concatenate(anc_m))), "local_mass_mean": float(np.mean(np.concatenate(loc_m))),
        "nonlocal_mass_mean": float(np.mean(np.concatenate(nonloc_m))),
        "delta_abs_mean": float(np.mean(np.abs(np.concatenate(delta_l)))), "delta_rms": float(np.sqrt(np.mean(np.concatenate(delta_l) ** 2))),
        "gamma_std": float(np.concatenate(gamma_l).std()), "beta_std": float(np.concatenate(beta_l).std()),
    }


def grad_coverage(model: nn.Module, tokenizer: nn.Module) -> dict:
    """{name: bool} -- whether this parameter received a nonzero gradient on the last backward."""
    out = {}
    for prefix, m in (("model", model), ("tokenizer", tokenizer)):
        for n, p in m.named_parameters():
            out[f"{prefix}.{n}"] = bool(p.grad is not None and p.grad.abs().sum() > 0)
    return out


def train_neighborhood(
    *, model: NeighborhoodFiLMModel, tokenizer: ResidueStructureTokenizer, handle: Stage1Handle,
    loaders: dict[str, DataLoader], cfg: dict, device: str, out_dir: Path, seed: int,
) -> dict:
    """Epoch 0 is evaluated and is an explicit best-candidate (strict >, same convention as
    the rest of this repo). Both best and last checkpoints are kept. Per-epoch validation
    predictions are saved with var_id. Test is never touched here."""
    set_all_seeds(seed)
    model.to(device); tokenizer.to(device)
    params = list(model.parameters()) + list(tokenizer.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.HuberLoss(delta=float(cfg["huber_delta"]), reduction="mean")

    init_state = {"model": copy.deepcopy(model.state_dict()), "tokenizer": copy.deepcopy(tokenizer.state_dict())}
    ep0 = evaluate(model, tokenizer, handle, loaders["val"], device)
    score0 = _select_monitor(ep0["stage2"])
    val_preds_by_epoch = {0: {"var_id": ep0["var_ids"], "label": ep0["labels"], "pred": ep0["preds_stage2"],
                              "stage1_pred": ep0["preds_stage1"], "group": ep0["groups"]}}

    history = [{
        "epoch": 0, "val_stage2_subset": score0, "val_stage2_spearman": ep0["stage2"]["spearman"],
        "val_stage2_rmse": ep0["stage2"]["rmse"], "val_stage2_mae": ep0["stage2"]["mae"],
        "val_stage1_subset": ep0["stage1"]["subset"], "val_by_group_spearman": ep0["stage2"]["by_group"],
        "val_by_group_n": ep0["stage2"]["by_group_n"], "delta_abs_mean": ep0["delta_abs_mean"],
        "delta_rms": ep0["delta_rms"], "gamma_std": ep0["gamma_std"], "beta_std": ep0["beta_std"],
        "entropy_mean": ep0["entropy_mean"], "entropy_norm_mean": ep0["entropy_norm_mean"],
        "anchor_mass_mean": ep0["anchor_mass_mean"], "local_mass_mean": ep0["local_mass_mean"],
        "nonlocal_mass_mean": ep0["nonlocal_mass_mean"], "train_task_loss": None, "seconds": 0.0,
    }]
    best = {"score": score0, "epoch": 0}
    best_state = init_state
    last_state = init_state
    # Two snapshots: step 1 of epoch 1 is dominated by the zero-init cascade from FiLM/head's
    # last layers (delta==0 exactly at init blocks almost every upstream gradient -- see
    # README/REPORT discussion; this is expected, not E1-specific). The first step of epoch 2
    # is AFTER at least one optimizer update has moved head's last layer off zero, so by then
    # the real E1-vs-E2 Q/K-gradient distinction (one valid key vs several) is visible.
    grad_cov_step1, grad_cov_epoch2_step1 = None, None

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train(); tokenizer.train()
        t0 = time.time()
        task_sum, n = 0.0, 0
        for batch in loaders["train"]:
            out, _ = forward_batch(model, tokenizer, handle, batch, device)
            loss = loss_fn(out["pred"], batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 1 and grad_cov_step1 is None:
                grad_cov_step1 = grad_coverage(model, tokenizer)
            if epoch == 2 and grad_cov_epoch2_step1 is None:
                grad_cov_epoch2_step1 = grad_coverage(model, tokenizer)
            torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
            optimizer.step()
            task_sum += float(loss.detach()); n += 1
        seconds = round(time.time() - t0, 1)
        last_state = {"model": copy.deepcopy(model.state_dict()), "tokenizer": copy.deepcopy(tokenizer.state_dict())}

        val = evaluate(model, tokenizer, handle, loaders["val"], device)
        val_preds_by_epoch[epoch] = {"var_id": val["var_ids"], "label": val["labels"], "pred": val["preds_stage2"],
                                     "stage1_pred": val["preds_stage1"], "group": val["groups"]}
        score = _select_monitor(val["stage2"])
        improved = score > best["score"] + float(cfg["min_delta"])
        if improved:
            best = {"score": score, "epoch": epoch}
            best_state = {"model": copy.deepcopy(model.state_dict()), "tokenizer": copy.deepcopy(tokenizer.state_dict())}
        history.append({
            "epoch": epoch, "train_task_loss": task_sum / max(n, 1), "val_stage2_subset": score,
            "val_stage2_spearman": val["stage2"]["spearman"], "val_stage2_rmse": val["stage2"]["rmse"],
            "val_stage2_mae": val["stage2"]["mae"], "val_stage1_subset": val["stage1"]["subset"],
            "val_by_group_spearman": val["stage2"]["by_group"], "val_by_group_n": val["stage2"]["by_group_n"],
            "delta_abs_mean": val["delta_abs_mean"], "delta_rms": val["delta_rms"],
            "gamma_std": val["gamma_std"], "beta_std": val["beta_std"], "entropy_mean": val["entropy_mean"],
            "entropy_norm_mean": val["entropy_norm_mean"], "anchor_mass_mean": val["anchor_mass_mean"],
            "local_mass_mean": val["local_mass_mean"], "nonlocal_mass_mean": val["nonlocal_mass_mean"],
            "seconds": seconds, "improved": bool(improved), "best_epoch_so_far": best["epoch"],
            "best_score_so_far": best["score"], "patience_counter": epoch - best["epoch"],
        })
        if not improved and (epoch - best["epoch"]) >= int(cfg["patience"]):
            stop_reason = "patience"
            break
    else:
        stop_reason = "max_epochs"

    return {"history": history, "best": best, "best_state": best_state, "last_state": last_state,
            "last_epoch": history[-1]["epoch"], "stop_reason": stop_reason,
            "val_preds_by_epoch": val_preds_by_epoch, "grad_coverage_step1": grad_cov_step1,
            "grad_coverage_epoch2_step1": grad_cov_epoch2_step1, "init_state": init_state}
