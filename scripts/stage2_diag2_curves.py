"""Diagnostic A: do numeric error and the rank metric move together or apart, epoch by epoch?

Scope (per the 2026-10 request): Stage 1 baseline, R1, R2, and R0 as a reference -- no new
models, no joint fine-tuning, no OOF retraining. Only runs/stage2_reverse/{r0,r1,r2}/seed42
(already trained) are read. Validation only beyond the inference noted below; test is never
loaded.

What is available without retraining, and what is not
  * val rmse/mae/spearman/subset/by-group, EVERY epoch 0..11: already logged in history.json
    (computed by src.stage2.engine.evaluate at the live model of that epoch) -- no inference
    needed, used as-is.
  * val Huber, train Huber/MAE/spearman/subset/by-group (eval mode): NOT logged per epoch.
    Computable only where model weights exist, i.e. the saved BEST checkpoint (epoch 1 for all
    three models). The LAST epoch (11) has no saved weights for R0/R1/R2, so last-epoch Huber
    and train-side metrics are left as "unavailable" rather than reconstructed by retraining.
  * train_task_loss in history.json is the train-mode Huber, averaged over an epoch's
    minibatches as parameters are updated mid-epoch -- not the loss of one frozen snapshot.
    It is plotted but not treated as equal to an eval-mode train Huber.

Outputs -> analysis/stage2_diag2/
  epoch_metrics_val.csv        every epoch, every model, all val fields from history.json
  best_epoch_full_metrics.csv  eval-mode inference at the best (saved) checkpoint: train AND
                               val Huber/MAE/RMSE/spearman/subset/by-group, Stage1 alongside
  best_vs_last_summary.csv     best vs last epoch val metrics (aggregate only; last has no
                               Huber / train side, flagged explicitly)
  curves_huber_mae.png, curves_spearman.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage1.engine import group_of  # noqa: E402
from src.stage2.diagnostics import DEFAULT_STAGE1, load_setup, load_trained_any, loader_for  # noqa: E402
from src.stage2.engine import _summarise, evaluate  # noqa: E402
from src.stage2.stage1_adapter import stage1_outputs  # noqa: E402

OUT = ROOT / "analysis/stage2_diag2"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"R0_struct_mean": "runs/stage2_reverse/r0_struct_mean/seed42",
          "R1_seq_query": "runs/stage2_reverse/r1_seq_query/seed42",
          "R2_multihead": "runs/stage2_reverse/r2_multihead_query/seed42"}


def huber(y: np.ndarray, p: np.ndarray, delta: float = 1.0) -> float:
    r = np.abs(y - p)
    return float(np.mean(np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))))


def full_metrics(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> dict:
    """all-rows + by-group + pooled-subset (missense+indel rows) Huber/MAE/RMSE/spearman."""
    m = _summarise(y, p, g)   # rmse, spearman, pearson, by_group, by_group_n, subset(=mean of 2 spearmans), mae
    out = {"huber_all": huber(y, p), "mae_all": m["mae"], "rmse_all": m["rmse"], "spearman_all": m["spearman"],
           "subset_spearman": m["subset"]}
    for grp in ("missense", "synonymous", "indel"):
        mask = g == grp
        out[f"n_{grp}"] = int(mask.sum())
        if mask.sum() >= 2:
            out[f"spearman_{grp}"] = m["by_group"].get(grp)
            out[f"huber_{grp}"] = huber(y[mask], p[mask])
            out[f"mae_{grp}"] = float(np.mean(np.abs(y[mask] - p[mask])))
            out[f"rmse_{grp}"] = float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))
    sub = np.isin(g, ["missense", "indel"])
    out["huber_subset_pooled"] = huber(y[sub], p[sub])
    out["mae_subset_pooled"] = float(np.mean(np.abs(y[sub] - p[sub])))
    out["rmse_subset_pooled"] = float(np.sqrt(np.mean((y[sub] - p[sub]) ** 2)))
    out["n_subset_pooled"] = int(sub.sum())
    return out


def main() -> None:
    st = load_setup(DEFAULT_STAGE1, "cuda", splits=("train", "val"))
    ld = {"train": loader_for(st, st.by_split["train"], 128, shuffle=False),
          "val": loader_for(st, st.by_split["val"], 128, shuffle=False)}
    y = {s: np.array([float(e["row"]["z_score_D4_D14"]) for e in st.by_split[s]]) for s in ld}
    g = {s: np.array([group_of(e["edit"].edit_type) for e in st.by_split[s]]) for s in ld}
    var_ids = {s: [e["var_id"] for e in st.by_split[s]] for s in ld}

    # Stage 1 baseline, eval mode, ID-aligned to the row order the loaders use (shuffle=False)
    p1 = {}
    for s, loader in ld.items():
        preds, ids = [], []
        with torch.no_grad():
            for batch in loader:
                preds.append(stage1_outputs(st.handle, batch, grad=False)["y1"].cpu().numpy())
                ids += batch["var_id"]
        assert ids == var_ids[s], f"{s}: loader row order drifted from the cached var_id order"
        p1[s] = np.concatenate(preds)
    stage1_full = {s: full_metrics(y[s], p1[s], g[s]) for s in ld}
    (OUT / "stage1_baseline_full_metrics.json").write_text(json.dumps(stage1_full, indent=2))

    # 1) every logged val epoch, every model (no inference needed)
    epoch_rows = []
    for name, d in MODELS.items():
        h = json.loads((ROOT / d / "history.json").read_text())
        for x in h["history"]:
            if x["epoch"] == 0:
                continue   # ep0_row in these files only carries a partial field set; epoch 1 IS the first real log
            row = {"model": name, "epoch": x["epoch"], "is_best": x["epoch"] == h["best"]["epoch"],
                   "is_last": x["epoch"] == h["history"][-1]["epoch"], "train_huber_trainmode": x.get("train_task_loss"),
                   "val_rmse": x["val_stage2_rmse"], "val_mae": x.get("val_stage2_mae"),
                   "val_spearman_all": x["val_stage2_spearman"], "val_subset_spearman": x["val_stage2_subset"],
                   "val_stage1_subset": x["val_stage1_subset"], "delta_abs_mean": x.get("delta_abs_mean")}
            bg, bn = x.get("val_by_group_spearman", {}), x.get("val_by_group_n", {})
            for grp in ("missense", "synonymous", "indel"):
                row[f"val_spearman_{grp}"] = bg.get(grp)
                row[f"val_n_{grp}"] = bn.get(grp)
            epoch_rows.append(row)
    edf = pd.DataFrame(epoch_rows)
    edf.to_csv(OUT / "epoch_metrics_val.csv", index=False)

    # 2) eval-mode inference at the BEST checkpoint (weights exist) -- train AND val, full metrics
    best_rows = []
    for name, d in MODELS.items():
        m, tok, ck = load_trained_any(st, Path(ROOT / d / "best_stage2.pt"))
        cfgj = json.loads((ROOT / d / "config.json").read_text())
        for s in ("train", "val"):
            r = evaluate(m, tok, st.handle, ld[s], "cuda")
            p2 = np.array(r["preds_stage2"])
            assert np.array(r["labels"]) is not None
            fm = full_metrics(y[s], p2, g[s])
            fm.update({"model": name, "split": s, "epoch": cfgj["best_epoch"]})
            best_rows.append(fm)
    bdf = pd.DataFrame(best_rows)
    cols = ["model", "split", "epoch"] + [c for c in bdf.columns if c not in ("model", "split", "epoch")]
    bdf = bdf[cols]
    bdf.to_csv(OUT / "best_epoch_full_metrics.csv", index=False)

    # Stage1 rows alongside, same format, for direct best-epoch comparison
    s1_rows = [{"model": "Stage1_baseline", "split": s, "epoch": None, **stage1_full[s]} for s in ("train", "val")]
    pd.concat([bdf, pd.DataFrame(s1_rows)[cols]], ignore_index=True).to_csv(
        OUT / "best_epoch_full_metrics_with_stage1.csv", index=False)

    # 3) best vs last (val only, aggregate from history -- no weights needed/available for "last")
    rows = []
    for name, d in MODELS.items():
        h = json.loads((ROOT / d / "history.json").read_text())
        hist = {x["epoch"]: x for x in h["history"] if x["epoch"] >= 1}
        be, le = h["best"]["epoch"], h["history"][-1]["epoch"]
        for tag, ep in (("best", be), ("last", le)):
            x = hist[ep]
            bg = x.get("val_by_group_spearman", {})
            rows.append({"model": name, "which": tag, "epoch": ep,
                        "val_rmse": x["val_stage2_rmse"], "val_mae": x.get("val_stage2_mae"),
                        "val_spearman_all": x["val_stage2_spearman"], "val_subset_spearman": x["val_stage2_subset"],
                        "val_subset_minus_stage1": x["val_stage2_subset"] - x["val_stage1_subset"],
                        "val_spearman_missense": bg.get("missense"), "val_spearman_indel": bg.get("indel"),
                        "val_spearman_synonymous": bg.get("synonymous"),
                        "val_huber": None if tag == "last" else None,   # filled below for 'best' only
                        "note": "" if tag == "best" else "no saved weights; train-side and val Huber unavailable"})
    bl = pd.DataFrame(rows)
    # fill val_huber for 'best' rows from best_epoch_full_metrics.csv (already computed above)
    bmap = bdf[bdf.split == "val"].set_index("model")["huber_all"].to_dict()
    bl.loc[bl.which == "best", "val_huber"] = bl.loc[bl.which == "best", "model"].map(bmap)
    bl.to_csv(OUT / "best_vs_last_summary.csv", index=False)
    print(bl.round(4).to_string())

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    colors = {"R0_struct_mean": "tab:green", "R1_seq_query": "tab:blue", "R2_multihead": "tab:red"}
    for name in MODELS:
        d = edf[edf.model == name].sort_values("epoch")
        axes[0].plot(d.epoch, d.train_huber_trainmode, color=colors[name], ls="--", marker=".", label=f"{name} train Huber (train-mode)")
        axes[0].plot(d.epoch, d.val_mae, color=colors[name], ls="-", marker="o", label=f"{name} val MAE")
    axes[0].axhline(stage1_full["val"]["mae_all"], color="k", ls=":", lw=1, label="Stage1 val MAE")
    axes[0].set_ylabel("loss / MAE"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=6)
    axes[0].set_title("train Huber (train-mode) vs val MAE")
    for name in MODELS:
        d = edf[edf.model == name].sort_values("epoch")
        axes[1].plot(d.epoch, d.val_subset_spearman, color=colors[name], marker="o", label=f"{name} val subset")
    axes[1].axhline(stage1_full["val"]["subset_spearman"], color="k", ls=":", lw=1, label="Stage1 val subset")
    axes[1].set_ylabel("val subset Spearman"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "curves_huber_mae.png", dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharex=True)
    for ax, metric, title in zip(axes, ("val_spearman_missense", "val_spearman_synonymous", "val_spearman_indel"),
                                 ("missense", "synonymous", "indel")):
        for name in MODELS:
            d = edf[edf.model == name].sort_values("epoch")
            ax.plot(d.epoch, d[metric], color=colors[name], marker=".", label=name)
        ax.axhline(stage1_full["val"].get(f"spearman_{title}", float("nan")), color="k", ls=":", lw=1)
        ax.set_title(title); ax.set_xlabel("epoch")
    axes[0].set_ylabel("val Spearman by type"); axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "curves_spearman.png", dpi=130)
    print(f"[diag2a] wrote {OUT}")


if __name__ == "__main__":
    main()
