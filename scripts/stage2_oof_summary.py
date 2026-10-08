"""Summarises the out-of-fold residual experiment (validation only; test is never read).

1. Residual distributions: train in-sample (shipped Stage 1), train out-of-fold, validation.
2. Fold models: validation subset of each fold's Stage 1 vs the shipped checkpoint.
3. Stage 2 runs: in-sample target (runs/stage2_tau0.1, runs/stage2_reverse) vs OOF target
   (runs/stage2_oof) for the same four models, seed 42.
Outputs in analysis/stage2_oof/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "runs/stage1_oof/unified_reference_delta/W10/seed44"
OUT = ROOT / "analysis/stage2_oof"
OUT.mkdir(parents=True, exist_ok=True)

# 1. residual distributions
oof = pd.read_csv(OOF / "oof_y1.csv")
val = pd.read_csv(ROOT / "analysis/stage2_diag/residual_per_sample.csv")
val = val[val.split == "val"]
rows = []
for name, d, y, y1 in (("train_in_sample", oof, "label", "y1_full"), ("train_oof", oof, "label", "y1_oof"),
                       ("val", val, "y", "y1")):
    for t, g in list(d.groupby("type")) + [("all", d)]:
        r = g[y] - g[y1]
        sub = {}
        for tt in ("missense", "indel"):
            gg = g[g.type == tt]
            if len(gg) > 2:
                sub[tt] = spearmanr(gg[y], gg[y1])[0]
        rows.append({"set": name, "type": t, "n": len(g), "resid_mean": r.mean(), "resid_median": r.median(),
                     "resid_rms": float(np.sqrt((r ** 2).mean())), "huber": float(np.mean(np.where(r.abs() <= 1, 0.5 * r ** 2, r.abs() - 0.5))),
                     "spearman_missense": sub.get("missense"), "spearman_indel": sub.get("indel")})
res = pd.DataFrame(rows)
res.to_csv(OUT / "residual_distributions.csv", index=False)
print(res.round(3).to_string())

# 2. fold models
folds = []
for k in range(5):
    vm = json.loads((OOF / f"fold{k}/val_metrics.json").read_text())
    cf = json.loads((OOF / f"fold{k}/config.json").read_text()) if (OOF / f"fold{k}/config.json").exists() else {}
    folds.append({"fold": k, "val_subset": vm["subset"], "best_epoch": cf.get("best_epoch"),
                  "n_train_rows": cf.get("n_train_rows"), "train_seconds": cf.get("train_seconds")})
folds = pd.DataFrame(folds)
folds.to_csv(OUT / "fold_models.csv", index=False)
print(folds.round(4).to_string())

# 3. Stage 2 comparison
RUNS = {"single_query": ("runs/stage2_tau0.1/single_query/seed42", "runs/stage2_oof/single_query/seed42"),
        "nine_query": ("runs/stage2_tau0.1/nine_query/seed42", "runs/stage2_oof/nine_query/seed42"),
        "r0_struct_mean": ("runs/stage2_reverse/r0_struct_mean/seed42", "runs/stage2_oof/r0_struct_mean/seed42"),
        "r1_seq_query": ("runs/stage2_reverse/r1_seq_query/seed42", "runs/stage2_oof/r1_seq_query/seed42")}
cmp_rows, curves = [], {}
for model, (d_in, d_oof) in RUNS.items():
    for target, d in (("in_sample", d_in), ("oof", d_oof)):
        p = ROOT / d / "history.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        hist = [x for x in h["history"] if x["epoch"] >= 1]
        b = next(x for x in hist if x["epoch"] == h["best"]["epoch"])
        cmp_rows.append({"model": model, "target": target, "best_epoch": h["best"]["epoch"], "epochs_run": hist[-1]["epoch"],
                         "val_subset_best": h["best"]["score"], "val_stage1_subset": b["val_stage1_subset"],
                         "gain_vs_stage1": h["best"]["score"] - b["val_stage1_subset"],
                         "val_missense_best": b["val_by_group_spearman"]["missense"],
                         "val_indel_best": b["val_by_group_spearman"]["indel"],
                         "val_rmse_best": b["val_stage2_rmse"], "val_subset_last": hist[-1]["val_stage2_subset"],
                         "train_loss_ep1": hist[0]["train_task_loss"], "train_loss_last": hist[-1]["train_task_loss"],
                         "delta_abs_mean_best": b["delta_abs_mean"]})
        curves[(model, target)] = pd.DataFrame([{"epoch": x["epoch"], "val": x["val_stage2_subset"]} for x in hist])
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(OUT / "stage2_in_sample_vs_oof_val.csv", index=False)
print(cmp_df.round(4).to_string())

fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
for ax, model in zip(axes, RUNS):
    for target, ls in (("in_sample", "--"), ("oof", "-")):
        if (model, target) in curves:
            c = curves[(model, target)]
            ax.plot(c.epoch, c.val, ls=ls, marker=".", label=f"{target} target")
    ax.axhline(cmp_df.val_stage1_subset.iloc[0], color="k", lw=0.8, ls=":", label="Stage 1 alone")
    ax.set_title(model, fontsize=9); ax.set_xlabel("epoch")
axes[0].set_ylabel("val subset Spearman"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "val_curves_in_sample_vs_oof.png", dpi=130)
