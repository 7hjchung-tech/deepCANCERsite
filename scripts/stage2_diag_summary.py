"""Collects validation-only results of the four Stage 2 variants (seed 42, Stage 1 seed44 W10)
into analysis/stage2_diag/model_comparison_val.csv and training_curves.png. Test is not read."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/stage2_diag"
RUNS = {
    "forward single_query (tau0.1)": ("runs/stage2_tau0.1/single_query/seed42", "runs/stage2_diag/instrumented/single_query"),
    "forward nine_query (tau0.1)": ("runs/stage2_tau0.1/nine_query/seed42", "runs/stage2_diag/instrumented/nine_query"),
    "R0 struct mean -> token FiLM": ("runs/stage2_reverse/r0_struct_mean/seed42", None),
    "R1 seq Q -> struct K/V -> token FiLM": ("runs/stage2_reverse/r1_seq_query/seed42", None),
}
rows, curves = [], {}
for name, (d, instr) in RUNS.items():
    h = json.loads((ROOT / d / "history.json").read_text())
    hist = [x for x in h["history"] if x["epoch"] >= 1]
    best = h["best"]
    b = next(x for x in hist if x["epoch"] == best["epoch"])
    cfg = json.loads((ROOT / d / "config.json").read_text())
    tp = cfg["trainable_params"]
    rows.append({"model": name, "stage2_params": tp.get("stage2_head", tp.get("stage2")), "tokenizer_params": tp["tokenizer"],
                 "best_epoch": best["epoch"], "epochs_run": hist[-1]["epoch"], "val_subset_best": best["score"],
                 "val_stage1_subset": b["val_stage1_subset"], "val_minus_stage1_best": best["score"] - b["val_stage1_subset"],
                 "val_subset_last": hist[-1]["val_stage2_subset"], "val_missense_best": b["val_by_group_spearman"]["missense"],
                 "val_indel_best": b["val_by_group_spearman"]["indel"], "val_rmse_best": b["val_stage2_rmse"],
                 "delta_abs_mean_best": b["delta_abs_mean"], "delta_abs_mean_last": hist[-1]["delta_abs_mean"]})
    curves[name] = pd.DataFrame([{"epoch": x["epoch"], "val": x["val_stage2_subset"], "train_loss": x["train_task_loss"]} for x in hist])
df = pd.DataFrame(rows)
df.to_csv(OUT / "model_comparison_val.csv", index=False)
print(df.round(4).to_string())

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for name, c in curves.items():
    ax[0].plot(c.epoch, c.val, marker=".", label=name)
    ax[1].plot(c.epoch, c.train_loss, marker=".", label=name)
s1 = rows[0]["val_stage1_subset"]
ax[0].axhline(s1, color="k", ls="--", lw=1, label="Stage 1 alone")
ax[0].set_ylabel("val subset Spearman"); ax[1].set_ylabel("train Huber (Stage 2)")
for a in ax:
    a.set_xlabel("epoch")
ax[0].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=130)
