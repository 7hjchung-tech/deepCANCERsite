"""Diagnostic B: is the correction direction useful but oversized?  y_alpha = y_stage1 + alpha*delta_y

Scope: Stage 1 baseline, R0 (reference), R1, R2, alpha in {0, 0.1, 0.25, 0.5, 1.0}, validation
only, no retraining. Checkpoints are limited to the saved BEST checkpoint (epoch 1 for all
three models) -- no LAST-epoch weights were saved by train_stage2_reverse.py, so last-epoch
alpha blending is not possible without retraining and is therefore not attempted here (see
analysis/stage2_diag2/REPORT_B.md's limitations section). alpha is chosen from this fixed grid
only; no additional search, no negative alpha.

Outputs -> analysis/stage2_diag2/
  alpha_metrics.csv         model x alpha: val Huber/MAE/RMSE/overall+subset+by-group spearman
  alpha_bootstrap.csv       position-grouped paired bootstrap of (subset metric at alpha) -
                            (subset metric at alpha=0), for every alpha > 0
  alpha_delta_diagnostics.csv   delta_y mean/std/rms, corr(delta_y, Stage1 residual), fraction
                            of rows where |error| shrinks vs Stage1, train AND val
  alpha_curves.png
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

OUT = ROOT / "analysis/stage2_diag2"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"R0_struct_mean": "runs/stage2_reverse/r0_struct_mean/seed42",
          "R1_seq_query": "runs/stage2_reverse/r1_seq_query/seed42",
          "R2_multihead": "runs/stage2_reverse/r2_multihead_query/seed42"}
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]
B = 2000


def huber(y, p, delta=1.0):
    r = np.abs(y - p)
    return float(np.mean(np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))))


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def subset_metric(fn, y, p, g):
    vals = [fn(y[g == t], p[g == t]) for t in ("missense", "indel") if (g == t).sum() >= 2]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    st = load_setup(DEFAULT_STAGE1, "cuda", splits=("train", "val"))
    ld = {"train": loader_for(st, st.by_split["train"], 128, shuffle=False),
          "val": loader_for(st, st.by_split["val"], 128, shuffle=False)}
    y = {s: np.array([float(e["row"]["z_score_D4_D14"]) for e in st.by_split[s]]) for s in ld}
    g = {s: np.array([group_of(e["edit"].edit_type) for e in st.by_split[s]]) for s in ld}
    pos = {s: np.array([int(e["row"]["pp"]) for e in st.by_split[s]]) for s in ld}

    rows_metric, rows_delta = [], []
    bootstrap_rows = []
    for name, d in MODELS.items():
        m, tok, ck = load_trained_any(st, Path(ROOT / d / "best_stage2.pt"))
        cfgj = json.loads((ROOT / d / "config.json").read_text())
        for s in ("train", "val"):
            # delta_y straight from the model's own output (verified below against y_final - y1)
            preds, y1s, deltas, ids = [], [], [], []
            for batch in ld[s]:
                with torch.no_grad():
                    from src.stage2.engine import _move
                    from src.stage2.stage1_adapter import stage1_outputs
                    s1 = stage1_outputs(st.handle, batch, grad=False)
                    S = tok(_move(batch["struct_raw"], "cuda"))
                    out = m(S, s1["K"], s1["V"], s1["valid"], batch["type_id"].to("cuda"), s1["y1"])
                preds.append(out["pred"].cpu().numpy()); y1s.append(s1["y1"].cpu().numpy())
                deltas.append(out["delta"].cpu().numpy()); ids += batch["var_id"]
            assert ids == [e["var_id"] for e in st.by_split[s]]
            pred, y1, delta = np.concatenate(preds), np.concatenate(y1s), np.concatenate(deltas)
            assert np.allclose(pred, y1 + delta, atol=1e-4), "y_final != y1 + delta_y for a saved checkpoint"

            resid = y[s] - y1
            rows_delta.append({
                "model": name, "split": s, "n": len(delta), "delta_mean": float(delta.mean()),
                "delta_std": float(delta.std()), "delta_rms": float(np.sqrt(np.mean(delta ** 2))),
                "corr_delta_resid_pearson": float(np.corrcoef(delta, resid)[0, 1]),
                "corr_delta_resid_spearman": spearman(delta, resid),
                "frac_abs_error_reduced": float((np.abs(y[s] - (y1 + delta)) < np.abs(resid)).mean()),
                "mean_abs_error_before": float(np.abs(resid).mean()),
                "mean_abs_error_after": float(np.abs(y[s] - (y1 + delta)).mean()),
            })
            for grp in ("missense", "synonymous", "indel"):
                mask = g[s] == grp
                if mask.sum() >= 2:
                    rows_delta[-1][f"corr_delta_resid_pearson_{grp}"] = float(np.corrcoef(delta[mask], resid[mask])[0, 1])
                    rows_delta[-1][f"frac_abs_error_reduced_{grp}"] = float(
                        (np.abs(y[s][mask] - (y1 + delta)[mask]) < np.abs(resid[mask])).mean())

            for a in ALPHAS:
                p_a = y1 + a * delta
                row = {"model": name, "split": s, "alpha": a, "epoch": cfgj["best_epoch"],
                       "huber_all": huber(y[s], p_a), "mae_all": float(np.mean(np.abs(y[s] - p_a))),
                       "rmse_all": float(np.sqrt(np.mean((y[s] - p_a) ** 2))), "spearman_all": spearman(y[s], p_a),
                       "subset_spearman": subset_metric(spearman, y[s], p_a, g[s])}
                for grp in ("missense", "synonymous", "indel"):
                    mask = g[s] == grp
                    if mask.sum() >= 2:
                        row[f"spearman_{grp}"] = spearman(y[s][mask], p_a[mask])
                        row[f"huber_{grp}"] = huber(y[s][mask], p_a[mask])
                        row[f"mae_{grp}"] = float(np.mean(np.abs(y[s][mask] - p_a[mask])))
                sub = np.isin(g[s], ["missense", "indel"])
                row["huber_subset_pooled"] = huber(y[s][sub], p_a[sub])
                row["mae_subset_pooled"] = float(np.mean(np.abs(y[s][sub] - p_a[sub])))
                rows_metric.append(row)

            if s == "val":
                upos = np.unique(pos["val"])
                rows_of = {pp: np.where(pos["val"] == pp)[0] for pp in upos}
                rng = np.random.default_rng(0)
                boots = [rng.choice(upos, size=len(upos), replace=True) for _ in range(B)]
                n_failed = 0
                for a in ALPHAS:
                    if a == 0.0:
                        continue
                    p_a = y1 + a * delta
                    base = subset_metric(spearman, y["val"], y1, g["val"])          # alpha=0 (Stage1)
                    point = subset_metric(spearman, y["val"], p_a, g["val"]) - base
                    diffs = []
                    for samp in boots:
                        idx = np.concatenate([rows_of[pp] for pp in samp])
                        yb, gb = y["val"][idx], g["val"][idx]
                        if (gb == "missense").sum() < 2 or (gb == "indel").sum() < 2:
                            n_failed += 1
                            continue
                        diffs.append(subset_metric(spearman, yb, p_a[idx], gb) - subset_metric(spearman, yb, y1[idx], gb))
                    diffs = np.array(diffs)
                    bootstrap_rows.append({"model": name, "alpha": a, "point_gain_subset": point,
                                           "ci95_lo": float(np.quantile(diffs, 0.025)) if len(diffs) else None,
                                           "ci95_hi": float(np.quantile(diffs, 0.975)) if len(diffs) else None,
                                           "p_gain_le_0": float((diffs <= 0).mean()) if len(diffs) else None,
                                           "n_valid_replicates": int(len(diffs)), "n_failed_replicates": n_failed})

    pd.DataFrame(rows_metric).to_csv(OUT / "alpha_metrics.csv", index=False)
    pd.DataFrame(rows_delta).to_csv(OUT / "alpha_delta_diagnostics.csv", index=False)
    bdf = pd.DataFrame(bootstrap_rows)
    bdf.to_csv(OUT / "alpha_bootstrap.csv", index=False)
    print(bdf.round(4).to_string())

    mdf = pd.DataFrame(rows_metric)
    v = mdf[mdf.split == "val"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colors = {"R0_struct_mean": "tab:green", "R1_seq_query": "tab:blue", "R2_multihead": "tab:red"}
    for name in MODELS:
        d = v[v.model == name].sort_values("alpha")
        axes[0].plot(d.alpha, d.huber_all, marker="o", color=colors[name], label=name)
        axes[1].plot(d.alpha, d.mae_all, marker="o", color=colors[name], label=name)
        axes[2].plot(d.alpha, d.subset_spearman, marker="o", color=colors[name], label=name)
    for ax, t in zip(axes, ("val Huber (all rows)", "val MAE (all rows)", "val subset Spearman")):
        ax.set_xlabel("alpha"); ax.set_title(t)
    axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "alpha_curves.png", dpi=130)
    print(f"[diag2b] wrote {OUT}")


if __name__ == "__main__":
    main()
