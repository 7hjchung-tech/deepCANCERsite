"""analyze_stage1_sweep.py -- Stage 1 multi-seed sweep analysis.

Reads runs/stage1_v2/<model>/W<window>/shipped_split/seed<seed>/{history.json,
test_metrics.json,config.json} written by run_stage1_sweep.py and produces:

    <out>/summary_by_run.csv
    <out>/summary_by_configuration.csv
    <out>/best_epoch_analysis.csv
    <out>/fixed_budget_comparison.csv     (extra: section 8's fixed-budget table)
    <out>/experiment_manifest.json
    <out>/analysis_report.md
    <out>/plots/runs/<run_id>.png                       (per-run learning curves)
    <out>/plots/aggregate/<model>_W<window>.png          (seed-aggregate curves)
    <out>/plots/aggregate/<model>_W<window>_valid_n.csv  (n(epoch) used for the CI band)
    <out>/plots/comparison/*.png                         (cross-config comparisons)

Read-only with respect to runs/: never writes into a run's own directory,
never re-evaluates test metrics, never picks a checkpoint by test score.

Usage:
    python analyze_stage1_sweep.py --runs-root runs/stage1_v2 --out analysis/stage1_v2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MODEL_MODES = ("paired_delta", "branched_projection", "unified_reference_delta")
MODEL_COLORS = {"paired_delta": "#4C72B0", "branched_projection": "#DD8452", "unified_reference_delta": "#55A868"}
WINDOW_COLORS = {5: "#4C72B0", 10: "#DD8452", 20: "#55A868"}
FIXED_EPOCHS = [10, 20, 30, 40, 60, 80]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_run(run_dir: Path) -> dict | None:
    hist_path, test_path, cfg_path = run_dir / "history.json", run_dir / "test_metrics.json", run_dir / "config.json"
    if not (hist_path.exists() and test_path.exists() and cfg_path.exists()):
        return None
    hist = json.loads(hist_path.read_text())
    return {
        "history": hist["history"], "best": hist["best"], "stop_reason": hist.get("stop_reason"),
        "test": json.loads(test_path.read_text()), "config": json.loads(cfg_path.read_text()),
        "run_dir": str(run_dir),
    }


def discover_runs(root: Path, models, windows, seeds) -> dict[tuple, dict]:
    out = {}
    for m in models:
        for w in windows:
            for s in seeds:
                r = load_run(root / m / f"W{w}" / "shipped_split" / f"seed{s}")
                if r is not None:
                    out[(m, w, s)] = r
    return out


# ---------------------------------------------------------------------------
# per-run derived stats (feeds best_epoch_analysis.csv and the W10 deep-dive)
# ---------------------------------------------------------------------------
def run_derived_stats(r: dict) -> dict:
    df = pd.DataFrame(r["history"])
    best_epoch = r["best"]["epoch"]
    stopped_epoch = int(df["epoch"].iloc[-1])
    improved_epochs = df.loc[df["improved"] == True, "epoch"].tolist()  # noqa: E712
    last_improvement_epoch = improved_epochs[-1] if improved_epochs else None

    deltas = []
    prev_best = -np.inf
    for _, row in df.iterrows():
        if row["improved"] and np.isfinite(prev_best):
            deltas.append({"epoch": int(row["epoch"]), "delta": float(row["monitor_value"] - prev_best)})
        prev_best = row["best_score_so_far"]

    plateau = df.loc[df["epoch"] < best_epoch, "monitor_value"]
    plateau_mean = float(plateau.mean()) if len(plateau) else None
    best_minus_plateau = (r["best"]["score"] - plateau_mean) if plateau_mean is not None else None
    min_reset_delta = min((d["delta"] for d in deltas), default=None)

    return dict(
        best_epoch=best_epoch, stopped_epoch=stopped_epoch, stop_reason=r["stop_reason"],
        n_epochs=len(df), median_seconds_per_epoch=float(df["seconds"].median()),
        total_train_seconds=float(df["seconds"].sum()),
        last_improvement_epoch=last_improvement_epoch, n_patience_resets=len(improved_epochs),
        reset_epochs=improved_epochs, reset_deltas=deltas,
        plateau_mean_before_best=plateau_mean, best_minus_plateau=best_minus_plateau,
        min_reset_delta=min_reset_delta,
    )


# ---------------------------------------------------------------------------
# summary tables
# ---------------------------------------------------------------------------
def build_summary_by_run(runs: dict[tuple, dict]) -> pd.DataFrame:
    rows = []
    for (m, w, s), r in runs.items():
        d = run_derived_stats(r)
        df = pd.DataFrame(r["history"])
        best_row = df[df["epoch"] == d["best_epoch"]].iloc[0]
        test = r["test"]
        by_group, by_group_n = test.get("by_group", {}), test.get("by_group_n", {})
        rows.append({
            "model": m, "window": w, "seed": s,
            "best_epoch": d["best_epoch"], "stopped_epoch": d["stopped_epoch"], "stop_reason": d["stop_reason"],
            "n_epochs": d["n_epochs"], "median_seconds_per_epoch": d["median_seconds_per_epoch"],
            "total_train_seconds": d["total_train_seconds"],
            "last_improvement_epoch": d["last_improvement_epoch"], "n_patience_resets": d["n_patience_resets"],
            "best_minus_plateau": d["best_minus_plateau"], "min_reset_delta": d["min_reset_delta"],
            "val_spearman_best": best_row["val_spearman"], "val_pearson_best": best_row["val_pearson"],
            "val_rmse_best": best_row["val_rmse"], "val_subset_best": best_row["val_subset_spearman"],
            "test_spearman": test.get("spearman"), "test_pearson": test.get("pearson"),
            "test_rmse": test.get("rmse"), "test_subset": test.get("subset"), "test_n": test.get("n"),
            "test_missense": by_group.get("missense"), "test_missense_n": by_group_n.get("missense"),
            "test_synonymous": by_group.get("synonymous"), "test_synonymous_n": by_group_n.get("synonymous"),
            "test_indel": by_group.get("indel"), "test_indel_n": by_group_n.get("indel"),
            "run_dir": r["run_dir"],
        })
    return pd.DataFrame(rows).sort_values(["model", "window", "seed"]).reset_index(drop=True)


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize_score_field(values) -> dict:
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return dict(n_valid=0, mean=None, sd=None, median=None, ci95_lo=None, ci95_hi=None)
    lo, hi = bootstrap_ci(s.tolist())
    return dict(n_valid=int(len(s)), mean=float(s.mean()), sd=float(s.std()) if len(s) > 1 else 0.0,
                median=float(s.median()), ci95_lo=lo, ci95_hi=hi)


def summarize_epoch_field(values) -> dict:
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return dict(n_valid=0, mean=None, sd=None, median=None, iqr_lo=None, iqr_hi=None, min=None, max=None)
    return dict(n_valid=int(len(s)), mean=float(s.mean()), sd=float(s.std()) if len(s) > 1 else 0.0,
                median=float(s.median()), iqr_lo=float(s.quantile(0.25)), iqr_hi=float(s.quantile(0.75)),
                min=float(s.min()), max=float(s.max()))


EPOCH_FIELDS = ["best_epoch", "stopped_epoch", "median_seconds_per_epoch"]
SCORE_FIELDS = ["val_spearman_best", "val_subset_best", "test_spearman", "test_pearson", "test_rmse",
                 "test_subset", "test_missense", "test_synonymous", "test_indel"]


def build_summary_by_configuration(summary_by_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, w), grp in summary_by_run.groupby(["model", "window"]):
        row = {"model": m, "window": w, "n_seeds": len(grp),
               "total_train_seconds_sum": float(grp["total_train_seconds"].sum())}
        for f in EPOCH_FIELDS:
            row.update({f"{f}_{k}": v for k, v in summarize_epoch_field(grp[f]).items()})
        for f in SCORE_FIELDS:
            row.update({f"{f}_{k}": v for k, v in summarize_score_field(grp[f]).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "window"]).reset_index(drop=True)


def build_best_epoch_analysis(runs: dict[tuple, dict]) -> pd.DataFrame:
    rows = []
    for (m, w, s), r in runs.items():
        d = run_derived_stats(r)
        rows.append({
            "model": m, "window": w, "seed": s,
            "best_epoch": d["best_epoch"], "stopped_epoch": d["stopped_epoch"], "stop_reason": d["stop_reason"],
            "n_patience_resets": d["n_patience_resets"],
            "reset_epochs": ";".join(str(e) for e in d["reset_epochs"]),
            "reset_deltas": ";".join(f"{x['epoch']}:{x['delta']:.5f}" for x in d["reset_deltas"]),
            "min_reset_delta": d["min_reset_delta"],
            "last_improvement_epoch": d["last_improvement_epoch"],
            "plateau_mean_before_best": d["plateau_mean_before_best"],
            "best_minus_plateau": d["best_minus_plateau"],
        })
    return pd.DataFrame(rows).sort_values(["model", "window", "seed"]).reset_index(drop=True)


def build_fixed_budget_table(runs: dict[tuple, dict]) -> pd.DataFrame:
    """Validation-trajectory comparison at fixed epoch checkpoints. A cell is
    left NaN (never interpolated) when a run's early stop happened before
    that epoch -- see module docstring / section 8 of the task spec."""
    rows = []
    for (m, w, s), r in runs.items():
        by_epoch = {h["epoch"]: h for h in r["history"]}
        row = {"model": m, "window": w, "seed": s}
        for ep in FIXED_EPOCHS:
            h = by_epoch.get(ep)
            row[f"val_subset_at_ep{ep}"] = h["val_subset_spearman"] if h else None
            row[f"val_spearman_at_ep{ep}"] = h["val_spearman"] if h else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "window", "seed"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_learning_curve(model, window, seed, r, out_path: Path):
    df = pd.DataFrame(r["history"])
    best_epoch, stopped_epoch = r["best"]["epoch"], int(df["epoch"].iloc[-1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (axes[0, 0], [("train_loss", "#4C72B0"), ("val_loss", "#DD8452")], "train / val loss"),
        (axes[0, 1], [("val_spearman", "#55A868")], "val Spearman"),
        (axes[0, 2], [("val_subset_spearman", "#C44E52")], "val subset Spearman (missense+indel mean)"),
        (axes[1, 0], [("monitor_value", "#8172B2")], f"early-stop monitor ({df['monitor_metric'].iloc[0]})"),
        (axes[1, 1], [("lr", "#937860")], "learning rate"),
        (axes[1, 2], [("seconds", "#64B5CD")], "epoch wall-clock (s)"),
    ]
    for ax, series, title in panels:
        for field, color in series:
            ax.plot(df["epoch"], df[field], color=color, label=field)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.6)
        ax.axvline(stopped_epoch, color="red", linestyle=":", alpha=0.6)
        if len(series) > 1:
            ax.legend(fontsize=8)
    fig.suptitle(f"{model}  W{window}  seed{seed}   best_score={r['best']['score']:.4f} @ epoch {best_epoch}  "
                 f"(green --)   stopped @ {stopped_epoch} (red :)   stop_reason={r['stop_reason']}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_seed_aggregate(model, window, seed_runs: dict[int, dict], out_path: Path, n_csv_path: Path):
    fields = [("val_subset_spearman", "val subset Spearman"), ("val_spearman", "val Spearman"),
              ("monitor_value", "early-stop monitor")]
    fig, axes = plt.subplots(1, len(fields) + 1, figsize=(5.2 * (len(fields) + 1), 4.5))

    per_epoch_n = None
    for i, (field, label) in enumerate(fields):
        ax = axes[i]
        by_epoch = defaultdict(list)
        for seed, r in seed_runs.items():
            df = pd.DataFrame(r["history"])
            ax.plot(df["epoch"], df[field], color="gray", alpha=0.25, linewidth=1)
            for _, row in df.iterrows():
                by_epoch[int(row["epoch"])].append(row[field])
        epochs = sorted(by_epoch)
        means, los, his, ns = [], [], [], []
        for ep in epochs:
            vals = np.asarray(by_epoch[ep], dtype=float)
            means.append(vals.mean())
            ns.append(len(vals))
            if len(vals) > 1:
                se = vals.std(ddof=1) / np.sqrt(len(vals))
                los.append(vals.mean() - 1.96 * se)
                his.append(vals.mean() + 1.96 * se)
            else:
                los.append(vals.mean())
                his.append(vals.mean())
        ax.plot(epochs, means, color="#C44E52", linewidth=2, label="mean across seeds still running")
        ax.fill_between(epochs, los, his, color="#C44E52", alpha=0.2, label="95% CI (normal approx)")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
        if i == 0:
            per_epoch_n = pd.DataFrame({"epoch": epochs, "n_valid_seeds": ns})

    ax = axes[-1]
    ax.step(per_epoch_n["epoch"], per_epoch_n["n_valid_seeds"], where="post", color="#4C72B0")
    ax.set_title("n(epoch): seeds not yet stopped", fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylim(0, max(per_epoch_n["n_valid_seeds"]) + 1)

    best_epochs = [r["best"]["epoch"] for r in seed_runs.values()]
    stopped_epochs = [int(pd.DataFrame(r["history"])["epoch"].iloc[-1]) for r in seed_runs.values()]
    fig.suptitle(f"{model}  W{window}  -- {len(seed_runs)} seeds {sorted(seed_runs)}   "
                 f"best_epoch median={np.median(best_epochs):.0f}   stopped_epoch median={np.median(stopped_epochs):.0f}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    per_epoch_n.to_csv(n_csv_path, index=False)


def plot_grouped_bar(summary_cfg: pd.DataFrame, metric_prefix: str, title: str, out_path: Path):
    models = [m for m in MODEL_MODES if m in summary_cfg["model"].unique()]
    windows = sorted(summary_cfg["window"].unique())
    if not models or not windows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(windows))
    for i, m in enumerate(models):
        sub = summary_cfg[summary_cfg["model"] == m].set_index("window").reindex(windows)
        means = sub[f"{metric_prefix}_mean"].to_numpy(dtype=float)
        los = sub[f"{metric_prefix}_ci95_lo"].to_numpy(dtype=float)
        his = sub[f"{metric_prefix}_ci95_hi"].to_numpy(dtype=float)
        yerr = np.vstack([np.nan_to_num(means - los), np.nan_to_num(his - means)])
        ax.bar(x + (i - (len(models) - 1) / 2) * width, np.nan_to_num(means), width, yerr=yerr, capsize=3,
               label=m, color=MODEL_COLORS.get(m, None))
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{w}" for w in windows])
    ax.set_title(title + "\n(error bars: 95% bootstrap CI across seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_epoch_boxplot(summary_by_run: pd.DataFrame, field: str, title: str, out_path: Path):
    order = [(m, w) for m in MODEL_MODES for w in sorted(summary_by_run["window"].unique())
             if not summary_by_run[(summary_by_run.model == m) & (summary_by_run.window == w)].empty]
    if not order:
        return
    data = [summary_by_run[(summary_by_run.model == m) & (summary_by_run.window == w)][field].dropna().to_numpy()
            for m, w in order]
    labels = [f"{m.split('_')[0]}\nW{w}" for m, w in order]
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(order)), 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_scatter_with_corr(summary_by_run: pd.DataFrame, xfield: str, yfield: str, title: str, out_path: Path):
    d = summary_by_run.dropna(subset=[xfield, yfield])
    if len(d) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for m in MODEL_MODES:
        sub = d[d.model == m]
        ax.scatter(sub[xfield], sub[yfield], label=m, color=MODEL_COLORS.get(m), alpha=0.75)
    rho, p = stats.spearmanr(d[xfield], d[yfield])
    ax.set_title(f"{title}\nSpearman rho={rho:.3f} (p={p:.3f}, n={len(d)}) -- "
                 f"association across configs/seeds, NOT a causal claim", fontsize=10)
    ax.set_xlabel(xfield)
    ax.set_ylabel(yfield)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return rho, p, len(d)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def w10_deep_dive(summary_by_run: pd.DataFrame, best_epoch_analysis: pd.DataFrame, model: str) -> str:
    lines = [f"## W10 심층 분석 ({model})\n"]
    sub = summary_by_run[summary_by_run.model == model]
    if sub.empty:
        return f"## W10 심층 분석 ({model})\n\n데이터 없음 (해당 model의 run이 아직 없음).\n"

    by_w = {w: sub[sub.window == w] for w in sorted(sub.window.unique())}
    w10 = by_w.get(10, sub.iloc[0:0])

    lines.append("**1. seed42의 76 epoch가 다른 seed에서도 반복되는가?**\n")
    if not w10.empty:
        stopped = w10.set_index("seed")["stopped_epoch"].to_dict()
        lines.append(f"- W10 seed별 stopped_epoch: {stopped}\n")
        vals = list(stopped.values())
        lines.append(f"- median={np.median(vals):.1f}, sd={np.std(vals):.1f}, "
                      f"range=[{min(vals)}, {max(vals)}]\n")
    lines.append("\n**2. W10의 best_epoch가 W5/W20보다 일관되게 늦은가?**\n")
    for w in sorted(by_w):
        v = by_w[w]["best_epoch"]
        if len(v):
            lines.append(f"- W{w}: best_epoch median={v.median():.1f}, mean={v.mean():.1f}, "
                          f"values={sorted(v.tolist())}\n")
    lines.append("\n**3-4. 장기간 점진적 향상 vs plateau 중 fluctuation:**\n")
    ba = best_epoch_analysis[(best_epoch_analysis.model == model) & (best_epoch_analysis.window == 10)]
    for _, row in ba.iterrows():
        lines.append(f"- seed{row.seed}: n_patience_resets={row.n_patience_resets}, "
                      f"min_reset_delta={row.min_reset_delta}, best_minus_plateau={row.best_minus_plateau}\n")
    lines.append("  (min_reset_delta이 작을수록, 그리고 best_minus_plateau이 작을수록 '작은 fluctuation이 "
                  "patience를 반복 리셋'시켰을 가능성이 큼 -- 반대로 reset마다 delta가 꾸준히 크면 "
                  "점진적 향상 쪽에 더 가까움. min_delta=0으로 실행되었으므로 위 delta가 곧 실제 "
                  "개선 판정 기준이었다는 점에 유의.)\n")

    lines.append("\n**5. best epoch가 늦은 seed일수록 test 성능도 높은가? (같은 config 내 paired, n=5 -- 매우 작은 표본)**\n")
    if len(w10) >= 3:
        rho, p = stats.spearmanr(w10["best_epoch"], w10["test_subset"])
        lines.append(f"- W10 내부 5-seed Spearman(best_epoch, test_subset) = {rho:.3f} (p={p:.3f}, n={len(w10)}) "
                      f"-- n=5라 신뢰구간이 매우 넓음, 참고용.\n")

    lines.append("\n**6. epoch 수-성능 관계가 특정 seed 하나에 의해 주도되는가? (leave-one-out)**\n")
    if len(w10) >= 4:
        for excl in w10["seed"]:
            rest = w10[w10.seed != excl]
            rho, p = stats.spearmanr(rest["best_epoch"], rest["test_subset"])
            lines.append(f"- seed{excl} 제외 시 rho={rho:.3f} (p={p:.3f}, n={len(rest)})\n")

    lines.append("\n**7. W10의 우위가 전체 test 성능인지 missense/indel(subset)에 국한되는지:**\n")
    for w in sorted(by_w):
        d = by_w[w]
        if len(d):
            lines.append(f"- W{w}: test_spearman(전체) mean={d.test_spearman.mean():.4f}, "
                          f"test_missense mean={d.test_missense.mean():.4f}, "
                          f"test_indel mean={d.test_indel.mean():.4f}, "
                          f"test_subset mean={d.test_subset.mean():.4f}\n")
    lines.append("\n**결론 프레이밍**: 위 수치만으로 '더 오래 학습해서 좋아졌다'를 단정하지 않는다. "
                  "가능성: (a) 추가 epoch의 실제 기여, (b) W10 representation이 좋은 성능과 장기 개선을 "
                  "동시에 유발(공통원인), (c) validation noise가 early stopping을 지연시켰을 뿐(min_delta=0의 "
                  "영향), (d) seed42 단일 outlier, (e) validation 기준 model selection의 winner's curse. "
                  "5-seed로는 이 중 하나를 확정할 수 없고, 위 세부 수치(리셋 delta 크기, leave-one-out 안정성, "
                  "그룹별 분해)가 각 가설에 대한 상대적 증거를 제공할 뿐이다.\n")
    return "".join(lines)


def build_report(summary_by_run, summary_by_configuration, best_epoch_analysis, fixed_budget,
                  manifest: dict, scatter_stats: dict) -> str:
    lines = ["# Stage 1 Multi-Seed Sweep -- Analysis Report\n"]
    lines.append(f"Generated from {manifest['n_runs_found']} / {manifest['n_runs_planned']} planned runs "
                  f"under `{manifest['runs_root']}`.\n")
    if manifest["n_runs_found"] < manifest["n_runs_planned"]:
        lines.append(f"\n**주의: 전체 sweep이 아직 완료되지 않음.** 아래 통계는 현재 존재하는 "
                      f"{manifest['n_runs_found']}개 run만으로 계산됨 -- 특히 n<3인 configuration의 "
                      f"SD/CI는 사실상 해석 불가.\n")

    lines.append("\n## 1. Configuration별 요약 (파라미터/속도/정확도)\n")
    lines.append("전체 수치는 `summary_by_configuration.csv` 참고. 핵심 열:\n\n")
    cols = ["model", "window", "n_seeds", "best_epoch_median", "stopped_epoch_median",
            "median_seconds_per_epoch_median", "val_subset_best_mean", "val_subset_best_sd",
            "test_subset_mean", "test_subset_sd", "test_subset_ci95_lo", "test_subset_ci95_hi"]
    present_cols = [c for c in cols if c in summary_by_configuration.columns]
    lines.append(summary_by_configuration[present_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n")

    lines.append("\n## 2. Fixed-budget (고정 epoch) validation 비교\n")
    lines.append("각 셀은 해당 epoch까지 실제로 도달한 seed만의 평균 -- 조기 종료로 도달 못한 seed는 "
                  "보간/추정하지 않고 NaN으로 남김 (`fixed_budget_comparison.csv` 참고).\n\n")
    fb_summary_rows = []
    for (m, w), grp in fixed_budget.groupby(["model", "window"]):
        row = {"model": m, "window": w}
        for ep in FIXED_EPOCHS:
            col = f"val_subset_at_ep{ep}"
            vals = grp[col].dropna()
            row[f"ep{ep}_mean"] = round(vals.mean(), 4) if len(vals) else None
            row[f"ep{ep}_n"] = len(vals)
        fb_summary_rows.append(row)
    fb_summary = pd.DataFrame(fb_summary_rows).sort_values(["model", "window"])
    lines.append(fb_summary.to_markdown(index=False))
    lines.append("\n")

    lines.append("\n## 3. Best epoch / stopped epoch 분포 (9 configuration 전체)\n")
    lines.append("`best_epoch_analysis.csv` 및 `plots/comparison/best_epoch_boxplot.png`, "
                  "`stopped_epoch_boxplot.png` 참고.\n")

    lines.append("\n## 4. best/test score와 best_epoch의 연관성 (scatter)\n")
    for key, (rho, p, n) in scatter_stats.items():
        lines.append(f"- {key}: Spearman rho={rho:.3f} (p={p:.3f}, n={n}) -- 연관성일 뿐 인과관계 아님.\n")

    for model in MODEL_MODES:
        lines.append("\n" + w10_deep_dive(summary_by_run, best_epoch_analysis, model))

    lines.append("\n## 5. 통계적 주의사항\n")
    lines.append("- seed 5개는 여전히 작은 표본. 아래는 강한 유의성 주장이 아니라 방향성 참고용.\n")
    syn_n = summary_by_run["test_synonymous_n"].dropna()
    if len(syn_n):
        lines.append(f"- synonymous 그룹 test 표본 수: min={int(syn_n.min())}, median={syn_n.median():.0f} -- "
                      f"작을 경우 synonymous Spearman은 매우 불안정하니 강조하지 말 것.\n")
    # seed variance vs model/window variance
    if len(summary_by_run):
        seed_var = summary_by_run.groupby(["model", "window"])["test_subset"].std().mean()
        config_var = summary_by_run.groupby(["model", "window"])["test_subset"].mean().std()
        lines.append(f"- 같은 configuration 내 seed 간 test_subset SD 평균={seed_var:.4f} vs "
                      f"configuration 평균값들의 SD={config_var:.4f} -- "
                      f"{'seed 분산이 configuration(모델/window) 간 차이에 비해 작지 않을 수 있음 (해석 주의)' if seed_var >= config_var * 0.5 else 'configuration 간 차이가 seed 분산보다 큰 편'}.\n")
    lines.append("- val 성능과 test 성능은 항상 분리해서 봐야 함 (`val_subset_best*` vs `test_subset*` 컬럼 구분).\n")
    lines.append("- 동일 split/seed를 공유하는 configuration끼리는 seed로 매칭한 paired difference가 "
                  "가능 -- `summary_by_run.csv`를 (model,window)로 pivot해 직접 계산 권장 (예: 같은 seed의 "
                  "paired_delta W10 vs unified_reference_delta W10 test_subset 차이).\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs/stage1_v2")
    ap.add_argument("--out", default="analysis/stage1_v2")
    ap.add_argument("--models", nargs="+", default=list(MODEL_MODES))
    ap.add_argument("--windows", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    runs_root, out = Path(args.runs_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(runs_root, args.models, args.windows, args.seeds)
    n_planned = len(args.models) * len(args.windows) * len(args.seeds)
    print(f"[analyze] found {len(runs)} / {n_planned} planned runs under {runs_root}")
    if not runs:
        print("[analyze] nothing to analyze yet -- exiting without writing output files.")
        return

    summary_by_run = build_summary_by_run(runs)
    summary_by_configuration = build_summary_by_configuration(summary_by_run)
    best_epoch_analysis = build_best_epoch_analysis(runs)
    fixed_budget = build_fixed_budget_table(runs)

    summary_by_run.to_csv(out / "summary_by_run.csv", index=False)
    summary_by_configuration.to_csv(out / "summary_by_configuration.csv", index=False)
    best_epoch_analysis.to_csv(out / "best_epoch_analysis.csv", index=False)
    fixed_budget.to_csv(out / "fixed_budget_comparison.csv", index=False)
    print(f"[analyze] wrote summary_by_run.csv, summary_by_configuration.csv, "
          f"best_epoch_analysis.csv, fixed_budget_comparison.csv")

    # -- plots: per-run learning curves --
    for (m, w, s), r in runs.items():
        plot_learning_curve(m, w, s, r, out / "plots" / "runs" / f"{m}__W{w}__seed{s}.png")
    print(f"[analyze] wrote {len(runs)} per-run learning curve plots")

    # -- plots: seed aggregate per configuration --
    by_config = defaultdict(dict)
    for (m, w, s), r in runs.items():
        by_config[(m, w)][s] = r
    for (m, w), seed_runs in by_config.items():
        plot_seed_aggregate(m, w, seed_runs, out / "plots" / "aggregate" / f"{m}_W{w}.png",
                             out / "plots" / "aggregate" / f"{m}_W{w}_valid_n.csv")
    print(f"[analyze] wrote {len(by_config)} seed-aggregate plots")

    # -- plots: comparisons --
    cmp_dir = out / "plots" / "comparison"
    plot_grouped_bar(summary_by_configuration, "test_subset", "test subset Spearman by model x window",
                      cmp_dir / "test_subset_by_model_window.png")
    plot_grouped_bar(summary_by_configuration, "val_subset_best", "best-epoch val subset Spearman by model x window",
                      cmp_dir / "val_subset_by_model_window.png")
    plot_epoch_boxplot(summary_by_run, "best_epoch", "best_epoch distribution across 9 configurations",
                        cmp_dir / "best_epoch_boxplot.png")
    plot_epoch_boxplot(summary_by_run, "stopped_epoch", "stopped_epoch distribution across 9 configurations",
                        cmp_dir / "stopped_epoch_boxplot.png")
    scatter_stats = {}
    r1 = plot_scatter_with_corr(summary_by_run, "best_epoch", "val_subset_best",
                                 "best validation score vs best_epoch", cmp_dir / "val_score_vs_best_epoch.png")
    if r1:
        scatter_stats["val_subset_best vs best_epoch"] = r1
    r2 = plot_scatter_with_corr(summary_by_run, "best_epoch", "test_subset",
                                 "test score vs best_epoch", cmp_dir / "test_score_vs_best_epoch.png")
    if r2:
        scatter_stats["test_subset vs best_epoch"] = r2
    print(f"[analyze] wrote comparison plots to {cmp_dir}")

    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "runs_root": str(runs_root), "analysis_out": str(out),
        "n_runs_planned": n_planned, "n_runs_found": len(runs),
        "models": args.models, "windows": args.windows, "seeds": args.seeds,
        "missing_runs": [f"{m}__W{w}__seed{s}" for m in args.models for w in args.windows for s in args.seeds
                          if (m, w, s) not in runs],
    }
    sweep_manifest_path = runs_root / "sweep_manifest.json"
    if sweep_manifest_path.exists():
        manifest["sweep_manifest"] = json.loads(sweep_manifest_path.read_text())
    with open(out / "experiment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    report = build_report(summary_by_run, summary_by_configuration, best_epoch_analysis, fixed_budget,
                           manifest, scatter_stats)
    (out / "analysis_report.md").write_text(report)
    print(f"[analyze] wrote experiment_manifest.json, analysis_report.md")
    print(f"[analyze] DONE -- all outputs under {out}")


if __name__ == "__main__":
    main()
