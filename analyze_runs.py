"""
analyze_runs.py — per-subset breakdown of finished runs, without retraining.

train.py saves runs/<MODEL>/seed<N>/test_predictions.npy in test-split row
order, so every question below can be answered from files already on disk.

WHY THE OVERALL TEST METRIC HIDES THE ONE THING M1-vs-M2 IS MEASURING
---------------------------------------------------------------------
All 11 Block B columns are constant across the 5,528 SAV rows (verified
against rad51c_X.npy, not assumed). Block B therefore only carries
information for indels -- 53 of the 882 test rows. On the full test split
M1 and M2 receive identical inputs for 94% of rows, so their overall
Spearman is the same number twice, plus noise. The comparison has to be
made on the indel subset.

Likewise the LLR baseline is only defined for missense variants (llr is NaN
for synonymous and indels), so "did we beat 0.59?" is only a fair question
on the same missense rows the baseline was scored on.

HOW TO RUN
----------
    python analyze_runs.py                    # every model under runs/
    python analyze_runs.py --runs runs --models M1 M2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train import compute_metrics

STRUCT_FEATURES_PATH = "data/structure/results/rad51c_struct_features.csv"
MANIFEST_PATH = "data/split_manifest.csv"
BASELINE_LLR_PATH = "data/baseline_llr.csv"


def load_test_frame() -> pd.DataFrame:
    """Test-split rows in the same order train.py's predictions were written."""
    struct = pd.read_csv(STRUCT_FEATURES_PATH)[["var_id", "split", "var_type"]]
    manifest = pd.read_csv(MANIFEST_PATH)[["var_id", "slim_consequence", "z_score_D4_D14"]]
    if not struct["var_id"].equals(pd.read_csv(MANIFEST_PATH)["var_id"]):
        raise SystemExit("struct_features / split_manifest var_id order differs")

    df = struct.merge(manifest, on="var_id", sort=False)
    df = df[df["split"] == "test"].reset_index(drop=True)

    llr = pd.read_csv(BASELINE_LLR_PATH)[["var_id", "llr"]]
    return df.merge(llr, on="var_id", how="left", sort=False)


def subsets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    cons = df["slim_consequence"]
    return {
        "all": np.ones(len(df), dtype=bool),
        "missense": (cons == "missense").to_numpy(),
        "synonymous": (cons == "synonymous").to_numpy(),
        "indel": (df["var_type"] != "sav").to_numpy(),
    }


def summarise(values: list[float]) -> str:
    a = np.array(values, dtype=np.float64)
    return f"{a.mean():.4f} +/- {a.std():.4f}"


def check_comparable(model_dirs: list[Path]) -> None:
    """Refuse to present M1-M4 side by side if they were not trained alike.

    The four models are only an ablation while they differ in esm_mode/use_lora
    and use_block_b and nothing else. train.py records the settings that must
    match in summary.json; a run left over from an earlier recipe would
    otherwise be averaged in silently and quietly become a comparison of
    training setups rather than of features.
    """
    settings: dict[str, dict] = {}
    for d in model_dirs:
        f = d / "summary.json"
        if not f.exists():
            print(f"  {d.name}: no summary.json (run predates the settings record)")
            continue
        s = json.loads(f.read_text(encoding="utf-8")).get("settings")
        if s:
            settings[d.name] = s

    if len(settings) < 2:
        return

    keys = sorted({k for s in settings.values() for k in s})
    mismatched = {
        k: {m: s.get(k) for m, s in settings.items()}
        for k in keys
        if len({json.dumps(s.get(k), sort_keys=True) for s in settings.values()}) > 1
    }
    # effective_batch is the quantity that must match; batch_size alone may
    # legitimately differ (M3/M4 split it into accumulation steps for memory).
    if not mismatched:
        print("  training settings identical across models -- comparable\n")
        return

    print("\n  *** WARNING: these models were NOT trained under the same settings ***")
    for k, vals in mismatched.items():
        print(f"    {k}: " + "  ".join(f"{m}={v}" for m, v in vals.items()))
    print("    Differences below may reflect the training recipe, not the features.")
    print("    Re-run the odd ones out with matching settings before drawing "
          "conclusions.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--models", nargs="*", default=None,
                    help="default: every subdirectory of --runs that has seeds")
    args = ap.parse_args()

    df = load_test_frame()
    masks = subsets(df)
    y_true = df["z_score_D4_D14"].to_numpy(dtype=np.float64)

    print(f"test split: {len(df)} rows  "
          + "  ".join(f"{k}={int(m.sum())}" for k, m in masks.items() if k != "all"))

    # Baseline: LLR is only defined for missense, so score it where it exists.
    has_llr = df["llr"].notna().to_numpy()
    base = compute_metrics(y_true[has_llr], df["llr"].to_numpy(dtype=np.float64)[has_llr])
    print(f"baseline LLR (n={base['n']}, missense only): "
          f"rho={base['spearman']:.4f}  r={base['pearson']:.4f}\n")

    runs_root = Path(args.runs)
    model_dirs = (
        [runs_root / m for m in args.models] if args.models
        else sorted(p for p in runs_root.iterdir() if p.is_dir())
    )

    table: dict[str, dict[str, list[float]]] = {}
    for model_dir in model_dirs:
        seed_dirs = sorted(model_dir.glob("seed*"))
        preds = []
        for sd in seed_dirs:
            f = sd / "test_predictions.npy"
            if f.exists():
                preds.append(np.load(f).astype(np.float64))
        if not preds:
            print(f"{model_dir.name}: no test_predictions.npy -- skipped")
            continue
        for p in preds:
            if len(p) != len(df):
                raise SystemExit(
                    f"{model_dir.name}: predictions have {len(p)} rows, "
                    f"test split has {len(df)}"
                )

        table[model_dir.name] = {
            name: [compute_metrics(y_true[m], p[m])["spearman"] for p in preds]
            for name, m in masks.items()
        }
        print(f"{model_dir.name}: {len(preds)} seed(s) from {model_dir}")

    if not table:
        raise SystemExit("nothing to analyse")

    print("\ncomparability check:")
    check_comparable([d for d in model_dirs if d.name in table])

    print(f"\n{'=' * 78}\nSpearman rho by subset  (mean +/- std over seeds)\n{'=' * 78}")
    names = list(masks)
    labels = [f"{n} (n={int(masks[n].sum())})" for n in names]
    header = f"{'model':<8}" + "".join(f"{lab:>20}" for lab in labels)
    print(header)
    for model, by_subset in table.items():
        print(f"{model:<8}" + "".join(f"{summarise(by_subset[n]):>20}" for n in names))

    # The Block B pairs: M1-M2 and M3-M4 differ only in use_block_b.
    print(f"\n{'=' * 78}\nBlock B effect (paired difference, same seed)\n{'=' * 78}")
    for with_b, without_b in (("M1", "M2"), ("M3", "M4")):
        if with_b not in table or without_b not in table:
            continue
        for name in names:
            a = np.array(table[with_b][name])
            b = np.array(table[without_b][name])
            k = min(len(a), len(b))
            d = a[:k] - b[:k]
            print(f"  {with_b}-{without_b}  {name:<12} "
                  f"{d.mean():+.4f} +/- {d.std():.4f}   "
                  f"({', '.join(f'{v:+.4f}' for v in d)})")
        print()

    print("Note: the indel subset is small, so its spread across seeds is the number "
          "to look at\n      before reading anything into the mean.")


if __name__ == "__main__":
    main()
