"""
analyze_scope.py  —  evidence for the variant-type scope decision.

WHY THIS EXISTS
---------------
We exclude nonsense (stop_gained), frameshift, and splice variants from the
sequence-difference model. The justification is empirical: those variants
truncate the protein / break splicing, so their functional score is almost
uniformly "depleted" (loss-of-function) — they carry little learnable signal
and would let the model shortcut on "is it truncating?" instead of biology.

This script measures, per consequence type, the z-score and the
functional_classification breakdown, so the claim is backed by numbers.

OUTPUT
------
    data/scope_justification.csv   (one row per consequence type)

HOW TO RUN
----------
    .venv/bin/python analyze_scope.py
"""

import os
import numpy as np
import pandas as pd

SGE_CSV = "RAD51C_SGE/sge_rad51c.csv"

# Display order: kept-in-scope types first, then the excluded ones.
ORDER = [
    "missense", "synonymous",
    "clinical_inframe_deletion", "clinical_inframe_insertion", "codon_deletion",
    "stop_gained", "frameshift", "splice_donor", "splice_acceptor",
    "start_lost", "stop_lost", "intron", "UTR",
]


def analyze(path=SGE_CSV):
    df = pd.read_csv(path, low_memory=False)

    rows = []
    for c in ORDER:
        s = df[df["slim_consequence"] == c]
        n = len(s)
        if n == 0:
            continue
        z = s["z_score_D4_D14"]
        fc = s["functional_classification"].value_counts()
        depleted = fc.get("fast depleted", 0) + fc.get("slow depleted", 0)
        rows.append({
            "consequence": c,
            "n": n,
            "mean_z": round(float(z.mean()), 2) if z.notna().any() else np.nan,
            "median_z": round(float(z.median()), 2) if z.notna().any() else np.nan,
            "pct_depleted": round(100 * depleted / n, 1),
            "pct_fast_depleted": round(100 * fc.get("fast depleted", 0) / n, 1),
            "pct_unchanged": round(100 * fc.get("unchanged", 0) / n, 1),
            "pct_enriched": round(100 * fc.get("enriched", 0) / n, 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = analyze()
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/scope_justification.csv", index=False)

    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\n[save] wrote data/scope_justification.csv")

    # The one-line takeaway the writeup leans on:
    trunc = out[out.consequence.isin(["stop_gained", "frameshift",
                                      "splice_donor", "splice_acceptor"])]
    mis = out[out.consequence == "missense"].iloc[0]
    print(f"\nTruncating variants are {trunc.pct_depleted.min():.0f}-"
          f"{trunc.pct_depleted.max():.0f}% depleted (mean z ~"
          f"{trunc.mean_z.mean():.0f}); missense is only {mis.pct_depleted}% "
          f"depleted (mean z {mis.mean_z}). -> truncating variants are trivially "
          f"predictable, so they're out of the sequence model's scope.")
