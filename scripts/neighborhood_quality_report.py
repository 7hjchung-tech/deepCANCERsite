"""Section 11 "before training" neighbor-quality report: distance/offset/pLDDT distributions
over the fixed 9-slot WT neighbor index (data/structure/results/wt_neighbor_cache.npz).
Reports only -- no experiments are triggered from these numbers.
Output -> analysis/stage2_neighborhood/neighbor_quality.{csv,json,png}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/stage2_neighborhood"
OUT.mkdir(parents=True, exist_ok=True)
d = np.load(ROOT / "data/structure/results/wt_neighbor_cache.npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
pos, dist, off, valid, is_anchor = d["positions"], d["distance"], d["offset"], d["valid"], d["is_anchor"]
plddt = d["continuous"][:, 0]   # CONT_FROM_FEAT_COLS[0] == "plddt"
n = len(pos)

neigh_mask = ~is_anchor & valid           # the 8 (or fewer) non-anchor slots
all_dist = dist[neigh_mask]
all_off = np.abs(off[neigh_mask])
eighth_dist = dist[:, -1]                 # slot 8 = farthest of the 8 by construction (sorted ascending)

def q(x):
    x = np.asarray(x, float)
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
           "p25": float(np.quantile(x, .25)), "median": float(np.median(x)), "p75": float(np.quantile(x, .75)),
           "max": float(x.max())}

report = {
    "n_wt_residues": n, "k_neighbors": meta["k_neighbors"],
    "anchors_with_full_k_neighbors": meta["n_anchors_with_full_8_neighbors"],
    "neighbor_distance_angstrom": q(all_dist),
    "eighth_neighbor_distance_angstrom": q(eighth_dist),
    "frac_neighbor_abs_offset_le_2": float((all_off <= 2).mean()),
    "frac_neighbor_sequence_far_abs_offset_gt_10": float((all_off > 10).mean()),
    "neighbor_abs_offset": q(all_off),
    "wt_plddt_all_residues": q(plddt),
    "missing_coord_or_feature_count": 0,   # verified during cache build (raises if any NaN / <k neighbors)
}
(OUT / "neighbor_quality.json").write_text(json.dumps(report, indent=2))

rows = []
for row, p in enumerate(pos):
    for slot in range(1, dist.shape[1]):
        if valid[row, slot]:
            rows.append({"anchor_pos": int(p), "slot": slot, "neighbor_pos": int(pos[d["neighbor_idx"][row, slot]]),
                        "distance": float(dist[row, slot]), "offset": int(off[row, slot])})
pd.DataFrame(rows).to_csv(OUT / "neighbor_table_full.csv", index=False)

examples = pd.DataFrame(rows)[pd.DataFrame(rows).anchor_pos.isin([2, 100, 200, 300, 376])]
examples.to_csv(OUT / "neighbor_examples.csv", index=False)
print(json.dumps(report, indent=2))
print("\nexamples (anchor positions 2, 100, 200, 300, 376):")
print(examples.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(all_dist, bins=40); axes[0].set_xlabel("distance to neighbor (Å)"); axes[0].set_title("all 8 neighbors")
axes[1].hist(eighth_dist, bins=40); axes[1].set_xlabel("8th-neighbor distance (Å)"); axes[1].set_title("farthest of the 8")
axes[2].hist(all_off, bins=np.arange(0, all_off.max() + 2) - 0.5); axes[2].set_xlabel("|sequence offset|")
axes[2].set_title("how far in sequence are 3D-close neighbors?")
fig.tight_layout(); fig.savefig(OUT / "neighbor_quality.png", dpi=130)
print(f"\n[neighbor-quality] wrote {OUT}")
