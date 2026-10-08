"""build_wt_neighbor_cache.py -- WT-residue Block A features + fixed 9-slot 3D neighbor index.

Reuses the existing extraction code (data/structure/code/block_b.py :: build_block_a) to get,
for ALL 376 WT residues (label-free -- no variant/z-score here), the same 9-field Block A
schema already used by structure_tokenizer/tokenizer.py, plus a residue-residue distance
matrix built from Cbeta-with-Calpha-fallback coordinates (see build_block_a's `rep`).

For every WT position p, N(p) = {p} U {8 nearest OTHER positions by that distance matrix}
(ties broken by WT sequence index). Slot 0 is always the anchor.

residue-number <-> WT-sequence-index check: build_block_a's `positions` are the PDB's own
res_id; this script asserts they are exactly 1..376 (no gaps, no insertion codes -- AlphaFold
models have none), single chain, and that every position's 3-letter PDB residue name matches
data/wt_sequence.txt at that same 1-based index. If any of this ever fails for a different
structure file, the script raises rather than silently proceeding.

In this specific structure every one of the 376 residues has a CA atom (build_block_a filters
to amino acids with a CA; Cbeta falls back to CA, never to "no coordinate"), so no anchor or
neighbor is ever excluded for a missing coordinate -- but the general per-residue valid/
excluded-with-reason bookkeeping is still produced, in case this is ever rerun against a
different, less complete structure.

Output: data/structure/results/wt_neighbor_cache.npz, with the raw per-residue feature table,
the neighbor index/distance/offset/is_anchor/valid tensors, and a metadata dict (pdb/annotation
sha256, k, feature schema, build date) used to refuse a stale cache.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PDB = ROOT / "data/structure/inputs/AF-O43502-F1.pdb"
ANNOT = ROOT / "data/structure/inputs/rad51c_residue_annotation.csv"
BLOCK_B = ROOT / "data/structure/code/block_b.py"
WT_SEQ = ROOT / "data/wt_sequence.txt"
OUT = ROOT / "data/structure/results/wt_neighbor_cache.npz"
K_NEIGHBORS = 8
CACHE_SCHEMA_VERSION = "wt-neighbor-cache-v1"

AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# order must match src/stage2/schema.CONTINUOUS_COLUMNS / structure_tokenizer CONT_FIELDS
CONT_FROM_FEAT_COLS = ["plddt", "rsasa", "dist_walker_a", "dist_walker_b", "dist_atp_contact",
                       "dist_ssdna_binding", "dist_bcdx2_interface", "dist_cx3_interface"]
SS_CLASSES = ["helix", "sheet", "loop"]   # == structure_tokenizer.tokenizer.SS_CLASSES order


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_block_b():
    spec = importlib.util.spec_from_file_location("block_b", BLOCK_B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    bb = load_block_b()
    ba = bb.build_block_a(str(PDB), str(ANNOT))
    positions, feats, dmat, df = ba["positions"], ba["feats"], ba["dmat"], ba["df"]
    n = len(positions)

    # --- residue-number <-> WT-sequence-index verification (not assumed) ---
    wt = WT_SEQ.read_text().strip()
    if len(wt) != n:
        raise ValueError(f"WT sequence length {len(wt)} != {n} residues extracted from the PDB")
    if not np.array_equal(positions, np.arange(1, n + 1)):
        raise ValueError("PDB residue numbers are not a contiguous 1..N range "
                         "(gap, insertion code, or non-standard numbering) -- mapping needs "
                         "an explicit per-residue table, not the direct pp==res_id assumption "
                         "this script makes for AF-O43502-F1.pdb")
    mismatches = [(int(p), wt[p - 1], AA3TO1.get(df["aa"].iloc[i], "?"))
                 for i, p in enumerate(positions) if wt[p - 1] != AA3TO1.get(df["aa"].iloc[i], "?")]
    if mismatches:
        raise ValueError(f"WT sequence vs PDB residue identity mismatch at {len(mismatches)} "
                         f"positions (first 5): {mismatches[:5]}")

    # --- 9-field raw schema (8 continuous + ss code), label-free, all 376 residues ---
    ss_oh = df[["ss_helix", "ss_sheet", "ss_loop"]].to_numpy()
    if not np.all(ss_oh.sum(1) == 1):
        raise ValueError("secondary-structure one-hot is not exactly one-hot for some residue")
    ss_code = ss_oh.argmax(1).astype(np.int64)
    continuous = df[CONT_FROM_FEAT_COLS].to_numpy(dtype=np.float32)
    n_nan = int(np.isnan(continuous).sum())
    if n_nan:
        raise ValueError(f"{n_nan} NaN values in the label-free WT residue feature table "
                         f"(a functional site likely has no annotated residue) -- cannot build "
                         f"a usable cache; see block_b.py's '[warn] no residues flagged' output")

    # --- fixed 9-slot neighbor index per anchor position ---
    neighbor_idx = np.zeros((n, K_NEIGHBORS + 1), dtype=np.int64)       # row index into `positions`
    distance = np.zeros((n, K_NEIGHBORS + 1), dtype=np.float32)
    offset = np.zeros((n, K_NEIGHBORS + 1), dtype=np.int64)             # signed i - p
    is_anchor = np.zeros((n, K_NEIGHBORS + 1), dtype=bool)
    valid = np.zeros((n, K_NEIGHBORS + 1), dtype=bool)
    excluded_report = []
    for row, p in enumerate(positions):
        neighbor_idx[row, 0], distance[row, 0], offset[row, 0], is_anchor[row, 0], valid[row, 0] = row, 0.0, 0, True, True
        d = dmat[row].copy()
        d[row] = np.inf                                                # exclude self from candidates
        order = np.lexsort((positions, d))                             # sort by distance, tie-break by position
        others = [j for j in order if np.isfinite(d[j])][:K_NEIGHBORS]
        if len(others) < K_NEIGHBORS:
            excluded_report.append({"position": int(p), "reason": f"only {len(others)} candidates with a finite distance"})
        for slot, j in enumerate(others, start=1):
            neighbor_idx[row, slot] = j
            distance[row, slot] = d[j]
            offset[row, slot] = int(positions[j]) - int(p)
            is_anchor[row, slot] = False
            valid[row, slot] = True
        # unfilled slots (only if len(others) < K_NEIGHBORS) stay at neighbor_idx=0 but INVALID
        for slot in range(len(others) + 1, K_NEIGHBORS + 1):
            valid[row, slot] = False

    meta = {
        "schema_version": CACHE_SCHEMA_VERSION, "pdb_sha256": sha256(PDB), "annot_sha256": sha256(ANNOT),
        "pdb_path": str(PDB.relative_to(ROOT)), "annot_path": str(ANNOT.relative_to(ROOT)),
        "k_neighbors": K_NEIGHBORS, "n_residues": int(n), "continuous_schema": CONT_FROM_FEAT_COLS,
        "ss_classes": SS_CLASSES, "distance_unit": "angstrom",
        "representative_coord_rule": "Cbeta, fallback to Calpha if Cbeta absent (block_b.py build_block_a's `rep`)",
        "tie_break": "WT sequence index ascending", "n_anchors_with_full_8_neighbors": int(n - len(excluded_report)),
        "excluded_report": excluded_report,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, positions=positions, continuous=continuous, ss=ss_code,
            neighbor_idx=neighbor_idx, distance=distance, offset=offset, is_anchor=is_anchor,
            valid=valid, meta_json=json.dumps(meta))
    print(f"[wt-neighbor-cache] wrote {OUT} ({n} residues, k={K_NEIGHBORS})")
    print(f"[wt-neighbor-cache] {len(excluded_report)} anchors short of {K_NEIGHBORS} neighbors "
          f"(expected 0 for this structure): {excluded_report}")


if __name__ == "__main__":
    main()
