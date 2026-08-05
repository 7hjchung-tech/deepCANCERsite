"""
build_meta_x.py — build the meta feature matrix consumed by model.py.

OUTPUT
------
    data/structure/results/rad51c_meta_X.npy    (N, 3) float32

ENCODING (3-dim one-hot, mutually exclusive)
--------------------------------------------
    col 0  is_missense     slim_consequence == "missense"                 (4,555)
    col 1  is_synonymous   slim_consequence == "synonymous"                 (973)
    col 2  is_indel        codon_deletion | clinical_inframe_{deletion,insertion}
                                                                            (359)

WHY THESE THREE AND NOTHING ELSE
--------------------------------
* missense vs synonymous is NOT recoverable from any other stream.
  rad51c_meta.csv's own `var_type` column collapses both into "sav", so the
  distinction only survives in split_manifest.csv's `slim_consequence`.
  It matters: all 973 synonymous variants have mut_seq == wt_seq (verified),
  so their diff embedding is *exactly* the zero vector and their magnitude
  scalars are exactly zero -- the ESM stream carries no signal for them at
  all, and struct/meta are the only inputs left.

* del_len / ins_len are deliberately EXCLUDED: they are already columns
  B_del_len_norm and B_ins_len_norm of rad51c_X.npy (Block B). Duplicating
  them here would leak indel-length information into M2/M4, which slice
  struct_x[:, :11] precisely to drop Block B -- the M1-vs-M2 gap would then
  no longer measure Block B's contribution.

* anchor_pos is excluded for the same reason as above: keep meta orthogonal
  to the struct blocks so the ablation stays interpretable.

* The finer 5-way split of slim_consequence is collapsed because
  clinical_inframe_deletion (n=8) and clinical_inframe_insertion (n=2) are
  too rare to support their own columns across a 4,124-row training split.

ROW ORDER
---------
Aligned to rad51c_struct_features.csv's var_id order, which is what
smoke_test.py and the training loop use to index into rad51c_X.npy /
rad51c_y.npy / rad51c_meta_X.npy. The alignment is asserted at runtime
against split_manifest.csv rather than assumed.

HOW TO RUN
----------
    python build_meta_x.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

META_COLUMNS = ["is_missense", "is_synonymous", "is_indel"]

# slim_consequence -> column index in the one-hot
CONSEQUENCE_TO_COL = {
    "missense": 0,
    "synonymous": 1,
    "codon_deletion": 2,
    "clinical_inframe_deletion": 2,
    "clinical_inframe_insertion": 2,
}


def build_meta_x(struct_features_path: str, manifest_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (meta_X (N,3) float32, the joined dataframe used to build it)."""
    struct = pd.read_csv(struct_features_path)
    manifest = pd.read_csv(manifest_path)

    # Row order must match struct_features exactly -- every downstream consumer
    # indexes meta_X with a row number taken from struct_features["var_id"].
    if len(struct) != len(manifest) or not struct["var_id"].equals(manifest["var_id"]):
        raise SystemExit(
            f"var_id order differs between\n  {struct_features_path} ({len(struct)} rows)\n"
            f"  {manifest_path} ({len(manifest)} rows)\n"
            "meta_X must be built in struct_features' row order -- refusing to guess."
        )

    unknown = set(manifest["slim_consequence"]) - set(CONSEQUENCE_TO_COL)
    if unknown:
        raise SystemExit(
            f"slim_consequence values with no meta column: {sorted(unknown)}. "
            "Add them to CONSEQUENCE_TO_COL (and say which column they belong in)."
        )

    meta_x = np.zeros((len(manifest), len(META_COLUMNS)), dtype=np.float32)
    cols = manifest["slim_consequence"].map(CONSEQUENCE_TO_COL).to_numpy()
    meta_x[np.arange(len(manifest)), cols] = 1.0
    return meta_x, manifest


ANNOT_PATH = "data/rad51c_residue_annotation.csv"
# 'position' is the join key and 'conservation' is all zeros (never computed),
# so neither is a usable feature.
ANNOT_DROP = {"position", "conservation"}


def add_annotation(meta_x: np.ndarray, manifest: pd.DataFrame,
                   annot_path: str) -> tuple[np.ndarray, list[str]]:
    """Append per-residue functional annotation flags, joined on position.

    Block A already carries the DISTANCE from each residue to these sites; the
    binary "is this residue part of the site" is different information, and the
    file has been sitting unused. Constant columns are dropped rather than fed
    to the model as dead inputs.
    """
    annot = pd.read_csv(annot_path)
    cols = [c for c in annot.columns if c not in ANNOT_DROP]
    kept = [c for c in cols if annot[c].nunique() > 1]
    dropped = sorted(set(cols) - set(kept))
    if dropped:
        print(f"[build_meta_x] 상수라서 제외한 주석 컬럼: {dropped}")

    lookup = annot.set_index("position")[kept]
    joined = manifest["pp"].map(lambda p: lookup.loc[p] if p in lookup.index else None)
    if joined.isna().any():
        raise SystemExit("annotation 파일에 없는 위치가 있습니다")
    extra = np.stack(joined.to_numpy()).astype(np.float32)
    return np.concatenate([meta_x, extra], axis=1), kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-features",
                    default="data/structure/results/rad51c_struct_features.csv")
    ap.add_argument("--manifest", default="data/split_manifest.csv")
    ap.add_argument("--out", default="data/structure/results/rad51c_meta_X.npy")
    ap.add_argument("--with-annotation", action="store_true",
                    help=f"append the unused per-residue flags from {ANNOT_PATH}")
    ap.add_argument("--with-position", action="store_true",
                    help="append the residue position normalised to [0,1]")
    ap.add_argument("--annot-path", default=ANNOT_PATH)
    args = ap.parse_args()

    meta_x, manifest = build_meta_x(args.struct_features, args.manifest)
    names = list(META_COLUMNS)

    if args.with_annotation:
        meta_x, kept = add_annotation(meta_x, manifest, args.annot_path)
        names += kept
    if args.with_position:
        seq_len = float(manifest["pp"].max())
        pos = (manifest["pp"].to_numpy(dtype=np.float32) / seq_len).reshape(-1, 1)
        meta_x = np.concatenate([meta_x, pos], axis=1)
        names.append("position_norm")
    if len(names) > len(META_COLUMNS):
        print(f"[build_meta_x] meta 차원 {len(META_COLUMNS)} -> {len(names)}: {names}")

    # The first three columns must still be exactly one of the three classes.
    row_sums = meta_x[:, :len(META_COLUMNS)].sum(axis=1)
    assert (row_sums == 1.0).all(), f"{int((row_sums != 1.0).sum())} rows are not one-hot"

    print(f"[build_meta_x] {meta_x.shape[0]} variants -> {meta_x.shape[1]} meta dims")
    for i, name in enumerate(META_COLUMNS):
        n = int(meta_x[:, i].sum())
        print(f"    {name:<15} {n:>5}  ({100 * n / len(meta_x):.1f}%)")

    for split in ["train", "val", "test"]:
        mask = (manifest["split"] == split).to_numpy()
        counts = meta_x[mask][:, :len(META_COLUMNS)].sum(axis=0).astype(int)
        print(f"    {split:<5} n={int(mask.sum()):>5}  "
              + "  ".join(f"{n}={c}" for n, c in zip(META_COLUMNS, counts)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, meta_x)
    print(f"[build_meta_x] saved -> {args.out}")


if __name__ == "__main__":
    main()
