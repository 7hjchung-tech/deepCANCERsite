"""
dataset.py  —  P1: load SGE data, build mutant sequences, position-based split,
                wrap as a PyTorch Dataset.

PIPELINE (what happens, in order)
---------------------------------
    sge_rad51c.csv (9,188 rows)
        -> filter to the variants in scope (A-1 = missense)
        -> for each variant, build the MUTANT protein sequence
        -> assign each variant to train / val / test by POSITION (no leakage)
        -> for each variant, assemble its feature vectors + labels
        -> serve through RAD51CDataset (a torch Dataset)

WHAT THIS FILE DOES *NOT* DO
----------------------------
The big ESM-2 embeddings (diff_emb, PLLR, structure) are computed ONCE by P2
(정호준 / 조승원) and saved to .pt files. This Dataset returns a `var_id` for
every variant so those embeddings can be joined in later. Until then it returns
the parts P1 owns: functional annotation (10-d), metadata (11-d), the modality
mask, and the labels. That is enough to (a) sanity-check the split and (b) build
P4 baselines once LLR is available.

HOW TO RUN
----------
    .venv/bin/python dataset.py
"""

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from build_annotation import (
    SEQ_LEN, load_sequence, build_residue_annotation, make_lookup, ANNOT_COLS,
)

SGE_CSV = "RAD51C_SGE/sge_rad51c.csv"

# The 8 variant types we one-hot encode (metadata dims 1-8). Anything not in
# this list (e.g. UTR) gets an all-zero one-hot — harmless, and out of scope.
VARIANT_TYPES = [
    "missense", "synonymous", "stop_gained", "intron",
    "splice_donor", "splice_acceptor",
    "clinical_inframe_deletion", "clinical_inframe_insertion",
]

# functional_classification -> integer label (for the classification head).
# We keep the raw 4 classes; collapsing (e.g. folding 'enriched') is a
# training-time choice, not baked into the data.
FUNC_CLASSES = ["unchanged", "slow depleted", "fast depleted", "enriched"]
FUNC_CLASS_TO_IDX = {c: i for i, c in enumerate(FUNC_CLASSES)}


# ==========================================================================
# 1. Load + light cleaning
# ==========================================================================
def load_sge(path=SGE_CSV):
    df = pd.read_csv(path, low_memory=False)
    # 'pp' = Protein_position as a number. Non-numeric / missing -> NaN.
    df["pp"] = pd.to_numeric(df["Protein_position"], errors="coerce")
    # A stable unique id per row, used later to join precomputed embeddings.
    df["var_id"] = df["chrom_pos_ref_alt"].astype(str)
    print(f"[load] {len(df)} rows from {path}")
    return df


# ==========================================================================
# 2. Scope filter. Two ways to express scope:
#    - include: keep ONLY these consequence types (allow-list), or
#    - exclude: keep everything EXCEPT these (block-list).
#    Use one or the other. exclude is handy when you think "drop intron,
#    frameshift, stop_gained, UTR and keep the rest".
# ==========================================================================
def filter_scope(df, include=None, exclude=None):
    if exclude is not None:
        sub = df[~df["slim_consequence"].isin(exclude)].copy()
        print(f"[scope] excluding {list(exclude)} -> kept {len(sub)} rows")
    else:
        include = include or ("missense",)
        sub = df[df["slim_consequence"].isin(include)].copy()
        print(f"[scope] including {list(include)} -> kept {len(sub)} rows")
    return sub


# 3-letter -> 1-letter amino-acid codes, used to read HGVSp protein changes.
AA3to1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*",
}


def _aa3_to_str(s):
    """'GlnArg' -> 'QR'. Returns None if any 3-letter chunk is unknown."""
    out = []
    for i in range(0, len(s), 3):
        aa = AA3to1.get(s[i:i + 3])
        if aa is None:
            return None
        out.append(aa)
    return "".join(out)


def parse_protein_change(hgvsp):
    """Parse the protein change in an HGVSp string into a structured edit.

    Handles inframe edits: deletions, duplications, insertions, delins.
    Returns a dict {op, start, end, inserted} (positions 1-based, inclusive),
    or None if it's something we don't build (e.g. 'p.Met1?').
    Examples:
        p.Ile64del                 -> del   64-64
        p.Phe103del                -> del  103-103
        p.Lys186dup                -> dup  186-186
        p.Leu338_Lys342delinsGln   -> delins 338-342 insert 'Q'
    """
    if pd.isna(hgvsp) or ":p." not in str(hgvsp):
        return None
    body = str(hgvsp).split(":p.")[1]

    # <Aaa><pos>[ _<Bbb><pos2> ] (del | dup | delins<Xxx..> | ins<Xxx..>)
    m = re.match(
        r"^([A-Za-z]{3})(\d+)(?:_([A-Za-z]{3})(\d+))?"
        r"(delins|del|dup|ins)([A-Za-z]*)$",
        body,
    )
    if not m:
        return None
    a1, p1, a2, p2, op, ins_aa3 = m.groups()
    start = int(p1)
    end = int(p2) if p2 else start
    inserted = ""
    if op in ("delins", "ins"):
        inserted = _aa3_to_str(ins_aa3)
        if inserted is None:
            return None
    return {"op": op, "start": start, "end": end, "inserted": inserted,
            "start_aa": AA3to1.get(a1), "end_aa": AA3to1.get(a2) if a2 else AA3to1.get(a1)}


# ==========================================================================
# 3. Build the mutant protein sequence for a variant.
#    Returns (mutant_sequence, ok_flag, start_position).
#    ok_flag=False means we couldn't build it cleanly -> the row is dropped.
#    start_position is the 1-based position used for the split / annotation /
#    metadata (None when not applicable).
# ==========================================================================
def build_mutant_sequence(wt_seq, row):
    cons = row["slim_consequence"]

    # ---------- missense: single-letter substitution ----------
    if cons == "missense":
        pos, ref_aa, alt_aa = row["pp"], row["ref_aa"], row["alt_aa"]
        # Multi-nucleotide "missense" rows have a range position and blank
        # ref_aa/alt_aa -> not a clean single substitution -> drop.
        if pd.isna(pos) or pd.isna(ref_aa) or pd.isna(alt_aa):
            return None, False, None
        if len(str(ref_aa)) != 1 or len(str(alt_aa)) != 1:
            return None, False, None
        pos = int(pos)
        if wt_seq[pos - 1] != str(ref_aa):
            raise ValueError(
                f"{row['var_id']}: WT seq has '{wt_seq[pos-1]}' at pos {pos}, "
                f"but ref_aa says '{ref_aa}'"
            )
        mut = wt_seq[:pos - 1] + str(alt_aa) + wt_seq[pos:]
        return mut, True, pos

    # ---------- synonymous: silent change -> mutant protein == WT ----------
    if cons == "synonymous":
        pos, ref_aa = row["pp"], row["ref_aa"]
        if pd.isna(pos) or pd.isna(ref_aa) or len(str(ref_aa)) != 1:
            return None, False, None
        pos = int(pos)
        if wt_seq[pos - 1] != str(ref_aa):
            raise ValueError(
                f"{row['var_id']}: WT '{wt_seq[pos-1]}' != ref_aa '{ref_aa}' at {pos}")
        # No amino-acid change: the mutant sequence is identical to WT. The
        # WT-difference embedding will be ~0 -> these act as silent controls.
        return wt_seq, True, pos

    # ---------- inframe indels (A-3) + codon_deletion (augmentation) ----------
    if cons in ("clinical_inframe_deletion", "clinical_inframe_insertion",
                "codon_deletion"):
        c = parse_protein_change(row["HGVSp"])
        if c is None:                       # e.g. 'p.Met1?'
            return None, False, None
        s, e = c["start"], c["end"]
        # Guardrail: the residue named in HGVSp must match the WT sequence.
        if c["start_aa"] is not None and wt_seq[s - 1] != c["start_aa"]:
            raise ValueError(f"{row['var_id']}: WT '{wt_seq[s-1]}' != HGVSp '{c['start_aa']}' at {s}")

        if c["op"] == "del":                # delete residues s..e
            mut = wt_seq[:s - 1] + wt_seq[e:]
        elif c["op"] == "dup":              # duplicate residues s..e (insert copy after e)
            mut = wt_seq[:e] + wt_seq[s - 1:e] + wt_seq[e:]
        elif c["op"] == "delins":           # delete s..e, insert new residues
            mut = wt_seq[:s - 1] + c["inserted"] + wt_seq[e:]
        elif c["op"] == "ins":              # insert between s and e (=s+1)
            mut = wt_seq[:s] + c["inserted"] + wt_seq[s:]
        else:
            return None, False, None
        return mut, True, s

    # ---------- not handled yet: nonsense (PLLR), splice, intron, etc. ----------
    return None, False, None


# ==========================================================================
# 4. Position-based split. The rule: every variant at the same protein
#    position goes to the SAME split. This prevents the model from "seeing"
#    a position in training and being tested on a neighbour of it (leakage).
# ==========================================================================
def split_by_position(positions, ratios=(0.70, 0.15, 0.15), seed=42):
    """positions: 1-d array of Protein_position (ints, no NaN).
    Returns a dict {position: 'train'|'val'|'test'}."""
    uniq = np.array(sorted(set(int(p) for p in positions)))
    rng = np.random.default_rng(seed)        # fixed seed -> reproducible split
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train_pos = set(uniq[:n_train])
    val_pos = set(uniq[n_train:n_train + n_val])
    test_pos = set(uniq[n_train + n_val:])

    mapping = {}
    for p in train_pos: mapping[p] = "train"
    for p in val_pos:   mapping[p] = "val"
    for p in test_pos:  mapping[p] = "test"
    print(f"[split] {n} unique positions -> "
          f"train {len(train_pos)} / val {len(val_pos)} / test {len(test_pos)}")
    return mapping


# ==========================================================================
# 5. Metadata vector (11-d), per the model spec §3.5.
#    [ 8 one-hot variant type | AA_pos_norm | CDS_pos_norm | affected_len_norm ]
# ==========================================================================
def _first_int(x):
    """CDS_position can be '772' or a range '772-773'. Grab the first integer."""
    if pd.isna(x):
        return np.nan
    return int(str(x).split("-")[0])


def build_metadata(row):
    meta = np.zeros(11, dtype=np.float32)

    # dims 0-7: variant type one-hot. codon_deletion is mechanically an
    # inframe deletion, so it shares that slot.
    cons = row["slim_consequence"]
    if cons == "codon_deletion":
        cons = "clinical_inframe_deletion"
    if cons in VARIANT_TYPES:
        meta[VARIANT_TYPES.index(cons)] = 1.0

    # dim 8: amino-acid position normalized to [0,1]. NaN (intronic) -> 0.
    pp = row["pp"]
    meta[8] = (pp / SEQ_LEN) if not pd.isna(pp) else 0.0

    # dim 9: CDS position normalized by the coding length (376 codons * 3).
    cds = _first_int(row.get("CDS_position"))
    meta[9] = (cds / (SEQ_LEN * 3)) if not pd.isna(cds) else 0.0

    # dim 10: affected length in bp, normalized (SNV=1bp). Clipped to [0,1].
    ref = str(row.get("ref", "") or "")
    new = str(row.get("new", "") or "")
    affected_bp = max(len(ref), len(new), 1)
    meta[10] = min(affected_bp / 50.0, 1.0)
    return meta


# ==========================================================================
# 6. The PyTorch Dataset.
# ==========================================================================
class RAD51CDataset(Dataset):
    def __init__(self, df_split, wt_seq, get_annotation):
        """df_split: the rows for ONE split (train/val/test), already filtered.
        wt_seq: the WT protein string.
        get_annotation: position -> 10-vector lookup."""
        self.df = df_split.reset_index(drop=True)
        self.wt_seq = wt_seq
        self.get_annotation = get_annotation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- features P1 owns ----
        annot = self.get_annotation(row["pp"])              # (10,)
        annot_valid = np.float32(not pd.isna(row["pp"]))    # 1 if coding
        meta = build_metadata(row)                          # (11,)

        # ---- labels ----
        # regression target: SGE z-score (continuous). The main A-1 target.
        z = row["z_score_D4_D14"]
        label_reg = np.float32(z if not pd.isna(z) else np.nan)
        # classification target: functional class index, or -1 if unknown.
        label_cls = FUNC_CLASS_TO_IDX.get(row["functional_classification"], -1)

        return {
            "var_id": row["var_id"],            # to join P2 embeddings later
            "annot": torch.from_numpy(annot),               # (10,)
            "annot_valid": torch.tensor(annot_valid),       # ()
            "meta": torch.from_numpy(meta),                 # (11,)
            "label_reg": torch.tensor(label_reg),           # ()
            "label_cls": torch.tensor(label_cls, dtype=torch.long),  # ()
            # placeholders P2 fills from precomputed .pt (kept None for now):
            # "diff_emb": ..., "pllr": ..., "struct": ...
        }


# ==========================================================================
# 7. One function that runs the whole P1 pipeline and returns 3 datasets.
# ==========================================================================
def save_manifest(sub, wt_seq, path="data/split_manifest.csv", wt_path="data/wt_sequence.txt"):
    """Write the train/val/test assignment + everything a teammate needs to
    attach ESM embeddings: var_id (join key), the split, the mutant sequence,
    and the labels. The shared WT sequence is written once to wt_path."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["var_id", "split", "slim_consequence", "pp", "ref_aa", "alt_aa",
            "HGVSp", "z_score_D4_D14", "functional_classification", "mut_seq"]
    sub[cols].to_csv(path, index=False)
    open(wt_path, "w").write(wt_seq)
    print(f"[save] wrote {path} ({len(sub)} rows) and {wt_path}")


def build_datasets(include=None, exclude=None, seed=42, save_dir=None):
    wt_seq = load_sequence()
    residue_annot = build_residue_annotation()
    _, get_annotation = make_lookup(residue_annot)

    df = load_sge()
    sub = filter_scope(df, include=include, exclude=exclude)

    # Build every mutant sequence. We get back (sequence, ok, start_pos).
    # Rows we can't build cleanly (MNV "missense", 'p.Met1?', etc.) are dropped
    # with a count instead of crashing later. For indels whose Protein_position
    # column is a range/blank, start_pos (parsed from HGVSp) is the clean
    # position we use for the split, annotation lookup, and metadata.
    built = sub.apply(lambda r: build_mutant_sequence(wt_seq, r), axis=1)
    sub["mut_seq"] = [b[0] for b in built]   # the mutant protein string (for P2/ESM)
    sub["_ok"] = [b[1] for b in built]
    sub["_start"] = [b[2] for b in built]
    n_drop = int((~sub["_ok"]).sum())
    # Per-consequence report: for each type in scope, how many were kept vs
    # dropped, and why dropping happens (no clean protein-sequence change).
    print("[seqs] kept / dropped by consequence type:")
    for cons, grp in sub.groupby("slim_consequence"):
        kept = int(grp["_ok"].sum()); dropped = len(grp) - kept
        note = "" if dropped == 0 else "  <- no protein-sequence change to build"
        print(f"    {cons:28s} kept {kept:4d} | dropped {dropped:4d}{note}")
    sub = sub[sub["_ok"]].copy()
    # Overwrite pp with the clean parsed start position (fixes range positions).
    sub["pp"] = sub["_start"].astype(int)
    sub = sub.drop(columns=["_ok", "_start"])
    print(f"[seqs] built {len(sub)}/{len(sub) + n_drop} mutant sequences OK")

    assert sub["pp"].notna().all()

    # Position-based split.
    mapping = split_by_position(sub["pp"].to_numpy(), seed=seed)
    sub["split"] = sub["pp"].astype(int).map(mapping)

    datasets = {}
    for name in ("train", "val", "test"):
        part = sub[sub["split"] == name]
        datasets[name] = RAD51CDataset(part, wt_seq, get_annotation)

    # ---- LEAKAGE CHECK: no position may appear in two splits ----
    pos_sets = {name: set(datasets[name].df["pp"].astype(int))
                for name in ("train", "val", "test")}
    assert pos_sets["train"].isdisjoint(pos_sets["val"])
    assert pos_sets["train"].isdisjoint(pos_sets["test"])
    assert pos_sets["val"].isdisjoint(pos_sets["test"])
    print("[leakage] OK: no protein position shared across splits")

    if save_dir is not None:
        save_manifest(sub, wt_seq, path=f"{save_dir}/split_manifest.csv",
                      wt_path=f"{save_dir}/wt_sequence.txt")

    return datasets


# ==========================================================================
# 8. Run as a script: build everything and print a report.
# ==========================================================================
if __name__ == "__main__":
    # Scope expressed as an exclude-list: drop these, keep everything else.
    # NOTE: splice_donor/acceptor and start_lost have no protein-sequence change
    # to feed this model, so they are kept "in scope" but dropped at build time
    # (you'll see them in the per-consequence report). The buildable result is
    # missense + synonymous + codon_deletion + inframe del/ins.
    ds = build_datasets(exclude=(
        "intron",
        "frameshift",
        "stop_gained",
        "UTR",
    ), save_dir="data")

    print("\n[sizes] variants per split:")
    for name in ("train", "val", "test"):
        print(f"  {name:5s}: {len(ds[name])} variants")

    print("\n[sample] first training example:")
    ex = ds["train"][0]
    for k, v in ex.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} shape={tuple(v.shape)}  values={v.numpy().round(3)}")
        else:
            print(f"  {k:12s} = {v}")

    # quick class balance per split (for P4 weighting decisions)
    print("\n[balance] functional_classification counts in train:")
    print(ds["train"].df["functional_classification"].value_counts().to_string())
