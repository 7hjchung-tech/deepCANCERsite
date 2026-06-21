"""
build_annotation.py  —  Part A: per-residue functional annotation table for RAD51C.

WHAT THIS DOES
--------------
RAD51C is 376 amino acids long. Instead of computing annotations per-variant,
we build ONE table with 376 rows (one per residue position) and 10 feature
columns, then every variant just looks up its row by Protein_position.

    residue_annot : shape [376, 10]

OUTPUT
------
    data/rad51c_residue_annotation.csv

The 10 columns (ORDER IS FIXED — never reorder, weights would silently break):
    1-3  domain_Nterm / domain_ATPase / domain_Cterm   (one-hot, mutually exclusive)
    4    walker_a        (Walker A / P-loop motif)
    5    walker_b        (Walker B motif — not in UniProt for RAD51C -> 0)
    6    atp_contact     (ATP binding site)
    7    ssdna_binding   (needs Olvera-Leon Data S2 -> 0 for now)
    8    bcdx2_interface (RAD51B/RAD51D/XRCC3 interaction region)
    9    cx3_interface   (XRCC3-specific contacts, needs Data S2 -> 0 for now)
    10   conservation    (ConSurf/EVE scalar — not in repo -> 0 for now)

Columns 4-9 are MULTI-HOT: one residue can be in the ATPase domain AND Walker A
AND an ATP contact simultaneously. Never collapse them with argmax.

HOW TO RUN
----------
    .venv/bin/python build_annotation.py
"""

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 0. Constants. SEQ_LEN is the protein length. All ranges below are 1-based
#    and INCLUSIVE (i.e. position 125 through 132 means residues 125,...,132),
#    matching how biologists number residues.
# --------------------------------------------------------------------------
SEQ_LEN = 376
UNIPROT_ID = "O43502"  # RAD51C in UniProt

# Coarse 3-domain boundaries. These were READ FROM THE DATA, not guessed:
# the repo's DOMAINS column (InterPro Gene3D signature 3.40.50.300, the
# "P-loop NTPase" fold) spans residues 85-353. We round to clean boundaries.
ATPASE_START, ATPASE_END = 85, 350
#   N-terminal : 1   .. 84
#   ATPase     : 85  .. 350
#   C-terminal : 351 .. 376

# Motif / interface ranges. Each value is a LIST of (start, end) tuples so a
# feature can occupy several disconnected stretches. Empty list -> column is
# all zeros (we don't have the data yet).
#
# PROVENANCE of every range (so a reviewer can check it):
MOTIF_RANGES = {
    # PROSITE P-loop pattern [AG]-x(4)-G-K-[ST] matched the sequence as
    # 'GAPGVGKT' at residues 125-132, with the catalytic Lysine (K131) inside.
    "walker_a":        [(125, 132)],

    # Walker B has no annotated feature in UniProt for RAD51C -> leave empty.
    "walker_b":        [],

    # UniProt O43502 "Binding site" feature, 125-132 = the ATP/P-loop contact.
    # (It overlaps Walker A on purpose — that's biologically correct.)
    "atp_contact":     [(125, 132)],

    # ssDNA binding needs Olvera-Leon Data S2 (not in this repo) -> empty.
    "ssdna_binding":   [],

    # UniProt O43502 "Region" 79-136 = "Interaction with RAD51B, RAD51D and
    # XRCC3" = the BCDX2-complex interface. Used as a proxy until Data S2.
    "bcdx2_interface": [(79, 136)],

    # XRCC3-specific (CX3) contacts need Data S2 -> empty.
    "cx3_interface":   [],
}

# The fixed column order. len must be 10.
ANNOT_COLS = [
    "domain_Nterm", "domain_ATPase", "domain_Cterm",
    "walker_a", "walker_b", "atp_contact",
    "ssdna_binding", "bcdx2_interface", "cx3_interface",
    "conservation",
]


# --------------------------------------------------------------------------
# 1. Get the protein sequence. We try UniProt over the network, and fall back
#    to a local cache file so the script still works offline.
# --------------------------------------------------------------------------
def load_sequence(cache_path="rad51c_seq.txt"):
    if os.path.exists(cache_path):
        seq = open(cache_path).read().strip()
        print(f"[seq] loaded from cache {cache_path}")
    else:
        import requests
        url = f"https://rest.uniprot.org/uniprotkb/{UNIPROT_ID}.fasta"
        fasta = requests.get(url, timeout=30).text
        # A FASTA file is: a ">header" line, then sequence lines. Drop line 0,
        # join the rest into one string.
        seq = "".join(fasta.strip().split("\n")[1:])
        open(cache_path, "w").write(seq)
        print(f"[seq] fetched from UniProt and cached to {cache_path}")

    # Sanity check: position 131 (index 130) must be Lysine 'K' (Walker A).
    assert len(seq) == SEQ_LEN, f"expected {SEQ_LEN} aa, got {len(seq)}"
    assert seq[130] == "K", f"seq[130] should be K (K131), got {seq[130]}"
    print(f"[seq] OK: length {len(seq)}, K131 confirmed")
    return seq


# --------------------------------------------------------------------------
# 2. Turn a list of 1-based inclusive ranges into a length-376 binary mask.
#    Example: ranges_to_mask([(125,132)]) -> array with 1.0 at indices 124..131.
# --------------------------------------------------------------------------
def ranges_to_mask(ranges, length=SEQ_LEN):
    m = np.zeros(length, dtype=np.float32)
    for start, end in ranges:
        # 1-based inclusive [start, end]  ->  0-based slice [start-1 : end]
        m[start - 1:end] = 1.0
    return m


# --------------------------------------------------------------------------
# 3. The 3-domain one-hot. For every position decide N-term / ATPase / C-term.
# --------------------------------------------------------------------------
def coarse_domain_onehot():
    dom = np.zeros((SEQ_LEN, 3), dtype=np.float32)  # columns: [Nterm, ATPase, Cterm]
    for p in range(1, SEQ_LEN + 1):                 # p is the 1-based position
        if p < ATPASE_START:
            dom[p - 1, 0] = 1.0                     # N-terminal
        elif p <= ATPASE_END:
            dom[p - 1, 1] = 1.0                     # ATPase core
        else:
            dom[p - 1, 2] = 1.0                     # C-terminal
    return dom


# --------------------------------------------------------------------------
# 4. Assemble the full [376 x 10] table as a DataFrame.
# --------------------------------------------------------------------------
def build_residue_annotation():
    load_sequence()  # runs the sanity checks; we don't need the string here

    ra = pd.DataFrame({"position": np.arange(1, SEQ_LEN + 1)})

    dom = coarse_domain_onehot()
    ra["domain_Nterm"] = dom[:, 0]
    ra["domain_ATPase"] = dom[:, 1]
    ra["domain_Cterm"] = dom[:, 2]

    for col, ranges in MOTIF_RANGES.items():
        ra[col] = ranges_to_mask(ranges)

    # conservation: stopgap 0.0 until we have ConSurf/EVE. Do NOT standardize
    # here — standardization happens later, fit on TRAIN positions only.
    ra["conservation"] = 0.0

    # Re-order columns exactly: 'position' first, then the 10 fixed features.
    ra = ra[["position"] + ANNOT_COLS]
    assert len(ANNOT_COLS) == 10
    return ra


# --------------------------------------------------------------------------
# 5. A lookup helper + a get_annotation(position) function. Other scripts
#    (the Dataset) import these.
# --------------------------------------------------------------------------
def make_lookup(residue_annot):
    """Return (lookup_df, get_annotation_fn). get_annotation maps a
    Protein_position (which may be NaN or out of range) to a 10-vector."""
    lookup = residue_annot.set_index("position")[ANNOT_COLS]

    def get_annotation(protein_position):
        # intronic / UTR variants have no protein position -> all zeros
        if pd.isna(protein_position):
            return np.zeros(len(ANNOT_COLS), dtype=np.float32)
        p = int(protein_position)
        if p not in lookup.index:          # e.g. 377, just past the C-terminus
            return np.zeros(len(ANNOT_COLS), dtype=np.float32)
        return lookup.loc[p].to_numpy(dtype=np.float32)

    return lookup, get_annotation


# --------------------------------------------------------------------------
# 6. Run as a script: build, save, and self-test.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ra = build_residue_annotation()

    os.makedirs("data", exist_ok=True)
    out = "data/rad51c_residue_annotation.csv"
    ra.to_csv(out, index=False)
    print(f"\n[save] wrote {out}  (shape {ra.shape})")

    print("\n[counts] residues flagged per column:")
    print(ra[ANNOT_COLS].sum().astype(int).to_string())

    _, get_annotation = make_lookup(ra)

    print("\n[example] R258H (position 258, ATPase core, no motif):")
    print(dict(zip(ANNOT_COLS, get_annotation(258))))

    print("\n[example] K131 (Walker A lysine):")
    print(dict(zip(ANNOT_COLS, get_annotation(131))))

    # The key correctness test: at position 131, domain_ATPase, walker_a and
    # atp_contact must ALL be 1 at the same time (multi-hot, not argmax).
    v131 = get_annotation(131)
    assert v131[1] == 1 and v131[3] == 1 and v131[5] == 1, "multi-hot broken!"
    print("\n[test] multi-hot at K131 OK  (domain_ATPase & walker_a & atp_contact all = 1)")

    # intronic guard: NaN and 377 both return all-zeros.
    assert get_annotation(float("nan")).sum() == 0
    assert get_annotation(377).sum() == 0
    print("[test] NaN / out-of-range -> all-zero vector OK")
    print("\nDone.")
