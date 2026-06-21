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

# Motif / interface features as explicit RESIDUE-POSITION lists (1-based).
# Sources below. All interface/binding residues are transcribed from
# Olvera-Leon et al. Cell 2024, Data S2 (mmc4.pdf) and were VERIFIED 1:1 against
# the WT sequence (the residue letter the paper gives matches our seq position).
MOTIF_POSITIONS = {
    # Walker A critical residues per Data S2 Fig 4A: G125, G130, K131, T132
    # ("no amino acid change is tolerated, except for G125"). (The full PROSITE
    # P-loop motif is 125-132 'GAPGVGKT'; we keep only the critical residues.)
    "walker_a": [125, 130, 131, 132],

    # Walker B critical residue per Data S2 Fig 4A: D242 (catalytic Asp).
    # Independently confirmed by alignment to human RAD51 (Q06609) -> D222.
    "walker_b": [242],

    # ATP binding surface = union of the two ATP interfaces in Data S2 Fig 4A/4B:
    #   RAD51B/C (17): G125 P127 G128 V129 G130 K131 T132 Q133 E161 S163 R168
    #                  D242 G243 T283 Q285 R322 I341
    #   RAD51C/D (8):  H307 A308 A309 T310 K328 S329 P330 K333
    "atp_contact": [125, 127, 128, 129, 130, 131, 132, 133, 161, 163, 168,
                    242, 243, 283, 285, 322, 341,
                    307, 308, 309, 310, 328, 329, 330, 333],

    # ssDNA binding surface (Data S2 Fig 4C, 14 residues): R249 R258 R260 L262
    # N263 G264 T287 T288 K289 A300 L301 G302 S304 R312.
    "ssdna_binding": [249, 258, 260, 262, 263, 264, 287, 288, 289,
                      300, 301, 302, 304, 312],

    # BCDX2 interface = RAD51C residues contacting RAD51B / RAD51D / XRCC2
    # (Data S2 Figs 2, 3A/3B, 4): NTD M10 R24 G31 F32 E37; linker->D A87 L88 L90
    # L91; ATPase pocket for B-linker F164 M165 V166 V169 L205 I208 Y210; ATPase
    # ->B P127 Q133 G162 S163 A221 Y224 R260 Q285 T287 K342; ATPase->D K119 E303
    # R260 H307.
    "bcdx2_interface": [10, 24, 31, 32, 37, 87, 88, 90, 91,
                        164, 165, 166, 169, 205, 208, 210,
                        127, 133, 162, 163, 221, 224, 260, 285, 287, 342,
                        119, 303, 307],

    # CX3 interface = RAD51C residues contacting XRCC3 (Data S2 Figs 2C, 3D, 4):
    # NTD P43 S44 K54; ATPase pocket for X3-linker F164 V166 V169 L205 I208 Y210;
    # ATPase Q133 G162 S163 A221 Y224 R249 R260 Q285 R322.
    "cx3_interface": [43, 44, 54,
                      164, 166, 169, 205, 208, 210,
                      133, 162, 163, 221, 224, 249, 260, 285, 322],
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
# 2. Turn a list of 1-based residue positions into a length-376 binary mask.
#    Example: positions_to_mask([131, 242]) -> 1.0 at indices 130 and 241.
# --------------------------------------------------------------------------
def positions_to_mask(positions, length=SEQ_LEN):
    m = np.zeros(length, dtype=np.float32)
    for p in positions:
        m[p - 1] = 1.0          # 1-based position -> 0-based index
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
    seq = load_sequence()  # also runs the sanity checks

    ra = pd.DataFrame({"position": np.arange(1, SEQ_LEN + 1)})

    dom = coarse_domain_onehot()
    ra["domain_Nterm"] = dom[:, 0]
    ra["domain_ATPase"] = dom[:, 1]
    ra["domain_Cterm"] = dom[:, 2]

    # Guardrail: every flagged position must be inside the protein (1..376).
    for col, positions in MOTIF_POSITIONS.items():
        assert all(1 <= p <= SEQ_LEN for p in positions), f"{col}: position out of range"
        ra[col] = positions_to_mask(positions)

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
