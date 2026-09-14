"""Shared constants and small enums for the Stage 1 package.

Nothing here does any computation -- it exists so alignment.py, window.py,
modules.py, dataset.py and the tests all agree on the same integers/strings
instead of each hardcoding them separately.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# slot_kind integer codes (see README_STAGE1.md "Mask semantics" table)
# ---------------------------------------------------------------------------
SLOT_PAIRED = 0
SLOT_WT_ONLY = 1
SLOT_MUT_ONLY = 2
SLOT_PAD = 3    # batch padding only -- never produced by alignment.py itself

SLOT_KIND_NAMES = {
    SLOT_PAIRED: "paired",
    SLOT_WT_ONLY: "wt_only",
    SLOT_MUT_ONLY: "mut_only",
    SLOT_PAD: "pad",
}

# ---------------------------------------------------------------------------
# model mode identifiers (deliberately distinct from the legacy M1-M4 ids)
# ---------------------------------------------------------------------------
MODEL_PAIRED_DELTA = "paired_delta"
MODEL_BRANCHED_PROJECTION = "branched_projection"
MODEL_UNIFIED_REFERENCE_DELTA = "unified_reference_delta"
MODEL_MODES = (MODEL_PAIRED_DELTA, MODEL_BRANCHED_PROJECTION, MODEL_UNIFIED_REFERENCE_DELTA)

# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------
TOKEN_DIM = 128            # final content/K/V dimension for every model mode
DEFAULT_BOTTLENECK_DIM = 32
FLAGS_DIM = 6              # [wt_present, mut_present, delta_valid, is_paired, is_wt_only, is_mut_only]
PE_DIM = 128               # 64 anchor-relative coordinate + 64 insertion rank
PE_HALF_DIM = PE_DIM // 2
DEFAULT_D_ESM = 1280       # ESM-2 650M hidden size

# ---------------------------------------------------------------------------
# edit-type vocabulary for the common edit metadata token (sequence/edit
# derived only -- see metadata.py). Order is fixed; never reorder.
# ---------------------------------------------------------------------------
EDIT_TYPES = ("missense", "synonymous", "deletion", "insertion", "delins")
RAW_META_DIM = len(EDIT_TYPES) + 5   # 5 one-hot... + [u, d, m, wt_len, mut_len]

# ---------------------------------------------------------------------------
# cache / checkpoint schema versions -- bump on any incompatible format change
# ---------------------------------------------------------------------------
CACHE_SCHEMA_VERSION = "stage1-raw-v1"
ALIGNMENT_VERSION = "stage1-align-v1"
CHECKPOINT_SCHEMA_VERSION = "stage1-ckpt-v1"

# consequence categories this Stage 1 build supports (mirrors dataset.py's
# buildable scope: missense + synonymous + codon_deletion + inframe del/ins).
SUPPORTED_CONSEQUENCES = frozenset({
    "missense",
    "synonymous",
    "codon_deletion",
    "clinical_inframe_deletion",
    "clinical_inframe_insertion",
})
