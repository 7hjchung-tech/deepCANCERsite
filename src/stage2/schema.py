"""Shared Stage 2 constants.

Structure tokens: Block A only (Block B / indel-seam features are excluded).
Token j of the tokenizer output S[:, j] corresponds to STRUCT_TOKEN_NAMES[j];
the order is part of the contract and must not be reshuffled by a tokenizer.
"""

from __future__ import annotations

from src.stage1.schema import MODEL_UNIFIED_REFERENCE_DELTA

STRUCT_TOKEN_NAMES = (
    "plddt",
    "rsasa",
    "dist_walker_a",
    "dist_walker_b",
    "dist_atp_contact",
    "dist_ssdna_binding",
    "dist_bcdx2_interface",
    "dist_cx3_interface",
    "secondary_structure",
)
N_STRUCT_TOKENS = len(STRUCT_TOKEN_NAMES)
STRUCT_TOKEN_DIM = 32

# raw Block A columns in rad51c_struct_features.csv (A_ss_* is one-hot)
CONTINUOUS_COLUMNS = (
    "A_plddt",
    "A_rsasa",
    "A_dist_walker_a",
    "A_dist_walker_b",
    "A_dist_atp_contact",
    "A_dist_ssdna_binding",
    "A_dist_bcdx2_interface",
    "A_dist_cx3_interface",
)
SS_COLUMNS = ("A_ss_helix", "A_ss_sheet", "A_ss_loop")   # code 0, 1, 2
SS_CODES = {"helix": 0, "sheet": 1, "loop": 2}

VARIANT_TYPES = ("missense", "synonymous", "indel")
VARIANT_TYPE_ID = {name: i for i, name in enumerate(VARIANT_TYPES)}

QUERY_MODES = ("single_query", "nine_query")
TRAIN_MODES = ("frozen_stage1", "joint_l2sp")
STAGE1_MODE = MODEL_UNIFIED_REFERENCE_DELTA
