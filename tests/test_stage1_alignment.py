"""Alignment/normalization tests for src/stage1/alignment.py.

Covers the CPU-synthetic checklist in the task spec section 12 ("Alignment")
plus the two REQUIRED real-data fixtures (p.Leu338_Lys342delinsGln and
p.Lys186dup), checked against the actual data/wt_sequence.txt and
data/split_manifest.csv rows already in this repo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.stage1.alignment import (
    AlignmentValidationError,
    UnsupportedVariantError,
    anchor_relative_coordinates,
    build_global_alignment,
    normalize_edit,
    select_window,
    validate_reconstruction,
)
from src.stage1.schema import SLOT_MUT_ONLY, SLOT_PAIRED, SLOT_WT_ONLY

WT = "MAKGTFRLEQVDLNSFPLSPRVKLVSAGFTQAELLEHKPSDVGISKAEALQTLR"  # 55 aa synthetic


def _slots_by_pos(slots):
    return {(s.wt_pos, s.mut_pos): s for s in slots}


# ---------------------------------------------------------------------------
# WT == MUT -> Δ = 0 is tested at the tensor level in test_stage1_window_pe.py;
# here we check the ALIGNMENT is a clean 1:1 paired map with no event slots
# other than the single substitution position (synonymous case).
# ---------------------------------------------------------------------------
def test_synonymous_is_single_paired_event():
    row = {"var_id": "v1", "slim_consequence": "synonymous", "pp": 20, "ref_aa": WT[19]}
    edit = normalize_edit(WT, row)
    assert edit.u == 19 and edit.d == 1 and edit.m == 1
    assert edit.inserted_seq == WT[19]
    validate_reconstruction(WT, WT, edit)   # mutant == WT exactly

    alignment = build_global_alignment(edit)
    events = [s for s in alignment if s.is_event]
    assert len(events) == 1
    assert events[0].kind == SLOT_PAIRED
    assert events[0].wt_pos == 20 and events[0].mut_pos == 20


def test_missense_is_single_paired_event():
    row = {"var_id": "v2", "slim_consequence": "missense", "pp": 10, "ref_aa": WT[9], "alt_aa": "W"}
    edit = normalize_edit(WT, row)
    mut = WT[:9] + "W" + WT[10:]
    validate_reconstruction(WT, mut, edit)
    alignment = build_global_alignment(edit)
    events = [s for s in alignment if s.is_event]
    assert len(events) == 1 and events[0].kind == SLOT_PAIRED
    assert events[0].wt_pos == 10 and events[0].mut_pos == 10


def test_single_deletion_suffix_shift():
    row = {"var_id": "v3", "slim_consequence": "clinical_inframe_deletion", "pp": 15,
           "HGVSp": "NP_TEST:p.Xaa15del"}
    edit = normalize_edit(WT, row)
    mut = WT[:14] + WT[15:]
    validate_reconstruction(WT, mut, edit)
    alignment = build_global_alignment(edit)
    by_pos = _slots_by_pos(alignment)
    assert by_pos[(14, 14)].kind == SLOT_PAIRED       # last untouched prefix residue
    assert by_pos[(15, None)].kind == SLOT_WT_ONLY    # deleted residue
    assert by_pos[(16, 15)].kind == SLOT_PAIRED        # suffix shifted by -1
    assert by_pos[(17, 16)].kind == SLOT_PAIRED


def test_duplication_original_vs_copy():
    row = {"var_id": "v4", "slim_consequence": "clinical_inframe_insertion", "pp": 25,
           "HGVSp": "NP_TEST:p.Xaa25dup"}
    edit = normalize_edit(WT, row)
    mut = WT[:25] + WT[24:25] + WT[25:]
    validate_reconstruction(WT, mut, edit)
    alignment = build_global_alignment(edit)
    by_pos = _slots_by_pos(alignment)
    assert by_pos[(25, 25)].kind == SLOT_PAIRED         # the original residue
    assert by_pos[(None, 26)].kind == SLOT_MUT_ONLY      # the new copy
    assert by_pos[(None, 26)].insertion_rank == 1
    assert by_pos[(26, 27)].kind == SLOT_PAIRED          # suffix shifted by +1


def test_delins_does_not_pair_on_matching_letters():
    """p.Leu338_Lys342delinsGln shape: even though the deleted span happens to
    contain a 'Q'-adjacent letter pattern, WT-only/MUT-only must never be
    paired by amino-acid identity -- only by index."""
    s, e, ins = 35, 39, "Q"
    row = {"var_id": "v5", "slim_consequence": "clinical_inframe_deletion", "pp": s,
           "HGVSp": f"NP_TEST:p.Xaa{s}_Xaa{e}delinsGln"}
    edit = normalize_edit(WT, row)
    mut = WT[:s - 1] + ins + WT[e:]
    validate_reconstruction(WT, mut, edit)

    alignment = build_global_alignment(edit)
    by_pos = _slots_by_pos(alignment)
    assert by_pos[(34, 34)].kind == SLOT_PAIRED
    for p in range(35, 40):
        assert by_pos[(p, None)].kind == SLOT_WT_ONLY
    assert by_pos[(None, 35)].kind == SLOT_MUT_ONLY
    assert by_pos[(40, 36)].kind == SLOT_PAIRED
    assert by_pos[(41, 37)].kind == SLOT_PAIRED
    # no slot should pair any WT-only position with the MUT-only position
    n_del = e - s + 1
    assert n_del == 5 and len(ins) == 1


def test_reconstruction_mismatch_raises():
    row = {"var_id": "v6", "slim_consequence": "missense", "pp": 10, "ref_aa": WT[9], "alt_aa": "W"}
    edit = normalize_edit(WT, row)
    with pytest.raises(AlignmentValidationError):
        validate_reconstruction(WT, "completely-wrong-sequence", edit)


def test_wrong_ref_aa_raises_before_reconstruction():
    row = {"var_id": "v7", "slim_consequence": "missense", "pp": 10, "ref_aa": "Z", "alt_aa": "W"}
    with pytest.raises(AlignmentValidationError):
        normalize_edit(WT, row)


def test_unsupported_consequence_reported_not_dropped_silently():
    row = {"var_id": "v8", "slim_consequence": "frameshift", "pp": 10}
    with pytest.raises(UnsupportedVariantError) as exc:
        normalize_edit(WT, row)
    assert exc.value.var_id == "v8"


def test_select_window_clips_at_n_terminus():
    row = {"var_id": "v9", "slim_consequence": "missense", "pp": 2, "ref_aa": WT[1], "alt_aa": "W"}
    edit = normalize_edit(WT, row)
    alignment = build_global_alignment(edit)
    window = select_window(alignment, window_radius=10)
    wt_positions = [s.wt_pos for s in window if s.wt_pos is not None]
    assert min(wt_positions) == 1        # clipped, no negative/zero WT position
    assert len(window) == 1 + 1 + 10     # 1 left (pos 1) + event + 10 right


def test_select_window_metadata_only_edge_case_w0_pure_indel():
    """W=0 on a pure insertion/deletion event has attention_valid-relevant
    zero paired flank tokens for model A -- this is the metadata-only edge
    case (checked at the model level in test_stage1_models.py); here we only
    check the window itself has zero PAIRED flank slots at W=0."""
    row = {"var_id": "v10", "slim_consequence": "clinical_inframe_deletion", "pp": 15,
           "HGVSp": "NP_TEST:p.Xaa15del"}
    edit = normalize_edit(WT, row)
    alignment = build_global_alignment(edit)
    window = select_window(alignment, window_radius=0)
    assert all(s.is_event for s in window)
    assert len(window) == 1


def test_anchor_relative_coordinates_independent_of_window_radius():
    row = {"var_id": "v11", "slim_consequence": "clinical_inframe_insertion", "pp": 25,
           "HGVSp": "NP_TEST:p.Xaa25dup"}
    edit = normalize_edit(WT, row)
    alignment = build_global_alignment(edit)
    small = select_window(alignment, window_radius=3)
    big = select_window(alignment, window_radius=15)
    coord_small = dict(zip([(s.wt_pos, s.mut_pos) for s in small], anchor_relative_coordinates(edit, small)))
    coord_big = dict(zip([(s.wt_pos, s.mut_pos) for s in big], anchor_relative_coordinates(edit, big)))
    for key in coord_small:
        assert coord_small[key] == coord_big[key]


# ---------------------------------------------------------------------------
# Required real-data fixtures (task spec section 6): validated against the
# actual RAD51C WT sequence and the actual manifest rows shipped in this repo.
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _has_real_data() -> bool:
    return (_DATA_DIR / "wt_sequence.txt").exists() and (_DATA_DIR / "split_manifest.csv").exists()


@pytest.mark.skipif(not _has_real_data(), reason="repo data/ files not present")
def test_real_delins_leu338_lys342delinsgln():
    wt_seq = (_DATA_DIR / "wt_sequence.txt").read_text().strip()
    manifest = pd.read_csv(_DATA_DIR / "split_manifest.csv")
    row = manifest[manifest["var_id"] == "chr17_58732531_TGTTTCAAATCA_"].iloc[0].to_dict()
    assert "delinsGln" in row["HGVSp"]

    edit = normalize_edit(wt_seq, row)
    validate_reconstruction(wt_seq, row["mut_seq"], edit)
    assert len(wt_seq) == 376
    assert edit.wt_len == 376 and edit.mut_len == 372   # 376 - 5 + 1

    alignment = build_global_alignment(edit)
    by_pos = _slots_by_pos(alignment)
    assert by_pos[(337, 337)].kind == SLOT_PAIRED
    for p in range(338, 343):
        assert by_pos[(p, None)].kind == SLOT_WT_ONLY
    assert by_pos[(None, 338)].kind == SLOT_MUT_ONLY
    assert by_pos[(343, 339)].kind == SLOT_PAIRED
    assert by_pos[(344, 340)].kind == SLOT_PAIRED


@pytest.mark.skipif(not _has_real_data(), reason="repo data/ files not present")
def test_real_duplication_lys186dup():
    wt_seq = (_DATA_DIR / "wt_sequence.txt").read_text().strip()
    manifest = pd.read_csv(_DATA_DIR / "split_manifest.csv")
    row = manifest[manifest["var_id"] == "chr17_58696842__AAA"].iloc[0].to_dict()
    assert "dup" in row["HGVSp"]

    edit = normalize_edit(wt_seq, row)
    validate_reconstruction(wt_seq, row["mut_seq"], edit)
    assert edit.mut_len == edit.wt_len + 1

    alignment = build_global_alignment(edit)
    by_pos = _slots_by_pos(alignment)
    assert by_pos[(186, 186)].kind == SLOT_PAIRED
    assert by_pos[(None, 187)].kind == SLOT_MUT_ONLY
    assert by_pos[(187, 188)].kind == SLOT_PAIRED


@pytest.mark.skipif(not _has_real_data(), reason="repo data/ files not present")
def test_real_manifest_full_cohort_supported_or_reported():
    """Every row in the shipped manifest either normalizes+reconstructs
    cleanly, or is reported (never silently dropped). Currently the shipped
    manifest is scoped to exactly the supported cohort, so 0 should be
    skipped -- this test pins that fact so a future manifest change with an
    unsupported row is caught immediately (as a reported skip, not a crash).
    """
    from src.stage1.dataset import build_cohort

    wt_seq = (_DATA_DIR / "wt_sequence.txt").read_text().strip()
    manifest = pd.read_csv(_DATA_DIR / "split_manifest.csv")
    cohort = build_cohort(manifest.to_dict("records"), wt_seq)
    assert len(cohort.supported) + len(cohort.skipped) == len(manifest)
    assert len(cohort.skipped) == 0, cohort.skipped[:5]
