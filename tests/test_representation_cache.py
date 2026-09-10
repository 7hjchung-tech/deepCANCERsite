"""Model-free tests for the Task C window/alignment/gather/provenance contract.

Nothing here loads ESM-2. Representations are replaced by a synthetic tensor
whose value at residue p encodes p, so a mis-gathered slot is detectable by
inspection rather than by a norm comparison.
"""

import json

import pandas as pd
import pytest
import torch

from src.embeddings import representation_cache as rc
from src.embeddings.variant_map import build_variant_record

WT = open("data/wt_sequence.txt").read().strip()
L = len(WT)


def fake_full(seq: str, tag: float) -> torch.Tensor:
    """[3, len(seq), 1280] where residue p (1-based) holds tag*1000 + p + layer."""
    n = len(seq)
    base = torch.arange(1, n + 1, dtype=torch.float32).view(1, n, 1)
    layer = torch.arange(3, dtype=torch.float32).view(3, 1, 1)
    return (tag * 1000.0 + base + layer).expand(3, n, 1280).contiguous()


def rec_for(row: dict, manifest_df=None) -> dict:
    return build_variant_record(row, WT, manifest_df=manifest_df)


def mut_for(row: dict) -> str:
    from dataset import build_mutant_sequence

    if row.get("mut_seq"):
        return row["mut_seq"]
    seq, ok, _ = build_mutant_sequence(WT, row)
    assert ok, row["var_id"]
    return seq


def window(row: dict, mut_len: int, W: int = 10) -> dict:
    return rc.build_window_alignment(rec_for(row), L, mut_len, W=W)


MISSENSE = {
    "var_id": "m1", "split": "train", "slim_consequence": "missense",
    "pp": 100, "ref_aa": WT[99], "alt_aa": "A",
    "HGVSp": "ENSP00000336701.4:p.Xaa100Ala",
    "mut_seq": WT[:99] + "A" + WT[100:],
}
SYNONYMOUS = {
    "var_id": "s1", "split": "train", "slim_consequence": "synonymous",
    "pp": 100, "ref_aa": WT[99], "alt_aa": WT[99],
    "HGVSp": "ENSP00000336701.4:p.Xaa100=", "mut_seq": WT,
}
DELETION = {
    "var_id": "d1", "split": "train", "slim_consequence": "codon_deletion",
    "pp": 100, "ref_aa": WT[99], "alt_aa": None,
    "HGVSp": "ENSP00000336701.4:p.Ile100del",   # WT[99] == "I"
}
DELINS = {
    "var_id": "x1", "split": "train", "slim_consequence": "clinical_inframe_deletion",
    "pp": 338, "ref_aa": "L", "alt_aa": "Q",
    "HGVSp": "ENSP00000336701.4:p.Leu338_Lys342delinsGln",
}
DUP = {
    "var_id": "u1", "split": "train", "slim_consequence": "clinical_inframe_insertion",
    "pp": 186, "ref_aa": "K", "alt_aa": None,
    "HGVSp": "ENSP00000336701.4:p.Lys186dup",
}
DUP2 = {
    "var_id": "u2", "split": "test", "slim_consequence": "clinical_inframe_insertion",
    "pp": 351, "ref_aa": "V", "alt_aa": None,
    "HGVSp": "ENSP00000336701.4:p.Val351dup",
}
#: deletion of WT residue 1 / of the last WT residue -- the only places where a
#: window is allowed to be shorter than the full flank contract.
DEL_NTERM = dict(DELETION, var_id="dn", pp=1,
                 HGVSp=f"ENSP00000336701.4:p.Met1del")
DEL_CTERM = dict(DELETION, var_id="dc", pp=L,
                 HGVSp=f"ENSP00000336701.4:p.Leu{L}del")


# ==========================================================================
# window rule
# ==========================================================================
def test_missense_window_is_plus_minus_10_all_paired():
    w = window(MISSENSE, L)
    assert len(w["slot_kind"]) == 21
    assert set(w["slot_kind"]) == {"paired"}
    assert w["wt_pos"] == list(range(90, 111))
    assert w["mut_pos"] == w["wt_pos"]
    assert all(w["delta_valid"]) and all(w["token_valid"])


def test_synonymous_window_matches_missense_window_at_same_position():
    assert window(SYNONYMOUS, L) == window(MISSENSE, L)


def test_window_truncates_at_the_termini_without_wrapping():
    near_start = dict(MISSENSE, pp=3, ref_aa=WT[2], mut_seq=WT[:2] + "A" + WT[3:])
    w = window(near_start, L)
    assert w["wt_pos"] == list(range(1, 14))          # 1..13, nothing below 1
    near_end = dict(MISSENSE, pp=L - 2, ref_aa=WT[L - 3],
                    mut_seq=WT[: L - 3] + "A" + WT[L - 2:])
    w = window(near_end, L)
    assert w["wt_pos"][-1] == L


def test_indel_window_keeps_task_b_core_verbatim_and_contiguous():
    """The window is derived from the edit geometry, the core from Task B's slot
    builder. Two independent derivations that must agree exactly."""
    for row in (DELETION, DELINS, DUP, DUP2, MISSENSE, SYNONYMOUS):
        core = rec_for(row)
        w = window(row, len(mut_for(row)))
        at = rc.core_offset_in_window({k: core[k] for k in rc.ARRAY_KEYS}, w)
        assert at >= 0, f"{row['var_id']}: Task B core is not inside the window"
        n = len(core["slot_kind"])
        for k in ("wt_pos", "mut_pos", "slot_kind", "delta_valid", "token_valid"):
            assert list(w[k][at:at + n]) == list(core[k]), f"{row['var_id']}/{k}"


#: (row, wt_only slots, mut_only slots, total slots) -- the exact contract at
#: W=10, away from the termini. A substitution is +-W around one paired slot
#: (2W+1); an indel is EXACTLY W paired flank slots per side plus its edit
#: event, with the boundary pairs counted INSIDE those W -- no ">= W" slack.
EXACT_WINDOWS = [
    ("missense", MISSENSE, 0, 0, 21),
    ("synonymous", SYNONYMOUS, 0, 0, 21),
    ("one-residue deletion", DELETION, 1, 0, 21),
    ("duplication p.Lys186dup", DUP, 0, 1, 21),
    ("duplication p.Val351dup", DUP2, 0, 1, 21),
    ("delins p.Leu338_Lys342delinsGln", DELINS, 5, 1, 26),
]


@pytest.mark.parametrize(
    "label,row,n_wt_only,n_mut_only,n_total",
    EXACT_WINDOWS,
    ids=[c[0] for c in EXACT_WINDOWS],
)
def test_window_has_exactly_W_paired_flanks_and_the_edit_event(
    label, row, n_wt_only, n_mut_only, n_total
):
    W = 10
    w = window(row, len(mut_for(row)), W=W)
    kinds = w["slot_kind"]

    assert kinds.count("wt_only") == n_wt_only
    assert kinds.count("mut_only") == n_mut_only
    assert len(kinds) == n_total, f"{label}: {len(kinds)} slots, want {n_total}"

    if n_wt_only or n_mut_only:
        assert n_total == 2 * W + n_wt_only + n_mut_only
        left = next(i for i, k in enumerate(kinds) if k != "paired")
        right = next(i for i, k in enumerate(reversed(kinds)) if k != "paired")
        assert left == W, f"{label}: {left} paired slots left of the edit, want {W}"
        assert right == W, f"{label}: {right} paired slots right of the edit, want {W}"
        # the event itself is one contiguous run
        assert kinds[left:len(kinds) - right] == (
            ["wt_only"] * n_wt_only + ["mut_only"] * n_mut_only
        )
    else:
        assert n_total == 2 * W + 1
        assert set(kinds) == {"paired"}


def test_the_real_delins_window_is_26_slots_not_28():
    """10 paired + 5 wt_only + 1 mut_only + 10 paired. The old rule appended a
    full 10 OUTSIDE a core that already carried the two boundary pairs, giving
    28."""
    w = window(DELINS, len(mut_for(DELINS)))
    assert len(w["slot_kind"]) == 26
    assert w["wt_pos"][:10] == list(range(328, 338))          # left flank incl. 337
    assert w["wt_pos"][10:15] == [338, 339, 340, 341, 342]    # deleted span
    assert w["slot_kind"][15] == "mut_only" and w["mut_pos"][15] == 338
    assert w["wt_pos"][16:] == list(range(343, 353))          # right flank from 343


def test_a_one_residue_deletion_window_is_21_slots():
    w = window(DELETION, len(mut_for(DELETION)))
    assert len(w["slot_kind"]) == 21
    assert w["wt_pos"] == list(range(90, 111))
    assert w["slot_kind"].count("wt_only") == 1


@pytest.mark.parametrize("row", [DUP, DUP2])
def test_a_one_residue_duplication_window_is_21_slots(row):
    """10 paired up to the insertion boundary + 1 mut_only copy + 10 paired."""
    pp = row["pp"]
    w = window(row, len(mut_for(row)))
    assert len(w["slot_kind"]) == 21
    assert w["wt_pos"][:10] == list(range(pp - 9, pp + 1))    # ...up to the boundary
    assert w["slot_kind"][10] == "mut_only" and w["mut_pos"][10] == pp + 1
    assert w["wt_pos"][11:] == list(range(pp + 1, pp + 11))   # after the boundary
    assert w["mut_pos"][11:] == list(range(pp + 2, pp + 12))  # frame-shifted by +1


def test_only_a_terminus_may_shorten_the_flank():
    """Fewer than W flank slots is allowed at a terminus, and nowhere else."""
    W = 10
    for row, mut_len, want_wt in [
        (DEL_NTERM, L - 1, [1] + list(range(2, 12))),       # no left flank at all
        (DEL_CTERM, L - 1, list(range(L - 10, L + 1))),     # no right flank at all
    ]:
        w = window(row, mut_len, W=W)
        assert w["wt_pos"] == want_wt
        assert w["slot_kind"].count("wt_only") == 1
        assert len(w["slot_kind"]) == W + 1                 # one side truncated away
        # nothing wrapped around or invented
        assert all(1 <= p <= L for p in w["wt_pos"] if p)

    near_start = dict(MISSENSE, pp=3, ref_aa=WT[2], mut_seq=WT[:2] + "A" + WT[3:])
    assert window(near_start, L)["wt_pos"] == list(range(1, 14))
    near_end = dict(MISSENSE, pp=L - 2, ref_aa=WT[L - 3],
                    mut_seq=WT[: L - 3] + "A" + WT[L - 2:])
    assert window(near_end, L)["wt_pos"] == list(range(L - 12, L + 1))


# ==========================================================================
# alignment correctness
# ==========================================================================
def test_deletion_right_side_is_frame_shifted_not_same_index():
    w = window(DELETION, L - 1)
    paired = {wp: mp for wp, mp, k in zip(w["wt_pos"], w["mut_pos"], w["slot_kind"])
              if k == "paired"}
    assert all(mp == wp for wp, mp in paired.items() if wp < 100)
    assert all(mp == wp - 1 for wp, mp in paired.items() if wp > 100)
    assert [wp for wp, _, k in zip(w["wt_pos"], w["mut_pos"], w["slot_kind"])
            if k == "wt_only"] == [100]


def test_duplication_right_side_shifts_by_plus_one_and_copy_is_mut_only():
    from dataset import build_mutant_sequence

    mut, ok, _ = build_mutant_sequence(WT, DUP)
    assert ok and len(mut) == L + 1
    w = window(DUP, len(mut))
    trip = list(zip(w["wt_pos"], w["mut_pos"], w["slot_kind"]))
    paired = {wp: mp for wp, mp, k in trip if k == "paired"}
    assert [mp for _, mp, k in trip if k == "mut_only"] == [187]
    assert mut[186] == "K"                                  # the duplicated copy
    assert all(mp == wp for wp, mp in paired.items() if wp <= 186)
    assert all(mp == wp + 1 for wp, mp in paired.items() if wp > 186)
    # every paired slot must point at the same residue on both sides
    assert all(WT[wp - 1] == mut[mp - 1] for wp, mp in paired.items())


def test_delins_pairs_wt343_with_mut339():
    from dataset import build_mutant_sequence

    mut, ok, _ = build_mutant_sequence(WT, DELINS)
    assert ok
    w = window(DELINS, len(mut))
    trip = list(zip(w["wt_pos"], w["mut_pos"], w["slot_kind"]))
    paired = {wp: mp for wp, mp, k in trip if k == "paired"}
    assert paired[343] == 339
    assert paired[337] == 337
    assert [wp for wp, _, k in trip if k == "wt_only"] == [338, 339, 340, 341, 342]
    assert [mp for _, mp, k in trip if k == "mut_only"] == [338]
    assert mut[337] == "Q"
    assert all(WT[wp - 1] == mut[mp - 1] for wp, mp in paired.items())


def test_gaps_are_token_valid_and_never_delta_valid():
    w = window(DELINS, L - 4)
    for k, dv, tv in zip(w["slot_kind"], w["delta_valid"], w["token_valid"]):
        if k in ("wt_only", "mut_only"):
            assert tv is True or tv == 1     # a gap is NOT padding
            assert not dv
        if k == "paired":
            assert dv and tv


def test_validate_accepts_every_real_window():
    for row, mut_len in [(MISSENSE, L), (SYNONYMOUS, L), (DELETION, L - 1),
                         (DELINS, L - 4), (DUP, L + 1)]:
        assert rc._validate_arrays(window(row, mut_len), L, mut_len) == ""


def test_validate_rejects_a_delta_marked_valid_over_a_gap():
    w = window(DELINS, L - 4)
    i = w["slot_kind"].index("wt_only")

    # marking the gap delta-valid while only the WT side exists
    bad = dict(w, delta_valid=list(w["delta_valid"]))
    bad["delta_valid"][i] = True
    assert rc._validate_arrays(bad, L, L - 4) == f"delta_valid_without_both_sides@{i}"

    # ... and even with a MUT side faked in, a wt_only slot may not be subtracted
    bad = dict(bad, mut_present=list(w["mut_present"]), mut_pos=list(w["mut_pos"]))
    bad["mut_present"][i], bad["mut_pos"][i] = 1, 200
    assert rc._validate_arrays(bad, L, L - 4) == f"delta_valid_on_wt_only@{i}"


def test_validate_rejects_out_of_range_coordinates():
    w = window(MISSENSE, L)
    bad = dict(w, wt_pos=list(w["wt_pos"]))
    bad["wt_pos"][0] = L + 5
    assert rc._validate_arrays(bad, L, L).startswith("wt_pos_out_of_range")


# ==========================================================================
# padding
# ==========================================================================
def test_pad_marks_only_padding_as_token_invalid():
    w = window(MISSENSE, L)
    p = rc.pad_arrays(w, 30)
    assert p["slot_kind"][21:] == ["pad"] * 9
    assert p["token_valid"][:21] == [True] * 21
    assert p["token_valid"][21:] == [False] * 9
    assert p["wt_present"][21:] == [False] * 9
    assert p["delta_valid"][21:] == [False] * 9


def test_pad_refuses_to_shrink_a_window():
    with pytest.raises(ValueError):
        rc.pad_arrays(window(MISSENSE, L), 5)


# ==========================================================================
# gather + delta
# ==========================================================================
def test_gather_reads_the_positions_the_arrays_name():
    full = fake_full(WT, 0.0)
    out = rc.gather_window(full, [5, 9, 0], [True, True, False])
    assert out.shape == (3, 3, 1280)
    assert out[0, 0, 0].item() == 5.0        # layer 0, residue 5
    assert out[2, 1, 0].item() == 9.0 + 2    # layer 2, residue 9
    assert out[:, 2, :].abs().max().item() == 0.0   # absent slot -> exact zeros


def test_delta_is_shifted_subtraction_not_same_index():
    """The whole point: for a deletion, delta at WT101 must use MUT100."""
    from dataset import build_mutant_sequence

    mut, _, _ = build_mutant_sequence(WT, DELETION)
    w = rc.pad_arrays(window(DELETION, len(mut)), 24)
    wt_full, mut_full = fake_full(WT, 0.0), fake_full(mut, 1.0)
    h_wt = rc.gather_window(wt_full, w["wt_pos"], w["wt_present"])
    h_mut = rc.gather_window(mut_full, w["mut_pos"], w["mut_present"])
    dv = torch.tensor(w["delta_valid"])
    delta = rc.compute_delta(h_wt, h_mut, dv)

    i = w["wt_pos"].index(101)
    assert w["mut_pos"][i] == 100
    # MUT value 1000+100+layer minus WT value 101+layer  ->  999 on every layer
    assert torch.allclose(delta[:, i, :], torch.full((3, 1280), 999.0))
    # a same-index subtraction would have given 1000+101 - 101 = 1000
    assert not torch.allclose(delta[:, i, :], torch.full((3, 1280), 1000.0))


def test_delta_is_exactly_zero_on_gaps_and_padding():
    mut_len = L - 4
    w = rc.pad_arrays(window(DELINS, mut_len), 40)
    h_wt = rc.gather_window(fake_full(WT, 0.0), w["wt_pos"], w["wt_present"])
    h_mut = rc.gather_window(fake_full("A" * mut_len, 1.0), w["mut_pos"], w["mut_present"])
    delta = rc.compute_delta(h_wt, h_mut, torch.tensor(w["delta_valid"]))
    for i, k in enumerate(w["slot_kind"]):
        if k in ("wt_only", "mut_only", "pad"):
            assert delta[:, i, :].abs().max().item() == 0.0
        else:
            assert delta[:, i, :].abs().max().item() > 0.0


def test_synonymous_delta_is_exactly_zero_when_both_sides_share_a_tensor():
    w = rc.pad_arrays(window(SYNONYMOUS, L), 21)
    full = fake_full(WT, 0.0)
    h_wt = rc.gather_window(full, w["wt_pos"], w["wt_present"])
    h_mut = rc.gather_window(full, w["mut_pos"], w["mut_present"])
    delta = rc.compute_delta(h_wt, h_mut, torch.tensor(w["delta_valid"]))
    assert delta.abs().max().item() == 0.0


def test_wt_only_keeps_wt_side_and_zeroes_mut_side():
    mut_len = L - 4
    w = rc.pad_arrays(window(DELINS, mut_len), 40)
    h_wt = rc.gather_window(fake_full(WT, 0.0), w["wt_pos"], w["wt_present"])
    h_mut = rc.gather_window(fake_full("A" * mut_len, 1.0), w["mut_pos"], w["mut_present"])
    for i, k in enumerate(w["slot_kind"]):
        if k == "wt_only":
            assert h_wt[:, i, :].abs().max().item() > 0.0
            assert h_mut[:, i, :].abs().max().item() == 0.0
        if k == "mut_only":
            assert h_mut[:, i, :].abs().max().item() > 0.0
            assert h_wt[:, i, :].abs().max().item() == 0.0


# ==========================================================================
# planning / dedup
# ==========================================================================
def _small_manifest() -> pd.DataFrame:
    rows = [MISSENSE, SYNONYMOUS, dict(SYNONYMOUS, var_id="s2"), DELETION]
    out = []
    for r in rows:
        r = dict(r)
        if not r.get("mut_seq"):
            from dataset import build_mutant_sequence

            r["mut_seq"] = build_mutant_sequence(WT, r)[0]
        out.append(r)
    return pd.DataFrame(out)


def test_plan_rows_deduplicates_identical_protein_edits():
    plans, skipped = rc.plan_rows(_small_manifest(), WT, in_scope_only=True)
    assert not skipped
    assert len(plans) == 4
    uniq = {p.mut_seq_hash for p in plans}
    assert len(uniq) == 3          # two synonymous rows share the WT protein
    syn = [p for p in plans if p.is_synonymous]
    assert len({(p.mut_seq_hash, p.window_sig) for p in syn}) == 1


def test_plan_rows_reports_out_of_scope_rows_instead_of_dropping_them_silently():
    df = pd.DataFrame([dict(MISSENSE, var_id="f1", slim_consequence="frameshift",
                            HGVSp="ENSP00000336701.4:p.Xaa100fs")])
    plans, skipped = rc.plan_rows(df, WT, in_scope_only=True)
    assert plans == []
    assert skipped and skipped[0]["reason"] == "unsupported_consequence"


# ==========================================================================
# provenance
# ==========================================================================
def test_provenance_hash_depends_on_every_identity_field():
    base = {k: f"v_{k}" for k in rc.PROVENANCE_IDENTITY_FIELDS}
    h0 = rc.provenance_hash(base)
    for field in rc.PROVENANCE_IDENTITY_FIELDS:
        other = dict(base, **{field: "CHANGED"})
        assert rc.provenance_hash(other) != h0, f"{field} not covered by the hash"


def test_provenance_hash_ignores_non_identity_bookkeeping():
    base = {k: f"v_{k}" for k in rc.PROVENANCE_IDENTITY_FIELDS}
    assert rc.provenance_hash(dict(base, n_rows=1)) == rc.provenance_hash(
        dict(base, n_rows=99999)
    )


def test_provenance_diff_names_the_fields_that_changed():
    base = {k: f"v_{k}" for k in rc.PROVENANCE_IDENTITY_FIELDS}
    other = dict(base, base_checkpoint_hash="x", window_W=42)
    d = rc.provenance_diff(base, other)
    assert set(d) == {"base_checkpoint_hash", "window_W"}


def test_load_cache_rejects_stale_provenance(tmp_path):
    prov = {k: f"v_{k}" for k in rc.PROVENANCE_IDENTITY_FIELDS}
    prov["provenance_hash"] = rc.provenance_hash(prov)
    payload = {
        "provenance": prov, "layers": [31, 32, 33], "A": 2,
        "slot_kind_vocab": list(rc.SLOT_KINDS),
        "wt_full": torch.zeros(3, 4, 1280),
        "mut_windows": torch.zeros(1, 3, 2, 1280),
        "row_to_mut_window": torch.zeros(1, dtype=torch.long),
        "wt_pos": torch.ones(1, 2, dtype=torch.long),
        "mut_pos": torch.ones(1, 2, dtype=torch.long),
        "wt_present": torch.ones(1, 2, dtype=torch.bool),
        "mut_present": torch.ones(1, 2, dtype=torch.bool),
        "delta_valid": torch.ones(1, 2, dtype=torch.bool),
        "token_valid": torch.ones(1, 2, dtype=torch.bool),
        "slot_kind": torch.zeros(1, 2, dtype=torch.long),
        "var_id": ["a"], "split": ["train"],
    }
    path = tmp_path / "c.pt"
    torch.save(payload, path)

    assert rc.load_cache(path, expected_provenance=prov) is not None
    for field in ("base_checkpoint_hash", "repr_layers", "cache_precision",
                  "window_rule_version", "alignment_version", "manifest_hash",
                  "split_schema_hash", "forward_precision"):
        with pytest.raises(rc.StaleCacheError) as e:
            rc.load_cache(path, expected_provenance=dict(prov, **{field: "STALE"}))
        assert field in str(e.value)


def test_load_cache_without_expectation_does_not_validate(tmp_path):
    """An explicit opt-out is allowed; the default path is the strict one."""
    prov = {"provenance_hash": "whatever"}
    payload = {"provenance": prov, "layers": [31, 32, 33], "A": 1,
               "slot_kind_vocab": list(rc.SLOT_KINDS), "var_id": [], "split": []}
    path = tmp_path / "c.pt"
    torch.save(payload, path)
    assert rc.load_cache(path).provenance == prov
