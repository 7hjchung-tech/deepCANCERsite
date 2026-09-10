import pandas as pd

from src.embeddings.variant_map import audit_manifest, build_variant_record


WT = "MRGKTFRFEMQRDLVSFPLSPAVRVKLVSAGFQTAEELLEVKPSELSKEVGISKAEALETLQIIRRECLTNKPRYAGTSESHKKCTALELLEQEHTQGFIITFCSALDDILGGGVPLMKTTEICGAPGVGKTQLCMQLAVDVQIPECFGGVAGEAVFIDTEGSFMVDRVVDLATACIQHLQLIAEKHKGEEHRKALEDFTLDNILSHIYYFRCRDYTELLAQVYLLPDFLSEHSKVRLVIVDGIAFPFRHDLDDLSLRTRLLNGLAQQMISLANNHRLAVILTNQMTTKIDRNQALLVPALGESWGHAATIRLIFHWDRKQRLATLYKSPSQKECTVLFQIKPQGFRDTVVTSACSLQTEGSLSTRKRSRDPEEEL"


def test_synonymous_reuses_wt_and_delta_zero():
    row = {
        "var_id": "syn_1",
        "split": "train",
        "slim_consequence": "synonymous",
        "pp": 10,
        "ref_aa": "T",
        "alt_aa": "T",
        "HGVSp": "ENSP00000336701.4:p.Thr10=",
        "mut_seq": WT,
    }
    rec = build_variant_record(row, WT)
    assert rec["slot_kind"] == ["paired"]
    assert rec["mut_seq"] == WT
    assert rec["delta_valid"] == [True]
    assert rec["wt_present"] == [1]
    assert rec["mut_present"] == [1]
    assert rec["in_eval_scope"] is True


def test_indel_delins_is_valid_and_preserves_tokens():
    row = {
        "var_id": "indel_1",
        "split": "train",
        "slim_consequence": "clinical_inframe_deletion",
        "pp": 338,
        "ref_aa": "L",
        "alt_aa": "Q",
        "HGVSp": "ENSP00000336701.4:p.Leu338_Lys342delinsGln",
    }
    rec = build_variant_record(row, WT)
    assert rec["variant_type"] == "inframe_indel"
    assert rec["n_del"] == 5
    assert rec["n_ins"] == 1
    assert rec["wt_only_positions"] == [338, 339, 340, 341, 342]
    assert rec["mut_only_positions"] == [338]
    assert rec["wt_pos"][:8] == [337, 338, 339, 340, 341, 342, 0, 343]
    assert rec["mut_pos"][:8] == [337, 0, 0, 0, 0, 0, 338, 339]
    assert rec["slot_kind"] == ["paired", "wt_only", "wt_only", "wt_only", "wt_only", "wt_only", "mut_only", "paired"]
    assert rec["slot_kind_detail"]["wt_only"] == 5
    assert rec["slot_kind_detail"]["mut_only"] == 1
    assert rec["slot_kind_detail"]["paired"] == 2
    assert rec["delta_valid"] == [True, False, False, False, False, False, False, True]
    assert rec["token_valid"] == [True, True, True, True, True, True, True, True]
    assert rec["boundary_wt"] == 343
    assert rec["boundary_mut"] == 339
    assert rec["in_eval_scope"] is True


def test_unsupported_frameshift_excluded():
    row = {
        "var_id": "frameshift_1",
        "split": "train",
        "slim_consequence": "frameshift",
        "pp": 10,
        "ref_aa": "T",
        "alt_aa": "A",
        "HGVSp": "ENSP00000336701.4:p.Thr10fs",
    }
    rec = build_variant_record(row, WT)
    assert rec["in_eval_scope"] is False
    assert rec["slot_kind"] == ["unknown_mut"]


def test_audit_manifest_has_separate_variant_and_slot_counts():
    df = pd.DataFrame(
        [
            {
                "var_id": "a",
                "split": "train",
                "slim_consequence": "missense",
                "pp": 10,
                "ref_aa": "T",
                "alt_aa": "A",
                "HGVSp": "ENSP00000336701.4:p.Thr10Ala",
                "mut_seq": WT[:9] + "A" + WT[10:],
            },
            {
                "var_id": "b",
                "split": "train",
                "slim_consequence": "synonymous",
                "pp": 11,
                "ref_aa": "R",
                "alt_aa": "R",
                "HGVSp": "ENSP00000336701.4:p.Arg11=",
                "mut_seq": WT,
            },
            {
                "var_id": "c",
                "split": "train",
                "slim_consequence": "frameshift",
                "pp": 12,
                "ref_aa": "K",
                "alt_aa": "*",
                "HGVSp": "ENSP00000336701.4:p.Lys12fs",
                "mut_seq": WT,
            },
        ]
    )
    out = audit_manifest(df, WT)
    assert out["summary"]["variant_type_counts"]["missense"] >= 1
    assert out["summary"]["variant_type_counts"]["synonymous"] >= 1
    assert out["summary"]["variant_type_counts"]["unsupported"] >= 1
    assert out["summary"]["slot_kind_counts"]["paired"] >= 2
    assert out["summary"]["slot_kind_counts"]["unknown_mut"] >= 1
    assert all(r["in_eval_scope"] for r in out["rows"] if r["var_id"] in {"a", "b"})


# ==========================================================================
# the split rule
#
# cross_span is decided ONLY by the WT positions the edit directly touches.
# Paired context / flank residues are attention context, not edited residues,
# and MUT coordinates are never read as WT split positions. These tests pin
# that down against the failure mode they replace: deriving cross_span from
# rec["wt_pos"] + rec["mut_pos"] flagged 227 one-residue deletions whose own
# deleted residue sat in a single split, purely because a flank did not.
# ==========================================================================
AA1to3 = {v: k for k, v in __import__("dataset").AA3to1.items()}
REAL_MANIFEST = "data/split_manifest.csv"


def _pos_row(var_id: str, split: str, pos: int) -> dict:
    """A plain missense row, used only to give a position a split."""
    return {
        "var_id": var_id,
        "split": split,
        "slim_consequence": "missense",
        "pp": pos,
        "ref_aa": WT[pos - 1],
        "alt_aa": "A",
        "HGVSp": f"ENSP00000336701.4:p.{AA1to3[WT[pos - 1]]}{pos}Ala",
        "mut_seq": WT[: pos - 1] + "A" + WT[pos:],
    }


def _del_row(var_id: str, split: str, start: int, end: int | None = None) -> dict:
    end = start if end is None else end
    body = f"{AA1to3[WT[start - 1]]}{start}"
    if end != start:
        body += f"_{AA1to3[WT[end - 1]]}{end}"
    return {
        "var_id": var_id,
        "split": split,
        "slim_consequence": "codon_deletion",
        "pp": start,
        "ref_aa": WT[start - 1],
        "alt_aa": None,
        "HGVSp": f"ENSP00000336701.4:p.{body}del",
    }


def test_one_residue_deletion_is_not_cross_span_when_only_a_flank_differs():
    """The deleted residue is train; its paired flanks are val and test.

    The edit touches exactly one WT position, so it belongs to exactly one
    split and must stay in scope -- even though the alignment legitimately
    reaches into residues that belong elsewhere.
    """
    df = pd.DataFrame([
        _pos_row("left_flank", "val", 99),
        _del_row("del_100", "train", 100),
        _pos_row("right_flank", "test", 101),
    ])
    rec = build_variant_record(_del_row("del_100", "train", 100), WT, manifest_df=df)

    assert rec["split_positions_considered"] == [100]
    assert rec["split_position_splits"] == {"train": [100]}
    assert rec["cross_span"] is False
    assert rec["exclusion_reason"] is None
    assert rec["in_eval_scope"] is True
    # the flanks really are in the alignment -- they are just not the rule's input
    assert 99 in rec["wt_pos"] and 101 in rec["wt_pos"]


def test_multi_residue_delins_crossing_two_splits_is_excluded():
    """p.Leu338_Lys342delinsGln directly edits WT 338..342, which straddles
    train (338-341) and val (342)."""
    row = {
        "var_id": "delins_338",
        "split": "train",
        "slim_consequence": "clinical_inframe_deletion",
        "pp": 338,
        "ref_aa": "L",
        "alt_aa": "Q",
        "HGVSp": "ENSP00000336701.4:p.Leu338_Lys342delinsGln",
    }
    df = pd.DataFrame(
        [_pos_row(f"p{p}", "train" if p <= 341 else "val", p) for p in range(338, 343)]
        + [row]
    )
    rec = build_variant_record(row, WT, manifest_df=df)

    assert rec["split_positions_considered"] == [338, 339, 340, 341, 342]
    assert rec["cross_span"] is True
    assert rec["cross_span_splits"] == ["train", "val"]
    assert rec["exclusion_reason"] == "cross_span_edit"
    assert rec["in_eval_scope"] is False


def test_delins_wholly_inside_one_split_is_kept():
    """Same 5-residue span, all of it train -> in scope. Confirms the delins is
    excluded for its span, not for being a delins."""
    row = {
        "var_id": "delins_338",
        "split": "train",
        "slim_consequence": "clinical_inframe_deletion",
        "pp": 338,
        "ref_aa": "L",
        "alt_aa": "Q",
        "HGVSp": "ENSP00000336701.4:p.Leu338_Lys342delinsGln",
    }
    df = pd.DataFrame(
        [_pos_row(f"p{p}", "train", p) for p in range(330, 350)] + [row]
    )
    rec = build_variant_record(row, WT, manifest_df=df)
    assert rec["cross_span"] is False
    assert rec["in_eval_scope"] is True


def test_mut_coordinates_are_never_used_as_wt_split_positions():
    """A deletion frame-shifts every downstream MUT coordinate. Those MUT
    numbers must not be looked up in the WT position->split table."""
    df = pd.DataFrame([
        _pos_row("a", "train", 100),
        _pos_row("b", "val", 99),        # == the MUT coordinate of WT 101
        _del_row("del_100", "train", 100),
    ])
    rec = build_variant_record(_del_row("del_100", "train", 100), WT, manifest_df=df)
    assert 99 in rec["mut_pos"]                  # MUT side really does hit 99
    assert rec["split_positions_considered"] == [100]
    assert rec["cross_span"] is False


def test_both_real_duplication_rows_use_the_insertion_boundary_rule():
    """The two real dup rows, decided on the WT boundary the copy lands at.

    p.Lys186dup inserts between WT 186 and 187, both train -> in scope.
    p.Val351dup inserts between WT 351 (test) and 352 (val) -> excluded.
    Neither decision may involve the duplicated residue's flank or any MUT
    coordinate.
    """
    df = pd.read_csv(REAL_MANIFEST)
    dups = df[df.HGVSp.astype(str).str.contains("dup", na=False)]
    assert len(dups) == 2, "expected exactly two real duplication rows"

    expected = {
        "ENSP00000336701.4:p.Lys186dup": ([186, 187], {"train": [186, 187]}, False, True),
        "ENSP00000336701.4:p.Val351dup": ([351, 352], {"test": [351], "val": [352]}, True, False),
    }
    seen = set()
    for _, raw in dups.iterrows():
        rec = build_variant_record(raw.to_dict(), WT, manifest_df=df)
        positions, splits, cross, in_scope = expected[raw.HGVSp]
        assert rec["edit_type"] == "duplication"
        assert rec["split_positions_considered"] == positions
        assert rec["split_position_splits"] == splits
        assert rec["cross_span"] is cross
        assert rec["in_eval_scope"] is in_scope
        # the boundary, not the duplicated block's flank: 185 is in the
        # alignment but must not be part of the decision
        assert 185 not in rec["split_positions_considered"]
        seen.add(raw.HGVSp)
    assert seen == set(expected)


def test_every_real_deletion_is_single_residue_and_therefore_in_scope():
    """Regression on the corrected counts: all 357 real deletion rows delete a
    single residue, so none of them can be cross-span."""
    df = pd.read_csv(REAL_MANIFEST)
    dels = df[df.HGVSp.astype(str).str.endswith("del")]
    assert len(dels) > 0
    for _, raw in dels.iterrows():
        rec = build_variant_record(raw.to_dict(), WT, manifest_df=df)
        assert len(rec["split_positions_considered"]) == 1
        assert rec["cross_span"] is False
        assert rec["in_eval_scope"] is True


def test_real_manifest_excludes_exactly_the_two_multi_split_edits():
    """The whole-manifest outcome of the corrected rule."""
    df = pd.read_csv(REAL_MANIFEST)
    out = audit_manifest(df, WT)
    excluded = [r for r in out["rows"] if not r["in_eval_scope"]]
    assert {r["var_id"] for r in excluded} == {
        "chr17_58734138__TGT",              # p.Val351dup   test|val
        "chr17_58732531_TGTTTCAAATCA_",     # delins 338-342 train|val
    }
    assert out["summary"]["excluded"] == 2
    assert out["summary"]["supported"] == len(df) - 2
    assert out["summary"]["excluded_reasons"] == {"cross_span_edit": 2}
