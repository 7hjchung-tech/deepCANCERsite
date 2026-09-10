from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from dataset import parse_protein_change

SUPPORTED_CONSEQUENCES = {
    "missense",
    "synonymous",
    "clinical_inframe_deletion",
    "clinical_inframe_insertion",
    "codon_deletion",
}

SLOT_KINDS = ("paired", "wt_only", "mut_only", "unknown_mut", "pad")

#: The strict split rule. An in-frame edit is excluded as `cross_span_edit` only
#: when the WT positions it DIRECTLY edits fall in more than one split. Paired
#: context / flank residues are attention context, not edited residues, and MUT
#: coordinates are never interpreted as WT split positions.
SPLIT_RULE = (
    "cross_span is decided only by the directly edited WT positions: "
    "deletion -> the deleted WT span start..end; "
    "delins -> the directly edited WT span start..end; "
    "insertion -> the WT insertion boundary positions (start, start+1); "
    "duplication -> the WT boundary where the duplicate copy is inserted "
    "(end, end+1). Paired context/window residues are excluded, and MUT "
    "coordinates are never read as WT split positions."
)
SPLIT_RULE_VERSION = "split_rule_v2_edited_wt_span_only"


def canonical_aa_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    txt = str(value).strip()
    return txt if len(txt) == 1 else txt[:1]


def normalize_edit_key(wt_seq: str, mut_seq: str, pp: int | None, consequence: str) -> str:
    payload = f"{pp or 0}|{consequence}|{wt_seq}|{mut_seq}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _edit_type_from_row(row: Dict[str, Any]) -> str:
    consequence = str(row.get("slim_consequence", "")).strip()
    if consequence == "missense":
        return "missense"
    if consequence == "synonymous":
        return "synonymous"
    parsed = parse_protein_change(row.get("HGVSp"))
    if parsed is None:
        return consequence or "unknown"
    op = parsed.get("op", "unknown")
    if op == "del":
        return "deletion"
    if op == "dup":
        return "duplication"
    if op == "ins":
        return "insertion"
    if op == "delins":
        return "delins"
    return consequence or op


def _edit_geometry(op: str, start: int, end: int, inserted: str) -> Dict[str, Any]:
    """Residue-level geometry of one in-frame edit, in WT/MUT 1-based coordinates.

    Mirrors dataset.build_mutant_sequence() exactly -- that function defines the
    MUT sequence, so the coordinates here must agree with it or the gathered
    representations would be paired against the wrong residues:

        del     s..e   mut = wt[:s-1] + wt[e:]                 WT p>e  -> MUT p-n_del
        delins  s..e   mut = wt[:s-1] + inserted + wt[e:]      WT p>e  -> MUT p-n_del+n_ins
        dup     s..e   mut = wt[:e] + wt[s-1:e] + wt[e:]       WT p>e  -> MUT p+n_ins
        ins     s,e=s+1 mut = wt[:s] + inserted + wt[s:]       WT p>s  -> MUT p+n_ins

    `dup` duplicates the block s..e, so it INSERTS (e-start+1) residues even
    though HGVSp carries no inserted string; the copy lands at MUT e+1..e+n_ins
    and the original block stays paired. `ins` places the new residues AFTER WT
    position `start`, i.e. at MUT start+1..start+n_ins.

    Three derived quantities define the *edit event* for everything downstream:

    left_boundary_wt / right_boundary_wt
        The last unchanged WT residue before the event and the first unchanged
        WT residue after it. Nothing between them survives untouched. They are
        the anchors the Task C analysis window is measured from -- the flank is
        counted inward from these, never from a slot list that already contains
        them.

    right_offset = n_ins - n_del
        MUT coordinate of any WT position at or after right_boundary_wt is
        wt_pos + right_offset. Left of the event the offset is 0.

    edited_wt_positions
        The WT positions the edit *directly* touches, and the ONLY thing the
        split rule may look at:

            del     -> the deleted WT span, start..end
            delins  -> the directly edited WT span, start..end
            ins     -> the two WT residues straddling the insertion point
            dup     -> the two WT residues straddling the point where the
                       duplicate copy is inserted

        Paired context / flank residues are deliberately NOT in this list, and
        MUT coordinates are never interpreted as WT split positions.
    """
    wt_only_positions: List[int] = []
    mut_only_positions: List[int] = []

    if op in {"del", "delins"}:
        wt_only_positions = list(range(start, end + 1))
    if op == "dup":
        mut_only_positions = [end + 1 + i for i in range(end - start + 1)]
    elif op == "delins" and inserted:
        mut_only_positions = [start + i for i in range(len(inserted))]
    elif op == "ins" and inserted:
        mut_only_positions = [start + 1 + i for i in range(len(inserted))]

    n_del = len(wt_only_positions)
    n_ins = len(mut_only_positions)

    if op in {"del", "delins"}:
        left_boundary_wt = start - 1
        right_boundary_wt = end + 1
        edited_wt_positions = list(range(start, end + 1))
    elif op == "dup":
        # WT residues s..e are untouched; the copy is inserted AFTER e, so the
        # edited locus is the boundary between WT e and WT e+1.
        left_boundary_wt = end
        right_boundary_wt = end + 1
        edited_wt_positions = [end, end + 1]
    elif op == "ins":
        # HGVS gives end == start + 1; the new residues land between them.
        left_boundary_wt = start
        right_boundary_wt = start + 1
        edited_wt_positions = [start, start + 1]
    else:
        left_boundary_wt = None
        right_boundary_wt = None
        edited_wt_positions = []

    right_offset = n_ins - n_del

    return {
        "wt_only_positions": wt_only_positions,
        "mut_only_positions": mut_only_positions,
        "n_del": n_del,
        "n_ins": n_ins,
        "left_boundary_wt": left_boundary_wt,
        "right_boundary_wt": right_boundary_wt,
        "left_offset": 0,
        "right_offset": right_offset,
        "edited_wt_positions": edited_wt_positions,
        # Kept for continuity: `boundary_wt`/`boundary_mut` have always meant the
        # first paired residue AFTER the edit and its MUT partner.
        "boundary_wt": right_boundary_wt,
        "boundary_mut": (
            None if right_boundary_wt is None else right_boundary_wt + right_offset
        ),
    }


def _indel_span_info(row: Dict[str, Any]) -> Dict[str, Any]:
    parsed = parse_protein_change(row.get("HGVSp"))
    if parsed is None:
        return {
            "op": "unknown",
            "start": None,
            "end": None,
            "inserted": "",
            "wt_only_positions": [],
            "mut_only_positions": [],
            "left_boundary_wt": None,
            "right_boundary_wt": None,
            "left_offset": 0,
            "right_offset": 0,
            "edited_wt_positions": [],
            "boundary_wt": None,
            "boundary_mut": None,
            "n_del": 0,
            "n_ins": 0,
        }

    start = int(parsed["start"])
    end = int(parsed["end"])
    op = parsed["op"]
    inserted = str(parsed.get("inserted", "") or "")
    geom = _edit_geometry(op, start, end, inserted)

    return {
        "op": op,
        "start": start,
        "end": end,
        "inserted": inserted,
        **geom,
    }


def _split_group_positions(df: pd.DataFrame, positions: List[int]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for pos in positions:
        if pos is None:
            continue
        row_matches = df[df["pp"] == pos]
        if row_matches.empty:
            continue
        for split_name in row_matches["split"].dropna().unique():
            groups.setdefault(str(split_name), []).append(int(pos))
    return groups


def _alignment_for_single_point(pos: int, wt_len: int, slot_kind: str = "paired") -> Dict[str, Any]:
    return {
        "wt_pos": [pos],
        "mut_pos": [pos],
        "wt_present": [1 if 1 <= pos <= wt_len else 0],
        "mut_present": [1 if 1 <= pos <= wt_len else 0],
        "delta_valid": [True],
        "token_valid": [True],
        "slot_kind": [slot_kind],
    }


def _alignment_for_indel(row: Dict[str, Any], wt_seq: str) -> Dict[str, Any]:
    parsed = parse_protein_change(row.get("HGVSp"))
    if parsed is None:
        return {
            "wt_pos": [],
            "mut_pos": [],
            "wt_present": [],
            "mut_present": [],
            "delta_valid": [],
            "token_valid": [],
            "slot_kind": [],
        }

    start = int(parsed["start"])
    end = int(parsed["end"])
    op = parsed["op"]
    inserted = str(parsed.get("inserted", "") or "")
    geom = _edit_geometry(op, start, end, inserted)
    deleted_positions = geom["wt_only_positions"]
    inserted_positions = geom["mut_only_positions"]

    entries: List[Dict[str, Any]] = []
    # Left boundary pair: the last WT residue the edit leaves untouched. For
    # del/delins/dup that is start-1 (dup then re-pairs its source block below);
    # for a pure insertion the new residues land AFTER WT `start`, so `start`
    # itself is the left boundary and stays paired.
    left_boundary = start if op == "ins" else start - 1
    if left_boundary >= 1:
        entries.append({
            "wt_pos": left_boundary,
            "mut_pos": left_boundary,
            "wt_present": 1,
            "mut_present": 1,
            "delta_valid": True,
            "token_valid": True,
            "slot_kind": "paired",
        })

    for pos in deleted_positions:
        entries.append({
            "wt_pos": pos,
            "mut_pos": 0,
            "wt_present": 1,
            "mut_present": 0,
            "delta_valid": False,
            "token_valid": True,
            "slot_kind": "wt_only",
        })

    # A duplication changes no WT residue: the original block s..e stays paired
    # (WT p -> MUT p), and only the extra copy at MUT e+1..e+n_ins is mut_only.
    if op == "dup":
        for pos in range(start, end + 1):
            entries.append({
                "wt_pos": pos,
                "mut_pos": pos,
                "wt_present": 1,
                "mut_present": 1,
                "delta_valid": True,
                "token_valid": True,
                "slot_kind": "paired",
            })

    for mut_pos in inserted_positions:
        entries.append({
            "wt_pos": 0,
            "mut_pos": mut_pos,
            "wt_present": 0,
            "mut_present": 1,
            "delta_valid": False,
            "token_valid": True,
            "slot_kind": "mut_only",
        })

    right_wt = geom["right_boundary_wt"]
    if right_wt is not None and right_wt <= len(wt_seq):
        right_mut = right_wt + geom["right_offset"]
        entries.append({
            "wt_pos": right_wt,
            "mut_pos": right_mut,
            "wt_present": 1,
            "mut_present": 1,
            "delta_valid": True,
            "token_valid": True,
            "slot_kind": "paired",
        })

    return {
        "wt_pos": [e["wt_pos"] for e in entries],
        "mut_pos": [e["mut_pos"] for e in entries],
        "wt_present": [e["wt_present"] for e in entries],
        "mut_present": [e["mut_present"] for e in entries],
        "delta_valid": [e["delta_valid"] for e in entries],
        "token_valid": [e["token_valid"] for e in entries],
        "slot_kind": [e["slot_kind"] for e in entries],
    }


def build_alignment_arrays(row: Dict[str, Any], wt_seq: str) -> Dict[str, Any]:
    consequence = str(row.get("slim_consequence", "")).strip()
    if consequence == "missense":
        pos = int(row["pp"]) if row.get("pp") is not None and not pd.isna(row.get("pp")) else 0
        return _alignment_for_single_point(pos, len(wt_seq), "paired")
    if consequence == "synonymous":
        pos = int(row["pp"]) if row.get("pp") is not None and not pd.isna(row.get("pp")) else 0
        return _alignment_for_single_point(pos, len(wt_seq), "paired")
    parsed = parse_protein_change(row.get("HGVSp"))
    if parsed is not None and parsed.get("op") in {"del", "ins", "delins", "dup"}:
        return _alignment_for_indel(row, wt_seq)
    return {
        "wt_pos": [0],
        "mut_pos": [0],
        "wt_present": [0],
        "mut_present": [0],
        "delta_valid": [False],
        "token_valid": [False],
        "slot_kind": ["unknown_mut"],
    }


def build_variant_record(row: Dict[str, Any], wt_seq: str, manifest_df: pd.DataFrame | None = None) -> Dict[str, Any]:
    consequence = str(row.get("slim_consequence", "")).strip()
    pp = row.get("pp")
    pos = int(pp) if pp is not None and not pd.isna(pp) else None
    mut_seq = str(row.get("mut_seq") or wt_seq)
    wt_len = len(wt_seq)
    variant_type = consequence if consequence in {"missense", "synonymous", "clinical_inframe_deletion", "clinical_inframe_insertion", "codon_deletion"} else "unsupported"
    edit_type = _edit_type_from_row(row)

    if consequence == "missense":
        variant_type = "missense"
    elif consequence == "synonymous":
        variant_type = "synonymous"
    elif consequence in {"clinical_inframe_deletion", "clinical_inframe_insertion", "codon_deletion"}:
        variant_type = "inframe_indel"
    else:
        variant_type = "unsupported"

    alignment = build_alignment_arrays(row, wt_seq)
    slot_kind = alignment["slot_kind"]
    slot_detail = {key: slot_kind.count(key) for key in SLOT_KINDS}

    rec: Dict[str, Any] = {
        "var_id": row.get("var_id", ""),
        "split": row.get("split", "unassigned"),
        "variant_type": variant_type,
        "edit_type": edit_type,
        "slim_consequence": consequence,
        "pp": pos,
        "wt_len": wt_len,
        "mut_len": len(mut_seq),
        "wt_seq": wt_seq,
        "mut_seq": mut_seq,
        "wt_pos": alignment["wt_pos"],
        "mut_pos": alignment["mut_pos"],
        "wt_present": alignment["wt_present"],
        "mut_present": alignment["mut_present"],
        "delta_valid": alignment["delta_valid"],
        "token_valid": alignment["token_valid"],
        "slot_kind": slot_kind,
        "slot_kind_detail": slot_detail,
        "cross_span": False,
        "cross_span_splits": [],
        "split_rule_version": SPLIT_RULE_VERSION,
        "edited_wt_positions": [],
        "split_positions_considered": [],
        "split_position_splits": {},
        "in_eval_scope": False,
        "exclusion_reason": None,
        "edit_key": normalize_edit_key(wt_seq, mut_seq, pos, consequence),
        "seq_hash": hashlib.sha256(mut_seq.encode("utf-8")).hexdigest(),
    }

    if consequence == "missense":
        ref_aa = canonical_aa_code(row.get("ref_aa"))
        alt_aa = canonical_aa_code(row.get("alt_aa"))
        rec.update({
            "wt_present": [1 if pos is not None and 1 <= pos <= wt_len else 0],
            "mut_present": [1 if pos is not None and 1 <= pos <= wt_len else 0],
            "delta_valid": [True],
            "token_valid": [True],
            "wt_pos": [pos],
            "mut_pos": [pos],
            "slot_kind": ["paired"],
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "slot_kind_detail": {"paired": 1, "wt_only": 0, "mut_only": 0, "unknown_mut": 0, "pad": 0},
        })
        rec["in_eval_scope"] = consequence in SUPPORTED_CONSEQUENCES
        return rec

    if consequence == "synonymous":
        rec.update({
            "wt_present": [1 if pos is not None and 1 <= pos <= wt_len else 0],
            "mut_present": [1 if pos is not None and 1 <= pos <= wt_len else 0],
            "delta_valid": [True],
            "token_valid": [True],
            "wt_pos": [pos],
            "mut_pos": [pos],
            "slot_kind": ["paired"],
            "slot_kind_detail": {"paired": 1, "wt_only": 0, "mut_only": 0, "unknown_mut": 0, "pad": 0},
        })
        rec["in_eval_scope"] = consequence in SUPPORTED_CONSEQUENCES
        return rec

    if consequence in {"clinical_inframe_deletion", "clinical_inframe_insertion", "codon_deletion"}:
        parsed = parse_protein_change(row.get("HGVSp"))
        if parsed is not None:
            start = int(parsed["start"])
            end = int(parsed["end"])
            inserted_str = str(parsed.get("inserted", "") or "")
            geom = _edit_geometry(parsed["op"], start, end, inserted_str)
            rec.update({
                **geom,
                "start": start,
                "end": end,
                "inserted": inserted_str,
                "op": parsed["op"],
            })
        rec["slot_kind_detail"] = {key: slot_kind.count(key) for key in SLOT_KINDS}

        if manifest_df is not None:
            # THE SPLIT RULE. Only the directly edited WT positions are consulted
            # (see SPLIT_RULE): the deleted / directly edited WT span for
            # del+delins, the WT insertion boundary for ins, and the WT boundary
            # where the copy lands for dup. The paired context residues in
            # rec["wt_pos"], and every MUT coordinate in rec["mut_pos"], are
            # deliberately NOT consulted -- a flank residue in another split does
            # not make the edit itself cross-span.
            edited = [p for p in rec.get("edited_wt_positions", []) if 1 <= p <= wt_len]
            rec["split_positions_considered"] = edited
            split_groups = _split_group_positions(manifest_df, edited)
            rec["split_position_splits"] = {
                s: sorted(v) for s, v in sorted(split_groups.items())
            }
            rec["cross_span_splits"] = sorted(split_groups.keys())
            rec["cross_span"] = len(split_groups) > 1
            if rec["cross_span"]:
                rec["in_eval_scope"] = False
                rec["exclusion_reason"] = "cross_span_edit"
            else:
                rec["in_eval_scope"] = consequence in SUPPORTED_CONSEQUENCES
                rec["exclusion_reason"] = None
        else:
            rec["split_positions_considered"] = [
                p for p in rec.get("edited_wt_positions", []) if 1 <= p <= wt_len
            ]
            rec["split_position_splits"] = {}
            rec["in_eval_scope"] = consequence in SUPPORTED_CONSEQUENCES
            rec["exclusion_reason"] = None
        return rec

    rec.update({
        "variant_type": "unsupported",
        "slot_kind": ["unknown_mut"],
        "slot_kind_detail": {"paired": 0, "wt_only": 0, "mut_only": 0, "unknown_mut": 1, "pad": 0},
        "wt_present": [0],
        "mut_present": [0],
        "delta_valid": [False],
        "token_valid": [False],
        "wt_pos": [0],
        "mut_pos": [0],
        "exclusion_reason": "unsupported_consequence",
    })
    rec["in_eval_scope"] = False
    return rec


def audit_manifest(df: pd.DataFrame, wt_seq: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    summary = {
        "total": len(df),
        "supported": 0,
        "excluded": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "supported_by_split": {"train": 0, "val": 0, "test": 0},
        "supported_by_split_and_variant_type": {"train": {"missense": 0, "synonymous": 0, "inframe_indel": 0, "unsupported": 0}, "val": {"missense": 0, "synonymous": 0, "inframe_indel": 0, "unsupported": 0}, "test": {"missense": 0, "synonymous": 0, "inframe_indel": 0, "unsupported": 0}},
        "variant_type_counts": {"missense": 0, "synonymous": 0, "inframe_indel": 0, "unsupported": 0},
        "edit_type_counts": {"missense": 0, "synonymous": 0, "deletion": 0, "insertion": 0, "duplication": 0, "delins": 0, "unsupported": 0},
        "slot_kind_counts": {"paired": 0, "wt_only": 0, "mut_only": 0, "unknown_mut": 0, "pad": 0},
        "edit_slot_kind_counts": {"paired": 0, "wt_only": 0, "mut_only": 0, "unknown_mut": 0, "pad": 0},
        "all_window_slot_kind_counts": {"paired": 0, "wt_only": 0, "mut_only": 0, "unknown_mut": 0, "pad": 0},
        "excluded_reasons": {},
        "excluded_by_variant_type": {},
        "excluded_by_edit_type": {},
        "cross_span_breakdown": {"variant_type": {}, "edit_type": {}},
        "cross_span_examples": [],
        "split_rule": SPLIT_RULE,
        "split_rule_version": SPLIT_RULE_VERSION,
        "indel_split_decisions": [],
    }

    for _, row in df.iterrows():
        rec = build_variant_record(row.to_dict(), wt_seq, manifest_df=df)
        rows.append(rec)

        if rec["variant_type"] == "inframe_indel":
            summary["indel_split_decisions"].append({
                "var_id": rec["var_id"],
                "HGVSp": row.get("HGVSp"),
                "edit_type": rec["edit_type"],
                "row_split": rec["split"],
                "edited_wt_positions": rec.get("split_positions_considered", []),
                "splits_of_edited_positions": rec.get("split_position_splits", {}),
                "cross_span": rec["cross_span"],
                "in_eval_scope": rec["in_eval_scope"],
            })

        summary["variant_type_counts"][rec["variant_type"]] = summary["variant_type_counts"].get(rec["variant_type"], 0) + 1
        summary["edit_type_counts"][rec["edit_type"]] = summary["edit_type_counts"].get(rec["edit_type"], 0) + 1

        slot_counter = Counter(rec["slot_kind"])
        for key in SLOT_KINDS:
            count = int(slot_counter.get(key, 0))
            summary["slot_kind_counts"][key] += count
            summary["edit_slot_kind_counts"][key] += count
            summary["all_window_slot_kind_counts"][key] += count

        if rec["in_eval_scope"]:
            summary["supported"] += 1
            split_name = str(rec["split"]).lower()
            if split_name in summary["supported_by_split"]:
                summary["supported_by_split"][split_name] += 1
            summary["split_counts"][split_name] = summary["split_counts"].get(split_name, 0) + 1
            if split_name in summary["supported_by_split_and_variant_type"]:
                vt = rec["variant_type"]
                summary["supported_by_split_and_variant_type"][split_name][vt] = summary["supported_by_split_and_variant_type"][split_name].get(vt, 0) + 1
        else:
            summary["excluded"] += 1
            summary["excluded_by_variant_type"][rec["variant_type"]] = summary["excluded_by_variant_type"].get(rec["variant_type"], 0) + 1
            summary["excluded_by_edit_type"][rec["edit_type"]] = summary["excluded_by_edit_type"].get(rec["edit_type"], 0) + 1
            if rec["cross_span"]:
                summary["cross_span_breakdown"]["variant_type"][rec["variant_type"]] = summary["cross_span_breakdown"]["variant_type"].get(rec["variant_type"], 0) + 1
                summary["cross_span_breakdown"]["edit_type"][rec["edit_type"]] = summary["cross_span_breakdown"]["edit_type"].get(rec["edit_type"], 0) + 1

            reason = rec["exclusion_reason"] or "unknown"
            summary["excluded_reasons"][reason] = summary["excluded_reasons"].get(reason, 0) + 1
            if rec["cross_span"]:
                summary["cross_span_examples"].append({
                    "var_id": rec["var_id"],
                    "variant_type": rec["variant_type"],
                    "edit_type": rec["edit_type"],
                    "split": rec["split"],
                    "pp": rec["pp"],
                    "start": rec.get("start"),
                    "end": rec.get("end"),
                    "HGVSp": row.get("HGVSp"),
                    "edited_wt_positions": rec.get("split_positions_considered", []),
                    "splits_of_edited_positions": rec.get("split_position_splits", {}),
                    "cross_span_splits": rec.get("cross_span_splits", []),
                    "slot_kind": rec.get("slot_kind"),
                })

    return {"rows": rows, "summary": summary}


def _write_manifest_and_audit(df: pd.DataFrame, wt_seq: str, out_dir: str) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    audited = audit_manifest(df, wt_seq)
    manifest = out_path / "validated_manifest.csv"
    out_df = df.copy()
    for field in [
        "in_eval_scope",
        "variant_type",
        "edit_type",
        "slot_kind",
        "slot_kind_detail",
        "wt_present",
        "mut_present",
        "delta_valid",
        "token_valid",
        "n_del",
        "n_ins",
        "wt_pos",
        "mut_pos",
        "wt_only_positions",
        "mut_only_positions",
        "left_boundary_wt",
        "right_boundary_wt",
        "right_offset",
        "edited_wt_positions",
        "split_positions_considered",
        "split_rule_version",
        "cross_span",
        "cross_span_splits",
        "exclusion_reason",
        "edit_key",
        "seq_hash",
    ]:
        out_df[field] = [r.get(field) for r in audited["rows"]]
    out_df.to_csv(manifest, index=False)
    (out_path / "audit.json").write_text(json.dumps(audited, indent=2), encoding="utf-8")
    return audited


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifest rows and produce a compact ESM alignment audit for the supported edit scope.")
    parser.add_argument("--manifest", default="data/split_manifest.csv")
    parser.add_argument("--wt-seq", default="data/wt_sequence.txt")
    parser.add_argument("--out-dir", default="data/variant_audit")
    args = parser.parse_args()

    wt_seq = Path(args.wt_seq).read_text(encoding="utf-8").strip()
    df = pd.read_csv(args.manifest)
    audit = _write_manifest_and_audit(df, wt_seq, args.out_dir)
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
