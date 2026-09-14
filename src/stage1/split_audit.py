"""Edited-span vs. split-position overlap audit.

The shipped split (dataset.py: split_by_position) groups every variant by its
single ANCHOR position (`pp`). That guarantees no two variants that share an
anchor land in different splits. It does NOT guarantee that an indel's full
edited span -- which can touch several WT residues other than its anchor --
stays inside one split: some OTHER variant anchored at one of those residues
may have been assigned to a different split.

This module only reports that; it never rewrites the shipped split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .alignment import NormalizedEdit


def edited_span_positions(edit: NormalizedEdit) -> set[int]:
    """1-based WT positions actually touched by the edit (anchor position(s)
    for a pure insertion, since it removes no WT residue itself)."""
    if edit.d >= 1:
        return set(range(edit.u + 1, edit.u + edit.d + 1))
    # pure insertion (d == 0): the two WT residues bracketing the boundary
    lo = max(1, edit.u)
    hi = min(edit.wt_len, edit.u + 1)
    return {lo, hi}


@dataclass
class SplitAuditResult:
    n_variants: int
    n_indels_audited: int
    start_position_overlaps: list[dict] = field(default_factory=list)   # should always be empty
    edited_span_overlaps: list[dict] = field(default_factory=list)
    unassigned_span_positions: list[dict] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.start_position_overlaps and not self.edited_span_overlaps


def audit_split_overlap(entries: list[dict], position_to_split: dict[int, str]) -> SplitAuditResult:
    """entries: [{"var_id", "row" (with 'split'), "edit"}], as built by
    stage1.dataset.build_cohort. position_to_split: {pp: split} built from
    the FULL manifest (every variant's own anchor -> its own split).
    """
    result = SplitAuditResult(n_variants=len(entries), n_indels_audited=0)

    for e in entries:
        edit: NormalizedEdit = e["edit"]
        own_split = e["row"].get("split")
        own_pos = int(e["row"]["pp"])

        # (a) start-position consistency -- should hold by construction.
        if position_to_split.get(own_pos) != own_split:
            result.start_position_overlaps.append(
                {"var_id": e["var_id"], "pp": own_pos, "own_split": own_split,
                 "position_split": position_to_split.get(own_pos)}
            )

        if edit.edit_type in ("missense", "synonymous"):
            continue
        result.n_indels_audited += 1

        # (b) edited-span consistency -- NOT guaranteed by construction.
        for pos in sorted(edited_span_positions(edit)):
            other_split = position_to_split.get(pos)
            if other_split is None:
                result.unassigned_span_positions.append(
                    {"var_id": e["var_id"], "position": pos}
                )
            elif other_split != own_split:
                result.edited_span_overlaps.append(
                    {"var_id": e["var_id"], "position": pos,
                     "own_split": own_split, "position_split": other_split}
                )
    return result


def build_position_to_split(manifest_rows: list[dict]) -> dict[int, str]:
    return {int(r["pp"]): r["split"] for r in manifest_rows if r.get("pp") is not None}
