"""Normalized WT<->MUT residue alignment for Stage 1.

Reuses the existing HGVSp parser (dataset.py: parse_protein_change / AA3to1)
rather than re-deriving it, per the "reuse the existing HGVS/parser" rule.
This module adds what dataset.py does not have: an explicit (u, d, inserted,
m) normalization, a WT<->MUT coordinate map for the whole protein, and
mutant-reconstruction validation.

Edit normalization
-------------------
Every supported edit (missense / synonymous / deletion / duplication /
insertion / delins) is reduced to:

    MUT = WT[:u] + inserted_seq + WT[u+d:]

    u : 0-based length of the WT prefix left untouched
    d : number of WT residues removed starting at WT[u]
    inserted_seq : the new residues (may be empty), length m

Missense is the special case d=1, m=1 with inserted_seq = alt_aa. Synonymous
is d=1, m=1 with inserted_seq == WT[u] (identity substitution -> exact Δ=0
once both sequences hit the same frozen ESM). This unification is why a
single downstream slot-mapping rule (see build_global_alignment) covers every
supported consequence without special-casing missense/synonymous separately.

Slot mapping rule
------------------
* WT prefix [0, u)            <-> MUT prefix [0, u)              paired
* Special case d==1 and m==1: the single edited residue is ALSO paired
  (this is what makes a validated single substitution a paired slot, per
  the task spec, instead of a WT-only + MUT-only pair).
* Otherwise:
    WT[u : u+d)                                                  wt_only
    MUT[u : u+m)                                                 mut_only
* WT suffix [u+d, wt_len) <-> MUT suffix [u+m, mut_len)           paired
  (MUT index = WT index + (m - d))

This is a purely index-based rule -- it never uses amino-acid identity to
decide pairing (see the delins fixture below: a coincidental shared letter
inside the deleted/inserted span must NOT become a paired slot).
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import parse_protein_change  # noqa: E402  (reuse existing parser)

from .schema import SLOT_MUT_ONLY, SLOT_PAIRED, SLOT_WT_ONLY, SUPPORTED_CONSEQUENCES  # noqa: E402


class UnsupportedVariantError(Exception):
    """Raised for a variant outside the currently supported edit cohort.

    Carries var_id + reason so callers can report (not silently drop) it.
    """

    def __init__(self, var_id: str, reason: str) -> None:
        self.var_id = var_id
        self.reason = reason
        super().__init__(f"{var_id}: unsupported -- {reason}")


class AlignmentValidationError(Exception):
    """Raised when the normalized edit does not reconstruct manifest mut_seq."""

    def __init__(self, var_id: str, reason: str) -> None:
        self.var_id = var_id
        self.reason = reason
        super().__init__(f"{var_id}: alignment validation failed -- {reason}")


@dataclass(frozen=True)
class NormalizedEdit:
    var_id: str
    u: int              # 0-based preserved WT prefix length
    d: int               # WT residues removed, starting at WT[u]
    inserted_seq: str    # new residues (may be empty), length m
    edit_type: str        # missense | synonymous | deletion | insertion | delins
    wt_len: int
    mut_len: int
    op_raw: str           # parser op: missense | synonymous | del | dup | delins | ins
    translation_status: str = "coding"   # constant in the current supported cohort

    @property
    def m(self) -> int:
        return len(self.inserted_seq)

    @property
    def is_single_substitution(self) -> bool:
        return self.d == 1 and self.m == 1


@dataclass(frozen=True)
class AlignedSlot:
    kind: int              # SLOT_PAIRED | SLOT_WT_ONLY | SLOT_MUT_ONLY
    wt_pos: Optional[int]  # 1-based, or None
    mut_pos: Optional[int]  # 1-based, or None
    is_event: bool          # part of the edited region (not pure flank)
    insertion_rank: int = 0  # 1-based order within an inserted span; 0 otherwise


def _classify_edit_type(u: int, d: int, inserted_seq: str, wt_seq: str) -> str:
    m = len(inserted_seq)
    if d == 1 and m == 1:
        return "synonymous" if inserted_seq == wt_seq[u] else "missense"
    if d >= 1 and m == 0:
        return "deletion"
    if d == 0 and m >= 1:
        return "insertion"
    return "delins"


def normalize_edit(wt_seq: str, row: dict) -> NormalizedEdit:
    """Build a NormalizedEdit from one split_manifest.csv row (as a dict).

    Raises UnsupportedVariantError for anything outside SUPPORTED_CONSEQUENCES
    or that the parser cannot resolve cleanly (e.g. 'p.Met1?').
    """
    var_id = str(row["var_id"])
    cons = row["slim_consequence"]
    if cons not in SUPPORTED_CONSEQUENCES:
        raise UnsupportedVariantError(var_id, f"consequence '{cons}' not in supported cohort")

    if cons == "missense":
        pp, ref_aa, alt_aa = row.get("pp"), row.get("ref_aa"), row.get("alt_aa")
        if pp is None or ref_aa is None or alt_aa is None or _isnan(pp):
            raise UnsupportedVariantError(var_id, "missense row missing pp/ref_aa/alt_aa")
        ref_aa, alt_aa = str(ref_aa), str(alt_aa)
        if len(ref_aa) != 1 or len(alt_aa) != 1:
            raise UnsupportedVariantError(var_id, "missense ref_aa/alt_aa is not single-letter")
        pos = int(pp)
        if not (1 <= pos <= len(wt_seq)) or wt_seq[pos - 1] != ref_aa:
            raise AlignmentValidationError(
                var_id, f"WT[{pos}]='{wt_seq[pos - 1] if 1 <= pos <= len(wt_seq) else '?'}' != ref_aa '{ref_aa}'"
            )
        return NormalizedEdit(
            var_id=var_id, u=pos - 1, d=1, inserted_seq=alt_aa,
            edit_type="missense", wt_len=len(wt_seq), mut_len=len(wt_seq), op_raw="missense",
        )

    if cons == "synonymous":
        pp, ref_aa = row.get("pp"), row.get("ref_aa")
        if pp is None or ref_aa is None or _isnan(pp):
            raise UnsupportedVariantError(var_id, "synonymous row missing pp/ref_aa")
        ref_aa = str(ref_aa)
        if len(ref_aa) != 1:
            raise UnsupportedVariantError(var_id, "synonymous ref_aa is not single-letter")
        pos = int(pp)
        if not (1 <= pos <= len(wt_seq)) or wt_seq[pos - 1] != ref_aa:
            raise AlignmentValidationError(
                var_id, f"WT[{pos}]='{wt_seq[pos - 1] if 1 <= pos <= len(wt_seq) else '?'}' != ref_aa '{ref_aa}'"
            )
        return NormalizedEdit(
            var_id=var_id, u=pos - 1, d=1, inserted_seq=ref_aa,
            edit_type="synonymous", wt_len=len(wt_seq), mut_len=len(wt_seq), op_raw="synonymous",
        )

    # ---- inframe indels: reuse the existing HGVSp parser ----
    c = parse_protein_change(row.get("HGVSp"))
    if c is None:
        raise UnsupportedVariantError(var_id, f"HGVSp not parseable: {row.get('HGVSp')!r}")
    s, e, op, inserted = c["start"], c["end"], c["op"], c["inserted"]

    def _check_anchor(pos: Optional[int], expected_aa: Optional[str], label: str) -> None:
        if pos is None or expected_aa is None:
            return
        if not (1 <= pos <= len(wt_seq)) or wt_seq[pos - 1] != expected_aa:
            got = wt_seq[pos - 1] if 1 <= pos <= len(wt_seq) else "<out-of-range>"
            raise AlignmentValidationError(var_id, f"WT[{pos}] ({label}) = '{got}' != HGVSp '{expected_aa}'")

    _check_anchor(s, c["start_aa"], "start")
    _check_anchor(e, c["end_aa"], "end")

    if op == "del":
        u, d, ins = s - 1, e - s + 1, ""
    elif op == "dup":
        u, d, ins = e, 0, wt_seq[s - 1:e]
    elif op == "delins":
        if inserted is None:
            raise UnsupportedVariantError(var_id, "delins insert sequence not parseable")
        u, d, ins = s - 1, e - s + 1, inserted
    elif op == "ins":
        if inserted is None:
            raise UnsupportedVariantError(var_id, "ins insert sequence not parseable")
        u, d, ins = s, 0, inserted
    else:
        raise UnsupportedVariantError(var_id, f"unhandled HGVSp op '{op}'")

    edit_type = _classify_edit_type(u, d, ins, wt_seq)
    return NormalizedEdit(
        var_id=var_id, u=u, d=d, inserted_seq=ins, edit_type=edit_type,
        wt_len=len(wt_seq), mut_len=len(wt_seq) - d + len(ins), op_raw=op,
    )


def _isnan(x) -> bool:
    try:
        return x != x
    except TypeError:
        return False


def validate_reconstruction(wt_seq: str, mut_seq: str, edit: NormalizedEdit) -> None:
    """Raise AlignmentValidationError unless WT[:u]+inserted+WT[u+d:] == mut_seq."""
    rebuilt = wt_seq[: edit.u] + edit.inserted_seq + wt_seq[edit.u + edit.d:]
    if rebuilt != mut_seq:
        raise AlignmentValidationError(
            edit.var_id,
            f"reconstructed mutant (len {len(rebuilt)}) != manifest mut_seq (len {len(mut_seq)})",
        )
    if len(rebuilt) != edit.mut_len:
        raise AlignmentValidationError(
            edit.var_id, f"reconstructed length {len(rebuilt)} != edit.mut_len {edit.mut_len}"
        )


def build_global_alignment(edit: NormalizedEdit) -> list[AlignedSlot]:
    """Full-protein WT<->MUT slot list (see module docstring for the rule)."""
    slots: list[AlignedSlot] = []
    u, d, m = edit.u, edit.d, edit.m

    # prefix: WT[0,u) <-> MUT[0,u), 0-based -> 1-based positions [1, u]
    for i in range(u):
        slots.append(AlignedSlot(kind=SLOT_PAIRED, wt_pos=i + 1, mut_pos=i + 1, is_event=False))

    if edit.is_single_substitution:
        slots.append(AlignedSlot(kind=SLOT_PAIRED, wt_pos=u + 1, mut_pos=u + 1, is_event=True))
    else:
        for k in range(d):
            slots.append(AlignedSlot(kind=SLOT_WT_ONLY, wt_pos=u + k + 1, mut_pos=None, is_event=True))
        for k in range(m):
            slots.append(
                AlignedSlot(
                    kind=SLOT_MUT_ONLY, wt_pos=None, mut_pos=u + k + 1,
                    is_event=True, insertion_rank=k + 1,
                )
            )

    # suffix: WT[u+d, wt_len) <-> MUT[u+m, mut_len), 0-based j -> MUT j+(m-d)
    for j in range(u + d, edit.wt_len):
        slots.append(
            AlignedSlot(kind=SLOT_PAIRED, wt_pos=j + 1, mut_pos=j + (m - d) + 1, is_event=False)
        )
    return slots


def select_window(alignment: list[AlignedSlot], window_radius: int) -> list[AlignedSlot]:
    """Event slots plus up to `window_radius` paired flank slots each side.

    All slots strictly before/after the contiguous event block are paired
    (by construction of build_global_alignment), so a plain slice already
    clips correctly at either protein terminus.
    """
    event_idx = [i for i, s in enumerate(alignment) if s.is_event]
    if not event_idx:
        raise ValueError("alignment has no event slots -- not a valid edit")
    lo, hi = event_idx[0], event_idx[-1]
    left = alignment[max(0, lo - window_radius): lo]
    right = alignment[hi + 1: hi + 1 + window_radius]
    return left + alignment[lo: hi + 1] + right


def anchor_relative_coordinates(edit: NormalizedEdit, slots: list[AlignedSlot]) -> list[float]:
    """Fixed PE input coordinate per slot: WT position (0-based) minus u.

    A MUT-only slot has no WT position, so its coordinate is 0 by convention
    (it sits exactly at the insertion boundary anchor u -> 0 - relative to
    itself). This is OUR convention for this build (documented in
    README_STAGE1.md), not something already fixed by prior docs.
    """
    coords: list[float] = []
    for s in slots:
        if s.wt_pos is not None:
            coords.append(float((s.wt_pos - 1) - edit.u))
        else:
            coords.append(0.0)
    return coords


def wt_sequence_hash(wt_seq: str) -> str:
    return hashlib.sha256(wt_seq.encode("utf-8")).hexdigest()[:16]
