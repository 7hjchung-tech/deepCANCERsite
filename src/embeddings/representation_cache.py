"""Frozen ESM-2 650M representation cache (Task C).

WHAT THIS BUILDS
----------------
For every in-scope variant row we want, per repr layer, the residue
representations of the WT and MUT proteins *aligned to each other* over a
window around the edit:

    H_WT     [B, 3, A, 1280]
    H_MUT    [B, 3, A, 1280]
    delta_H  [B, 3, A, 1280]      layer order = (31, 32, 33)

A is the number of alignment slots (padded to a fixed width per cache).

HOW THE ALIGNMENT IS OBTAINED
-----------------------------
The per-slot coordinate contract is Task B's, unchanged:
`variant_map.build_variant_record()` returns the canonical arrays

    wt_pos, mut_pos, wt_present, mut_present, delta_valid, token_valid, slot_kind

for the *edit core*. Task C builds the analysis window from the same edit
GEOMETRY (`left_boundary_wt` / `right_boundary_wt` / `right_offset`, all from
`variant_map._edit_geometry`), and then checks that Task B's core reappears
verbatim as a contiguous run inside it -- two independent derivations that must
agree, rather than one padded copy of the other.

WINDOW RULE
-----------
* missense / synonymous : +-10 residues around the substituted position
                          -> up to 21 paired slots
* indel                 : the full edit event (every wt_only + every mut_only)
                          plus EXACTLY 10 paired flank residues on each side,
                          counted inward from the edit's own boundaries. The
                          boundary residues are part of those 10, never an extra
                          pair appended outside them:

                              1-residue deletion     10 + 1 + 10          = 21
                              1-residue duplication  10 + 1 mut_only + 10 = 21
                              Leu338_Lys342delinsGln 10 + 5 + 1 + 10      = 26

Slots that fall off either terminus are simply not emitted -- a window near a
terminus is legitimately shorter; the row is then padded to A with
`slot_kind="pad"`, `token_valid=False`.

RULES THIS MODULE ENFORCES
--------------------------
* WT and MUT are gathered through *separate* index arrays (wt_pos / mut_pos).
  There is no same-array-index subtraction anywhere: for a deletion the right
  boundary pairs WT e+1 with MUT e+1-n_del, and that is what gets subtracted.
* delta_H is non-zero only where delta_valid is True (paired slots). wt_only and
  mut_only slots keep their real residue representation on the side that exists
  and contribute delta 0 -- they are NOT masked out like padding.
* token_valid marks padding and nothing else. A gap (wt_only / mut_only) is a
  real, token_valid=True slot.
* synonymous rows reuse the *same* WT tensor object for both sides, so their
  valid delta is exactly 0.0, not merely small.
* delta_H is always computed in FP32.

STORAGE
-------
The WT protein is forwarded once per checkpoint and its full-length hidden
states are stored once (3 x L_wt x 1280). MUT hidden states are NOT stored in
full: only the windowed slice is kept, deduplicated by (mut sequence hash,
window signature), so N DNA rows sharing a protein edit share one tensor.
H_WT and delta_H are reconstructed at load time in FP32 (bit-exact: the cache
precision is FP32 too).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .variant_map import (
    SLOT_KINDS,
    SPLIT_RULE,
    SPLIT_RULE_VERSION,
    build_variant_record,
)

# --------------------------------------------------------------------------
# Contract constants. Any change here must bump the version string, because the
# version goes into the provenance hash that gates cache reuse.
# --------------------------------------------------------------------------
REPR_LAYERS: Tuple[int, ...] = (31, 32, 33)
EMBED_DIM = 1280
MODEL_NAME = "esm2_t33_650M_UR50D"
BACKEND = "repo_native_esm"

LAYER_CONVENTION = (
    "esm.model.esm2.ESM2.forward repr_layers indices: 0 = token embedding before "
    "the first transformer layer; i in 1..33 = output of layers[i-1]; index 33 is "
    "overwritten by emb_layer_norm_after, so layer 33 carries the final LayerNorm."
)
WINDOW_RULE = (
    "missense/synonymous: +-W residues around pp (2W+1 = 21 paired slots at W=10). "
    "indel: the full edit event (all wt_only + all mut_only) plus EXACTLY W paired "
    "flank residues on each side, counted inward from left_boundary_wt / "
    "right_boundary_wt -- the boundary residues are part of those W, not an extra "
    "pair outside them. Out-of-terminus slots are dropped (a terminal window is "
    "shorter) and the row is padded to A with slot_kind=pad."
)
WINDOW_RULE_VERSION = "window_v2_flank_from_edit_boundaries"
ALIGNMENT_VERSION = "taskB_canonical_v2"

SLOT_CODE = {name: i for i, name in enumerate(SLOT_KINDS)}

_ROOT = Path(__file__).resolve().parents[2]


class StaleCacheError(RuntimeError):
    """Raised when a cache on disk does not match the expected provenance."""


# ==========================================================================
# Hashing / provenance primitives
# ==========================================================================
def sequence_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_path() -> Path:
    """Resolve the torch.hub cache path of the repo-native 650M checkpoint."""
    return Path(torch.hub.get_dir()) / "checkpoints" / f"{MODEL_NAME}.pt"


def checkpoint_sha256(memo_dir: Optional[Path] = None) -> str:
    """sha256 of the base checkpoint file, memoised on (size, mtime)."""
    path = checkpoint_path()
    if not path.exists():
        return "missing"
    st = path.stat()
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    memo_file = (memo_dir or _ROOT / "data") / ".esm_ckpt_sha256.json"
    if memo_file.exists():
        try:
            memo = json.loads(memo_file.read_text())
            if memo.get("key") == key:
                return memo["sha256"]
        except (json.JSONDecodeError, KeyError):
            pass
    digest = file_sha256(path)
    memo_file.parent.mkdir(parents=True, exist_ok=True)
    memo_file.write_text(json.dumps({"key": key, "sha256": digest}, indent=2))
    return digest


def git_commit() -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD") or "unknown",
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def code_hash(paths: Sequence[str]) -> Dict[str, str]:
    out = {}
    for rel in paths:
        p = _ROOT / rel
        out[rel] = file_sha256(p)[:16] if p.exists() else "missing"
    return out


def build_provenance(
    *,
    manifest_path: Path,
    wt_seq: str,
    forward_precision: str,
    cache_precision: str,
    device: str,
    window_W: int,
    sequence_set_hash: str,
    n_rows: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The full provenance record. `provenance_hash()` of this gates cache reuse."""
    import pandas as pd

    manifest_hash = file_sha256(manifest_path)
    split_series = pd.read_csv(manifest_path, usecols=["var_id", "split"])
    split_schema_hash = hashlib.sha256(
        split_series.sort_values("var_id").to_csv(index=False).encode()
    ).hexdigest()

    prov: Dict[str, Any] = {
        "backend": BACKEND,
        "model_name": MODEL_NAME,
        "base_checkpoint_hash": checkpoint_sha256(),
        "adapter": "none",
        "adapter_state": "frozen",
        "repr_layers": list(REPR_LAYERS),
        "layer_convention": LAYER_CONVENTION,
        "forward_precision": forward_precision,
        "cache_precision": cache_precision,
        "delta_precision": "torch.float32",
        "delta_storage": "derived_on_load_fp32",
        "device": device,
        "wt_sequence_hash": sequence_hash(wt_seq),
        "wt_length": len(wt_seq),
        "sequence_set_hash": sequence_set_hash,
        "manifest_path": str(manifest_path.relative_to(_ROOT)),
        "manifest_hash": manifest_hash,
        "split_schema_hash": split_schema_hash,
        "alignment_version": ALIGNMENT_VERSION,
        "variant_map_hash": code_hash(["src/embeddings/variant_map.py"])[
            "src/embeddings/variant_map.py"
        ],
        "window_rule": WINDOW_RULE,
        "window_rule_version": WINDOW_RULE_VERSION,
        "window_W": window_W,
        "split_rule": SPLIT_RULE,
        "split_rule_version": SPLIT_RULE_VERSION,
        "n_rows": n_rows,
        "code_hash": code_hash(
            [
                "src/embeddings/representation_cache.py",
                "src/embeddings/esm_encoder.py",
                "src/embeddings/variant_map.py",
                "dataset.py",
            ]
        ),
        "git": git_commit(),
        "contains_targets": False,
        "target_columns_excluded": ["z_score_D4_D14", "functional_classification"],
    }
    if extra:
        prov.update(extra)
    prov["provenance_hash"] = provenance_hash(prov)
    return prov


#: Fields that must match for a cache to be reusable. Anything that changes the
#: numbers in the cache belongs here; timing/bookkeeping does not.
PROVENANCE_IDENTITY_FIELDS = (
    "backend",
    "model_name",
    "base_checkpoint_hash",
    "adapter",
    "adapter_state",
    "repr_layers",
    "layer_convention",
    "forward_precision",
    "cache_precision",
    "wt_sequence_hash",
    "sequence_set_hash",
    "manifest_hash",
    "split_schema_hash",
    "alignment_version",
    "variant_map_hash",
    "window_rule",
    "window_rule_version",
    "window_W",
    "split_rule",
    "split_rule_version",
    # likelihood.py reuses this identity set; these are None for repr caches.
    "scoring_method",
    "llr_version",
    "masked_context",
)


def provenance_hash(prov: Dict[str, Any]) -> str:
    payload = {k: prov.get(k) for k in PROVENANCE_IDENTITY_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def provenance_diff(expected: Dict[str, Any], found: Dict[str, Any]) -> Dict[str, Any]:
    diff = {}
    for k in PROVENANCE_IDENTITY_FIELDS:
        if expected.get(k) != found.get(k):
            diff[k] = {"expected": expected.get(k), "found": found.get(k)}
    return diff


# ==========================================================================
# Window alignment: Task B core -> analysis window
# ==========================================================================
def _paired_slot(wt_pos: int, mut_pos: int) -> Dict[str, Any]:
    return {
        "wt_pos": wt_pos,
        "mut_pos": mut_pos,
        "wt_present": 1,
        "mut_present": 1,
        "delta_valid": True,
        "token_valid": True,
        "slot_kind": "paired",
    }


ARRAY_KEYS = (
    "wt_pos",
    "mut_pos",
    "wt_present",
    "mut_present",
    "delta_valid",
    "token_valid",
    "slot_kind",
)


def _as_slots(arrays: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = len(arrays["slot_kind"])
    return [{k: arrays[k][i] for k in ARRAY_KEYS} for i in range(n)]


def _as_arrays(slots: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {k: [s[k] for s in slots] for k in ARRAY_KEYS}


def _wt_only_slot(wt_pos: int) -> Dict[str, Any]:
    return {
        "wt_pos": wt_pos,
        "mut_pos": 0,
        "wt_present": 1,
        "mut_present": 0,
        "delta_valid": False,
        "token_valid": True,
        "slot_kind": "wt_only",
    }


def _mut_only_slot(mut_pos: int) -> Dict[str, Any]:
    return {
        "wt_pos": 0,
        "mut_pos": mut_pos,
        "wt_present": 0,
        "mut_present": 1,
        "delta_valid": False,
        "token_valid": True,
        "slot_kind": "mut_only",
    }


def build_window_alignment(
    rec: Dict[str, Any], wt_len: int, mut_len: int, W: int = 10
) -> Dict[str, List[Any]]:
    """Build the analysis window from the EDIT GEOMETRY, not by padding a core.

    missense / synonymous
        WT pp-W .. pp+W, every slot paired -> 2W+1 = 21 slots at W=10.

    indel
        exactly W paired residues on each side of the edit event, counted
        INWARD from the event's own boundaries, plus the event itself:

            [ W paired ..= left_boundary_wt ] [ wt_only* ] [ mut_only* ]
            [ right_boundary_wt =.. W paired ]

        The boundary residues are part of those W, never an extra pair on top of
        them. At W=10 that is:

            1-residue deletion        10 + 1 + 10                      = 21
            1-residue duplication     10 + 1 mut_only + 10             = 21
            Leu338_Lys342delinsGln    10 + 5 wt_only + 1 mut_only + 10 = 26

    Terminal slots that fall outside 1..wt_len (or whose MUT partner falls
    outside 1..mut_len) are simply not emitted, so a window near a terminus is
    shorter; the row is padded to A afterwards with slot_kind="pad".

    MUT coordinates come from the geometry too: 0 offset left of the event,
    `right_offset = n_ins - n_del` at and right of it. Task B's canonical edit
    core is reproduced exactly as a contiguous run inside the window -- checked
    by `core_offset_in_window()` rather than assumed.
    """
    cons = str(rec.get("slim_consequence", "")).strip()

    if cons in {"missense", "synonymous"}:
        pp = rec.get("pp")
        if pp is None:
            return _as_arrays(_as_slots({k: rec[k] for k in ARRAY_KEYS}))
        pp = int(pp)
        return _as_arrays([
            _paired_slot(p, p)
            for p in range(pp - W, pp + W + 1)
            if 1 <= p <= wt_len and 1 <= p <= mut_len
        ])

    left_b = rec.get("left_boundary_wt")
    right_b = rec.get("right_boundary_wt")
    if left_b is None or right_b is None:
        # unsupported / unparsable rows carry no edit geometry: no window.
        return _as_arrays(_as_slots({k: rec[k] for k in ARRAY_KEYS}))

    off = int(rec.get("right_offset", 0))
    slots: List[Dict[str, Any]] = []

    for p in range(int(left_b) - W + 1, int(left_b) + 1):
        if 1 <= p <= wt_len and 1 <= p <= mut_len:
            slots.append(_paired_slot(p, p))

    for p in rec.get("wt_only_positions", []):
        if 1 <= int(p) <= wt_len:
            slots.append(_wt_only_slot(int(p)))

    for m in rec.get("mut_only_positions", []):
        if 1 <= int(m) <= mut_len:
            slots.append(_mut_only_slot(int(m)))

    for p in range(int(right_b), int(right_b) + W):
        m = p + off
        if 1 <= p <= wt_len and 1 <= m <= mut_len:
            slots.append(_paired_slot(p, m))

    return _as_arrays(slots)


def core_offset_in_window(
    core: Dict[str, List[Any]], window: Dict[str, List[Any]]
) -> int:
    """Index at which Task B's edit core sits inside the window, or -1.

    The window is derived from the edit geometry independently of Task B's slot
    builder, so this is a real cross-check of two derivations, not a tautology:
    every core slot must reappear, in order, contiguously, with identical
    coordinates and kinds.
    """
    n, m = len(core["slot_kind"]), len(window["slot_kind"])
    if n == 0 or n > m:
        return -1
    for i in range(m - n + 1):
        if all(
            list(window[k][i : i + n]) == list(core[k])
            for k in ("wt_pos", "mut_pos", "slot_kind")
        ):
            return i
    return -1


def window_signature(arrays: Dict[str, List[Any]]) -> str:
    """Stable id for one window layout (used together with the MUT sequence hash
    to decide whether two rows can share a stored MUT window tensor)."""
    payload = json.dumps(
        {k: list(map(_jsonable, arrays[k])) for k in ARRAY_KEYS}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _jsonable(v: Any) -> Any:
    return bool(v) if isinstance(v, bool) else v


def pad_arrays(arrays: Dict[str, List[Any]], A: int) -> Dict[str, List[Any]]:
    n = len(arrays["slot_kind"])
    if n > A:
        raise ValueError(f"window has {n} slots but cache width A={A}")
    pad = A - n
    return {
        "wt_pos": list(arrays["wt_pos"]) + [0] * pad,
        "mut_pos": list(arrays["mut_pos"]) + [0] * pad,
        "wt_present": [bool(v) for v in arrays["wt_present"]] + [False] * pad,
        "mut_present": [bool(v) for v in arrays["mut_present"]] + [False] * pad,
        "delta_valid": [bool(v) for v in arrays["delta_valid"]] + [False] * pad,
        "token_valid": [bool(v) for v in arrays["token_valid"]] + [False] * pad,
        "slot_kind": list(arrays["slot_kind"]) + ["pad"] * pad,
    }


# ==========================================================================
# Gather
# ==========================================================================
def stack_layers(reps: Dict[int, torch.Tensor], layers: Sequence[int] = REPR_LAYERS) -> torch.Tensor:
    """{layer: [L,1280]} -> [n_layers, L, 1280] fp32 on CPU, in `layers` order."""
    return torch.stack([reps[l].detach().to(torch.float32).cpu() for l in layers], dim=0)


def gather_window(
    full: torch.Tensor, positions: Sequence[int], present: Sequence[bool]
) -> torch.Tensor:
    """Gather [n_layers, A, 1280] from a full-sequence [n_layers, L, 1280].

    `positions` are 1-based residue coordinates; absent slots (present=False)
    contribute exact zeros and their position value is ignored.
    """
    idx = torch.tensor([max(int(p) - 1, 0) for p in positions], dtype=torch.long)
    mask = torch.tensor([bool(v) for v in present], dtype=torch.bool)
    out = full.index_select(dim=1, index=idx)          # [n_layers, A, 1280]
    return out * mask.view(1, -1, 1).to(out.dtype)


def compute_delta(
    h_wt: torch.Tensor, h_mut: torch.Tensor, delta_valid: torch.Tensor
) -> torch.Tensor:
    """delta_H in FP32, exactly zero wherever delta_valid is False.

    Shapes: h_wt / h_mut are [..., n_layers, A, 1280] and delta_valid is
    [..., A] -- the layer axis is broadcast, so one mask covers all layers.

    Both inputs were gathered through their own coordinate arrays (wt_pos /
    mut_pos), so this subtracts *aligned residues*, never equal array indices of
    two differently-framed sequences.
    """
    d = h_mut.to(torch.float32) - h_wt.to(torch.float32)
    mask = delta_valid.to(torch.float32).unsqueeze(-2).unsqueeze(-1)  # [...,1,A,1]
    return d * mask


# ==========================================================================
# Sequence-level forward cache
# ==========================================================================
@dataclass
class ForwardStats:
    """Three distinct quantities that must never be conflated.

    `unique_sequences_encoded` counts distinct protein sequences handed to the
    model. `model_forward_batch_calls` counts actual `model(tokens)` invocations,
    read straight off the encoder's own counter -- with batch_size 8 those differ
    by roughly a factor of 8. `cache_hits` counts requests served from the store
    without any forward at all.
    """

    n_requested: int = 0
    unique_sequences_encoded: int = 0
    model_forward_batch_calls: int = 0
    cache_hits: int = 0
    forward_seconds: float = 0.0
    lengths: Dict[int, int] = field(default_factory=dict)


class SequenceForwardCache:
    """Forward each distinct (sequence, checkpoint) exactly once.

    Keyed on sha256(sequence); the checkpoint identity is fixed for the lifetime
    of the encoder that backs this cache, and is recorded in the provenance that
    gates reuse of anything written to disk.
    """

    def __init__(self, encoder, layers: Sequence[int] = REPR_LAYERS, batch_size: int = 8):
        self.encoder = encoder
        self.layers = list(layers)
        self.batch_size = batch_size
        self._store: Dict[str, torch.Tensor] = {}
        self.stats = ForwardStats()

    def __contains__(self, seq: str) -> bool:
        return sequence_hash(seq) in self._store

    def get(self, seq: str) -> torch.Tensor:
        """[n_layers, L, 1280] fp32 CPU for one sequence."""
        self.ensure([seq])
        return self._store[sequence_hash(seq)]

    def ensure(self, sequences: Iterable[str]) -> None:
        """Forward every not-yet-cached sequence, batched by residue length."""
        wanted: Dict[str, str] = {}
        for seq in sequences:
            self.stats.n_requested += 1
            h = sequence_hash(seq)
            if h in self._store:
                self.stats.cache_hits += 1
            elif h not in wanted:
                wanted[h] = seq
            else:
                self.stats.cache_hits += 1
        if not wanted:
            return

        seqs = list(wanted.values())
        calls_before = getattr(self.encoder, "forward_calls", 0)
        t0 = time.perf_counter()
        per_seq = self.encoder.encode_by_length(
            seqs, repr_layers=self.layers, batch_size=self.batch_size
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stats.forward_seconds += time.perf_counter() - t0
        self.stats.model_forward_batch_calls += (
            getattr(self.encoder, "forward_calls", 0) - calls_before
        )

        for seq, reps in zip(seqs, per_seq):
            self._store[sequence_hash(seq)] = stack_layers(reps, self.layers)
            self.stats.unique_sequences_encoded += 1
            self.stats.lengths[len(seq)] = self.stats.lengths.get(len(seq), 0) + 1

    @property
    def n_unique(self) -> int:
        return len(self._store)


# ==========================================================================
# Export
# ==========================================================================
@dataclass
class RowPlan:
    """Everything about one manifest row that the export needs, labels excluded."""
    var_id: str
    split: str
    variant_type: str
    edit_type: str
    slim_consequence: str
    pp: Optional[int]
    wt_seq: str
    mut_seq: str
    arrays: Dict[str, List[Any]]
    n_slots: int
    mut_seq_hash: str
    window_sig: str
    is_synonymous: bool


def plan_rows(
    manifest_df,
    wt_seq: str,
    *,
    W: int = 10,
    in_scope_only: bool = True,
) -> Tuple[List[RowPlan], List[Dict[str, Any]]]:
    """Build the per-row window plan. Returns (plans, skipped)."""
    plans: List[RowPlan] = []
    skipped: List[Dict[str, Any]] = []

    for _, raw in manifest_df.iterrows():
        row = raw.to_dict()
        rec = build_variant_record(row, wt_seq, manifest_df=manifest_df)
        if in_scope_only and not rec["in_eval_scope"]:
            skipped.append(
                {
                    "var_id": rec["var_id"],
                    "reason": rec["exclusion_reason"] or "not_in_eval_scope",
                    "variant_type": rec["variant_type"],
                    "edit_type": rec["edit_type"],
                }
            )
            continue

        mut_seq = str(row.get("mut_seq") or "")
        if not mut_seq:
            from dataset import build_mutant_sequence

            built, ok, _ = build_mutant_sequence(wt_seq, row)
            if not ok:
                skipped.append({"var_id": rec["var_id"], "reason": "mut_seq_unbuildable"})
                continue
            mut_seq = built

        arrays = build_window_alignment(rec, len(wt_seq), len(mut_seq), W=W)
        if not arrays["slot_kind"]:
            skipped.append({"var_id": rec["var_id"], "reason": "empty_window"})
            continue

        bad = _validate_arrays(arrays, len(wt_seq), len(mut_seq))
        if bad:
            skipped.append({"var_id": rec["var_id"], "reason": f"invalid_window:{bad}"})
            continue

        # The window came from the edit geometry; Task B's core came from the
        # slot builder. They must agree, or one of the two is wrong about what
        # the edit is -- refuse the row rather than cache a guess.
        core = {k: rec[k] for k in ARRAY_KEYS}
        if core_offset_in_window(core, arrays) < 0:
            skipped.append(
                {"var_id": rec["var_id"], "reason": "task_b_core_not_contiguous_in_window"}
            )
            continue

        is_syn = rec["slim_consequence"] == "synonymous"
        plans.append(
            RowPlan(
                var_id=str(rec["var_id"]),
                split=str(rec["split"]),
                variant_type=rec["variant_type"],
                edit_type=rec["edit_type"],
                slim_consequence=rec["slim_consequence"],
                pp=rec["pp"],
                wt_seq=wt_seq,
                mut_seq=mut_seq,
                arrays=arrays,
                n_slots=len(arrays["slot_kind"]),
                mut_seq_hash=sequence_hash(mut_seq),
                window_sig=window_signature(arrays),
                is_synonymous=is_syn,
            )
        )
    return plans, skipped


def _validate_arrays(arrays: Dict[str, List[Any]], wt_len: int, mut_len: int) -> str:
    n = len(arrays["slot_kind"])
    for k in ARRAY_KEYS:
        if len(arrays[k]) != n:
            return f"ragged_{k}"
    for i in range(n):
        kind = arrays["slot_kind"][i]
        wp, mp = arrays["wt_pos"][i], arrays["mut_pos"][i]
        wpr, mpr = bool(arrays["wt_present"][i]), bool(arrays["mut_present"][i])
        dv = bool(arrays["delta_valid"][i])
        if wpr and not (1 <= wp <= wt_len):
            return f"wt_pos_out_of_range@{i}"
        if mpr and not (1 <= mp <= mut_len):
            return f"mut_pos_out_of_range@{i}"
        if dv and not (wpr and mpr):
            return f"delta_valid_without_both_sides@{i}"
        if dv and kind != "paired":
            return f"delta_valid_on_{kind}@{i}"
        if kind == "wt_only" and (mpr or dv):
            return f"wt_only_has_mut_side@{i}"
        if kind == "mut_only" and (wpr or dv):
            return f"mut_only_has_wt_side@{i}"
    return ""


@dataclass
class ExportResult:
    path: Path
    provenance: Dict[str, Any]
    report: Dict[str, Any]


def export_representations(
    plans: List[RowPlan],
    wt_seq: str,
    encoder,
    out_path: Path,
    *,
    manifest_path: Path,
    W: int = 10,
    batch_size: int = 8,
    device: str = "cpu",
    A: Optional[int] = None,
    progress_every: int = 0,
) -> ExportResult:
    """Run the frozen export and write one .pt cache + provenance sidecar."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    A = A or max(p.n_slots for p in plans)
    fcache = SequenceForwardCache(encoder, REPR_LAYERS, batch_size=batch_size)

    # WT once per checkpoint, before anything else, so every row shares it.
    t_fwd0 = time.perf_counter()
    wt_full = fcache.get(wt_seq)                                # [3, L_wt, 1280]

    # Every distinct MUT protein, deduplicated across DNA rows.
    unique_mut = list({p.mut_seq_hash: p.mut_seq for p in plans}.values())
    fcache.ensure(unique_mut)
    forward_seconds = time.perf_counter() - t_fwd0

    # Deduplicated MUT window tensors, keyed by (mut sequence, window layout).
    t_align0 = time.perf_counter()
    window_index: Dict[Tuple[str, str], int] = {}
    mut_windows: List[torch.Tensor] = []
    row_to_window: List[int] = []

    wt_pos_all, mut_pos_all = [], []
    wp_all, mp_all, dv_all, tv_all, sk_all = [], [], [], [], []

    for i, plan in enumerate(plans):
        padded = pad_arrays(plan.arrays, A)
        key = (plan.mut_seq_hash, plan.window_sig)
        if key not in window_index:
            # Synonymous rows reuse the WT tensor object itself: MUT protein ==
            # WT protein, same sequence hash, so this is the same cache entry.
            mut_full = fcache.get(plan.mut_seq)
            window_index[key] = len(mut_windows)
            mut_windows.append(
                gather_window(mut_full, padded["mut_pos"], padded["mut_present"])
            )
        row_to_window.append(window_index[key])

        wt_pos_all.append(padded["wt_pos"])
        mut_pos_all.append(padded["mut_pos"])
        wp_all.append(padded["wt_present"])
        mp_all.append(padded["mut_present"])
        dv_all.append(padded["delta_valid"])
        tv_all.append(padded["token_valid"])
        sk_all.append([SLOT_CODE[k] for k in padded["slot_kind"]])

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  aligned {i + 1}/{len(plans)} rows", flush=True)
    align_seconds = time.perf_counter() - t_align0

    sequence_set_hash = hashlib.sha256(
        json.dumps(sorted({sequence_hash(wt_seq), *(p.mut_seq_hash for p in plans)})).encode()
    ).hexdigest()

    forward_precision = str(next(encoder.model.parameters()).dtype)
    prov = build_provenance(
        manifest_path=manifest_path,
        wt_seq=wt_seq,
        forward_precision=forward_precision,
        cache_precision="torch.float32",
        device=device,
        window_W=W,
        sequence_set_hash=sequence_set_hash,
        n_rows=len(plans),
        extra={"A": A, "batch_size": batch_size},
    )

    payload = {
        "format_version": 1,
        "provenance": prov,
        "layers": list(REPR_LAYERS),
        "A": A,
        "slot_kind_vocab": list(SLOT_KINDS),
        "wt_full": wt_full,                                     # [3, L_wt, 1280]
        "mut_windows": torch.stack(mut_windows, dim=0),         # [U, 3, A, 1280]
        "row_to_mut_window": torch.tensor(row_to_window, dtype=torch.long),
        "wt_pos": torch.tensor(wt_pos_all, dtype=torch.long),
        "mut_pos": torch.tensor(mut_pos_all, dtype=torch.long),
        "wt_present": torch.tensor(wp_all, dtype=torch.bool),
        "mut_present": torch.tensor(mp_all, dtype=torch.bool),
        "delta_valid": torch.tensor(dv_all, dtype=torch.bool),
        "token_valid": torch.tensor(tv_all, dtype=torch.bool),
        "slot_kind": torch.tensor(sk_all, dtype=torch.long),
        "var_id": [p.var_id for p in plans],
        "split": [p.split for p in plans],
        "variant_type": [p.variant_type for p in plans],
        "edit_type": [p.edit_type for p in plans],
        "pp": [p.pp for p in plans],
        "wt_seq_hash": [sequence_hash(p.wt_seq) for p in plans],
        "mut_seq_hash": [p.mut_seq_hash for p in plans],
    }

    t_ser0 = time.perf_counter()
    torch.save(payload, out_path)
    serialize_seconds = time.perf_counter() - t_ser0

    out_path.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2))

    report = {
        "rows": len(plans),
        "A": A,
        "unique_wt_sequences": 1,
        "unique_mut_sequences": len(unique_mut),
        "unique_stored_mut_windows": len(mut_windows),
        # These three are different quantities; see ForwardStats.
        "unique_sequences_encoded": fcache.stats.unique_sequences_encoded,
        "model_forward_batch_calls": fcache.stats.model_forward_batch_calls,
        "cache_hits": fcache.stats.cache_hits,
        "sequence_requests": fcache.stats.n_requested,
        "forward_seconds": forward_seconds,
        "alignment_seconds": align_seconds,
        "serialize_seconds": serialize_seconds,
        "cache_bytes": out_path.stat().st_size,
        "forward_lengths": fcache.stats.lengths,
    }
    return ExportResult(path=out_path, provenance=prov, report=report)


# ==========================================================================
# Load
# ==========================================================================
class RepresentationCache:
    """Read side of the cache. Materialises H_WT / H_MUT / delta_H on demand."""

    def __init__(self, payload: Dict[str, Any]):
        self.p = payload
        self.provenance = payload["provenance"]
        self.layers = list(payload["layers"])
        self.A = int(payload["A"])
        self.slot_kind_vocab = list(payload["slot_kind_vocab"])

    def __len__(self) -> int:
        return len(self.p["var_id"])

    @property
    def var_id(self) -> List[str]:
        return self.p["var_id"]

    @property
    def split(self) -> List[str]:
        return self.p["split"]

    def slot_kind_names(self, i: int) -> List[str]:
        return [self.slot_kind_vocab[c] for c in self.p["slot_kind"][i].tolist()]

    def batch(self, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        """H_WT / H_MUT / delta_H [B,3,A,1280] plus the per-slot masks."""
        idx = torch.tensor(list(indices), dtype=torch.long)
        wt_pos = self.p["wt_pos"].index_select(0, idx)
        wt_present = self.p["wt_present"].index_select(0, idx)
        delta_valid = self.p["delta_valid"].index_select(0, idx)

        h_wt = torch.stack(
            [
                gather_window(self.p["wt_full"], wt_pos[b].tolist(), wt_present[b].tolist())
                for b in range(len(idx))
            ],
            dim=0,
        )
        h_mut = self.p["mut_windows"].index_select(
            0, self.p["row_to_mut_window"].index_select(0, idx)
        )
        delta = compute_delta(h_wt, h_mut, delta_valid)

        return {
            "H_WT": h_wt,
            "H_MUT": h_mut,
            "delta_H": delta,
            "wt_pos": wt_pos,
            "mut_pos": self.p["mut_pos"].index_select(0, idx),
            "wt_present": wt_present,
            "mut_present": self.p["mut_present"].index_select(0, idx),
            "delta_valid": delta_valid,
            "token_valid": self.p["token_valid"].index_select(0, idx),
            "slot_kind": self.p["slot_kind"].index_select(0, idx),
        }

    def row(self, i: int) -> Dict[str, torch.Tensor]:
        b = self.batch([i])
        return {k: v[0] for k, v in b.items()}


def load_cache(
    path: Path, expected_provenance: Optional[Dict[str, Any]] = None, strict: bool = True
) -> RepresentationCache:
    """Load a cache, refusing one whose provenance does not match.

    A cache written under a different checkpoint, precision, alignment version,
    window rule, manifest or split schema is NOT silently reused: with
    strict=True (the default) this raises StaleCacheError naming every field
    that differs.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    found = payload["provenance"]
    if expected_provenance is not None:
        if provenance_hash(found) != provenance_hash(expected_provenance):
            diff = provenance_diff(expected_provenance, found)
            msg = (
                f"stale representation cache at {path}: provenance mismatch in "
                f"{sorted(diff)}\n" + json.dumps(diff, indent=2, default=str)
            )
            if strict:
                raise StaleCacheError(msg)
            print("WARNING: " + msg)
    return RepresentationCache(payload)
