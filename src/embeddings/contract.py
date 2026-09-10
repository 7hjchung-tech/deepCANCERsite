"""Startup validation of `configs/esm_repr_v1.yaml` against the Python constants.

The YAML is a *runtime* config: the CLI reads its `paths`, `batching`, `window`
and `esm.device` entries as defaults. It also restates the scientific contract
(model, layers, precisions, window rule, split rule, alignment version, LLR
method), and those restatements are checked field-by-field against the constants
the code actually uses, before any model is loaded.

The point is that a documented contract cannot silently drift from runtime
behaviour: if someone edits the YAML window rule without editing
`representation_cache.WINDOW_RULE` (or vice versa), every subcommand refuses to
start and names the fields that disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from . import representation_cache as rc
from .likelihood import AA20, LLR_VERSION, SCORING_METHOD
from .variant_map import SLOT_KINDS, SPLIT_RULE, SPLIT_RULE_VERSION

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _ROOT / "configs" / "esm_repr_v1.yaml"


class ContractMismatchError(RuntimeError):
    """Raised when the YAML contract and the Python constants disagree."""


def _norm(value: Any) -> Any:
    """Collapse YAML folded-scalar whitespace so prose compares by content."""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return value


def _dig(cfg: Dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return "<missing>"
        node = node[part]
    return node


#: (yaml path, python constant). Everything here is part of the scientific
#: contract; a mismatch is a hard error, never a warning.
def _expectations() -> List[Tuple[str, Any]]:
    return [
        ("esm.backend", rc.BACKEND),
        ("esm.model_name", rc.MODEL_NAME),
        ("esm.repr_layers", list(rc.REPR_LAYERS)),
        ("esm.layer_convention", rc.LAYER_CONVENTION),
        ("esm.adapter", "none"),
        ("esm.eval_mode", True),
        ("esm.grad", False),
        ("precision.forward", "fp32"),
        ("precision.cache", "fp32"),
        ("precision.delta", "fp32"),
        ("precision.logits", "fp32"),
        ("precision.autocast", False),
        ("batching.equal_length_batches_only", True),
        ("batching.group_by_length", True),
        ("window.rule", rc.WINDOW_RULE),
        ("window.version", rc.WINDOW_RULE_VERSION),
        ("alignment.version", rc.ALIGNMENT_VERSION),
        ("alignment.slot_kinds", list(SLOT_KINDS)),
        ("alignment.arrays", list(rc.ARRAY_KEYS)),
        ("split.rule", SPLIT_RULE),
        ("split.version", SPLIT_RULE_VERSION),
        ("llr.scoring_method", SCORING_METHOD),
        ("llr.version", LLR_VERSION),
        ("llr.aa_order", "".join(AA20)),
        ("embedding.dim", rc.EMBED_DIM),
    ]


@dataclass
class Contract:
    path: Path
    cfg: Dict[str, Any]

    # --- runtime knobs the CLI actually uses -----------------------------
    @property
    def window_W(self) -> int:
        return int(_dig(self.cfg, "window.W"))

    @property
    def batch_size(self) -> int:
        return int(_dig(self.cfg, "batching.batch_size"))

    @property
    def device(self) -> str:
        return str(_dig(self.cfg, "esm.device"))

    def path_of(self, key: str) -> Path:
        return _ROOT / str(_dig(self.cfg, f"paths.{key}"))


def check_contract(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {yaml path: {yaml, python}} for every field that disagrees."""
    bad: Dict[str, Dict[str, Any]] = {}
    for dotted, expected in _expectations():
        found = _dig(cfg, dotted)
        if _norm(found) != _norm(expected):
            bad[dotted] = {"yaml": found, "python": expected}
    return bad


def load_contract(path: Path | str | None = None, *, strict: bool = True) -> Contract:
    """Load the YAML and refuse to continue if it contradicts the constants."""
    p = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    bad = check_contract(cfg)
    if bad and strict:
        lines = [f"{k}:\n    yaml   = {v['yaml']!r}\n    python = {v['python']!r}"
                 for k, v in sorted(bad.items())]
        raise ContractMismatchError(
            f"{p} disagrees with the Python contract constants in "
            f"{len(bad)} field(s):\n  " + "\n  ".join(lines)
        )
    return Contract(path=p, cfg=cfg)
