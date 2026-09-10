"""The Stage 1 input contract.

A Stage 1 model consumes exactly this, per batch of B variants:

    H_WT         [B, n_layers, A, D]   float32   WT residue representations
    H_MUT        [B, n_layers, A, D]   float32   MUT residue representations
    delta_H      [B, n_layers, A, D]   float32   H_MUT - H_WT, 0 off paired slots

    wt_pos       [B, A]  int64   1-based WT residue coordinate per slot (0 = absent)
    mut_pos      [B, A]  int64   1-based MUT residue coordinate per slot (0 = absent)
    wt_present   [B, A]  bool    the slot has a WT residue
    mut_present  [B, A]  bool    the slot has a MUT residue
    delta_valid  [B, A]  bool    delta_H is meaningful here (paired slots only)
    token_valid  [B, A]  bool    the slot is a real residue slot, NOT padding
    slot_kind    [B, A]  int64   index into slot_kind_vocab

with `n_layers = 3` (layers 31, 32, 33) and `D = 1280` for the Task C cache.

THE MASK CONTRACT -- the part that is easy to get wrong
-------------------------------------------------------
`token_valid` marks PADDING and nothing else. An indel gap (`wt_only` /
`mut_only`) is a real slot: `token_valid=True`, one side present, the other
side's tensor exactly zero, and `delta_valid=False`. A model must therefore

    * pool over `token_valid`   -- gaps participate, padding does not;
    * subtract/compare over `delta_valid` -- gaps do not participate;
    * never use `delta_valid` as its padding mask, and never treat a gap as pad.

TARGETS ARE NOT IN HERE
-----------------------
`Stage1Batch` has no target field, by construction. The loader returns targets
as a separate `TargetBatch`, so a Stage 1 model physically cannot read a label
from its input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch

#: Tensors that carry representations, shaped [B, n_layers, A, D].
REPR_FIELDS: Tuple[str, ...] = ("H_WT", "H_MUT", "delta_H")

#: Per-slot coordinate/mask arrays, shaped [B, A].
SLOT_FIELDS: Tuple[str, ...] = (
    "wt_pos",
    "mut_pos",
    "wt_present",
    "mut_present",
    "delta_valid",
    "token_valid",
    "slot_kind",
)

#: Per-row identifiers carried alongside the tensors. None of these is a label.
ID_FIELDS: Tuple[str, ...] = ("var_id", "split", "variant_type", "edit_type", "pp")

#: Column names that must never appear anywhere in a Stage 1 feature batch.
FORBIDDEN_LABEL_FIELDS: Tuple[str, ...] = (
    "z_score_D4_D14",
    "functional_classification",
    "y",
    "target",
    "label",
)


class ContractViolation(ValueError):
    """A batch does not satisfy the Stage 1 input contract."""


@dataclass
class Stage1Batch:
    """One batch of Stage 1 inputs. Feature-only: it holds no target."""

    H_WT: torch.Tensor
    H_MUT: torch.Tensor
    delta_H: torch.Tensor

    wt_pos: torch.Tensor
    mut_pos: torch.Tensor
    wt_present: torch.Tensor
    mut_present: torch.Tensor
    delta_valid: torch.Tensor
    token_valid: torch.Tensor
    slot_kind: torch.Tensor

    var_id: List[str]
    split: List[str]
    variant_type: List[str]
    edit_type: List[str]
    pp: List[Optional[int]]

    slot_kind_vocab: List[str]
    layers: List[int]

    @property
    def batch_size(self) -> int:
        return self.H_WT.shape[0]

    @property
    def n_layers(self) -> int:
        return self.H_WT.shape[1]

    @property
    def A(self) -> int:
        return self.H_WT.shape[2]

    @property
    def embed_dim(self) -> int:
        return self.H_WT.shape[3]

    def slot_kind_names(self, b: int) -> List[str]:
        return [self.slot_kind_vocab[c] for c in self.slot_kind[b].tolist()]

    def kind_mask(self, name: str) -> torch.Tensor:
        """[B, A] bool mask selecting one slot kind, e.g. "wt_only"."""
        if name not in self.slot_kind_vocab:
            raise KeyError(f"unknown slot kind {name!r}; have {self.slot_kind_vocab}")
        return self.slot_kind == self.slot_kind_vocab.index(name)

    def to(self, device: Any) -> "Stage1Batch":
        moved = {
            f: getattr(self, f).to(device) for f in REPR_FIELDS + SLOT_FIELDS
        }
        return Stage1Batch(
            **moved,
            var_id=list(self.var_id),
            split=list(self.split),
            variant_type=list(self.variant_type),
            edit_type=list(self.edit_type),
            pp=list(self.pp),
            slot_kind_vocab=list(self.slot_kind_vocab),
            layers=list(self.layers),
        )

    # ------------------------------------------------------------------
    def validate(self, *, check_finite: bool = True) -> None:
        """Assert the full contract. Raises ContractViolation on any breach."""
        B = self.H_WT.shape[0]
        if self.H_WT.ndim != 4:
            raise ContractViolation(f"H_WT must be [B,L,A,D], got {tuple(self.H_WT.shape)}")
        shape = tuple(self.H_WT.shape)
        A = shape[2]

        for f in REPR_FIELDS:
            t = getattr(self, f)
            if tuple(t.shape) != shape:
                raise ContractViolation(f"{f} has shape {tuple(t.shape)}, expected {shape}")
            if t.dtype is not torch.float32:
                raise ContractViolation(f"{f} has dtype {t.dtype}, expected torch.float32")
            if check_finite and not torch.isfinite(t).all():
                raise ContractViolation(f"{f} contains non-finite values")

        if shape[1] != len(self.layers):
            raise ContractViolation(
                f"layer axis is {shape[1]} but layers={self.layers}"
            )

        for f in SLOT_FIELDS:
            t = getattr(self, f)
            if tuple(t.shape) != (B, A):
                raise ContractViolation(f"{f} has shape {tuple(t.shape)}, expected {(B, A)}")
        for f in ("wt_present", "mut_present", "delta_valid", "token_valid"):
            if getattr(self, f).dtype is not torch.bool:
                raise ContractViolation(f"{f} must be bool, got {getattr(self, f).dtype}")
        for f in ("wt_pos", "mut_pos", "slot_kind"):
            if getattr(self, f).dtype is not torch.long:
                raise ContractViolation(f"{f} must be int64, got {getattr(self, f).dtype}")

        for f in ID_FIELDS:
            v = getattr(self, f)
            if len(v) != B:
                raise ContractViolation(f"{f} has {len(v)} entries, expected {B}")

        # --- mask semantics ------------------------------------------------
        pad = self.slot_kind == self.slot_kind_vocab.index("pad")
        if bool((pad & self.token_valid).any()):
            raise ContractViolation("a pad slot is marked token_valid")
        if bool((~pad & ~self.token_valid).any()):
            raise ContractViolation("a non-pad slot is marked token_invalid")

        for name in ("wt_only", "mut_only"):
            gap = self.kind_mask(name)
            if bool((gap & ~self.token_valid).any()):
                raise ContractViolation(f"a {name} gap slot is marked token_invalid "
                                        f"-- gaps are NOT padding")
            if bool((gap & self.delta_valid).any()):
                raise ContractViolation(f"a {name} gap slot is marked delta_valid")

        if bool((self.delta_valid & ~(self.wt_present & self.mut_present)).any()):
            raise ContractViolation("delta_valid where a side is absent")

        off = ~self.delta_valid                                  # [B, A]
        if off.any():
            masked = self.delta_H * off.unsqueeze(1).unsqueeze(-1)
            if float(masked.abs().max()) != 0.0:
                raise ContractViolation("delta_H is non-zero on a slot that is not delta_valid")

        if pad.any():
            for f in REPR_FIELDS:
                masked = getattr(self, f) * pad.unsqueeze(1).unsqueeze(-1)
                if float(masked.abs().max()) != 0.0:
                    raise ContractViolation(f"{f} is non-zero on a padding slot")

    # ------------------------------------------------------------------
    def assert_no_labels(self) -> None:
        """Structural check that nothing label-shaped rode along."""
        for f in FORBIDDEN_LABEL_FIELDS:
            if hasattr(self, f):
                raise ContractViolation(f"label field {f!r} present in a feature batch")


@dataclass
class TargetBatch:
    """Targets for one batch, deliberately a SEPARATE object from Stage1Batch."""

    var_id: List[str]
    y_raw: torch.Tensor       # [B] float32, original z-score units
    y_std: Optional[torch.Tensor] = None   # [B] float32, standardised (train scaler)
    target_column: str = ""
    scaler: Optional[Dict[str, Any]] = field(default=None)

    def to(self, device: Any) -> "TargetBatch":
        return TargetBatch(
            var_id=list(self.var_id),
            y_raw=self.y_raw.to(device),
            y_std=None if self.y_std is None else self.y_std.to(device),
            target_column=self.target_column,
            scaler=self.scaler,
        )


class Stage1Model(Protocol):
    """What a token-level Stage 1 must implement to be connectable here.

    One method. It takes the feature batch and returns one scalar per variant,
    in STANDARDISED target units (see targets.py) -- the optimisation space.
    """

    def forward(self, batch: Stage1Batch) -> torch.Tensor:  # [B]
        ...


def check_prediction(pred: torch.Tensor, batch: Stage1Batch) -> None:
    """Validate a Stage 1 output against the batch that produced it."""
    if pred.ndim != 1 or pred.shape[0] != batch.batch_size:
        raise ContractViolation(
            f"Stage 1 must return [B]={batch.batch_size}, got {tuple(pred.shape)}"
        )
    if not torch.isfinite(pred).all():
        raise ContractViolation("Stage 1 returned non-finite predictions")
