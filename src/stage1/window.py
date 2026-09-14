"""Per-sample window token construction: alignment + raw cache -> raw tensors.

This is the "aligned slot" tensor schema from the task spec, produced BEFORE
any trainable projection touches it (only frozen ESM outputs + structural
bookkeeping). Shapes are per-sample; Stage1Dataset.collate pads them to a
common A across the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .alignment import (
    AlignedSlot,
    NormalizedEdit,
    anchor_relative_coordinates,
    build_global_alignment,
    select_window,
)
from .cache import RawStage1Cache


@dataclass
class SampleTokens:
    var_id: str
    n_slots: int                 # A (before batch padding)
    layers: list[int]
    H_wt: Tensor                 # (Lyr, A, D)
    H_mut: Tensor                # (Lyr, A, D)
    delta: Tensor                # (Lyr, A, D)
    wt_present: Tensor           # (A,)
    mut_present: Tensor          # (A,)
    delta_valid: Tensor          # (A,)
    token_valid: Tensor          # (A,) -- all 1 here; padding is added at collate time
    slot_kind: Tensor            # (A,) long
    wt_pos: Tensor                # (A,) long, 1-based, sentinel 0
    mut_pos: Tensor               # (A,) long, 1-based, sentinel 0
    anchor_rel_coord: Tensor      # (A,) float
    insertion_rank: Tensor        # (A,) float


def build_sample_tokens(
    var_id: str,
    edit: NormalizedEdit,
    cache: RawStage1Cache,
    window_radius: int,
    layers: list[int],
) -> SampleTokens:
    alignment = build_global_alignment(edit)
    window: list[AlignedSlot] = select_window(alignment, window_radius)
    A = len(window)
    D = cache.hidden_dim
    L = len(layers)

    H_wt = torch.zeros(L, A, D)
    H_mut = torch.zeros(L, A, D)
    wt_present = torch.zeros(A)
    mut_present = torch.zeros(A)
    delta_valid = torch.zeros(A)
    slot_kind = torch.zeros(A, dtype=torch.long)
    wt_pos = torch.zeros(A, dtype=torch.long)
    mut_pos = torch.zeros(A, dtype=torch.long)
    insertion_rank = torch.zeros(A)

    for li, layer in enumerate(layers):
        wt_full = cache.get_wt(layer)          # (L_wt, D)
        mut_full = cache.get_mut(var_id, layer)  # (L_mut, D)
        for a, slot in enumerate(window):
            if slot.wt_pos is not None:
                H_wt[li, a] = wt_full[slot.wt_pos - 1]
            if slot.mut_pos is not None:
                H_mut[li, a] = mut_full[slot.mut_pos - 1]

    for a, slot in enumerate(window):
        wt_present[a] = 1.0 if slot.wt_pos is not None else 0.0
        mut_present[a] = 1.0 if slot.mut_pos is not None else 0.0
        delta_valid[a] = 1.0 if (slot.wt_pos is not None and slot.mut_pos is not None) else 0.0
        slot_kind[a] = slot.kind
        wt_pos[a] = slot.wt_pos or 0
        mut_pos[a] = slot.mut_pos or 0
        insertion_rank[a] = float(slot.insertion_rank)

    delta = torch.where(
        delta_valid.bool().view(1, A, 1).expand(L, A, D), H_mut - H_wt, torch.zeros(L, A, D)
    )
    anchor_coord = torch.tensor(anchor_relative_coordinates(edit, window), dtype=torch.float32)
    token_valid = torch.ones(A)

    return SampleTokens(
        var_id=var_id, n_slots=A, layers=list(layers),
        H_wt=H_wt, H_mut=H_mut, delta=delta,
        wt_present=wt_present, mut_present=mut_present, delta_valid=delta_valid,
        token_valid=token_valid, slot_kind=slot_kind, wt_pos=wt_pos, mut_pos=mut_pos,
        anchor_rel_coord=anchor_coord, insertion_rank=insertion_rank,
    )


def attention_valid_for_mode(token_valid: Tensor, delta_valid: Tensor, model_mode: str) -> Tensor:
    """A: token_valid & delta_valid (paired-only attention). B/C: token_valid."""
    from .schema import MODEL_PAIRED_DELTA

    if model_mode == MODEL_PAIRED_DELTA:
        return token_valid * delta_valid
    return token_valid
