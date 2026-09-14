"""Fixed (non-trainable) positional encoding for Stage 1 residue tokens.

Convention chosen for THIS build (documented, not claimed to be an already-
settled formula from prior docs):

    dim 128 = [ 64 : signed sinusoidal encoding of anchor-relative coordinate
              | 64 : signed sinusoidal encoding of insertion rank, zeroed
                     out for every slot that is not mut_only ]

Both halves are deterministic functions of integer inputs -- same coordinate
always produces the same encoding regardless of window_radius W, batch
padding, or how many layers are requested.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .schema import PE_DIM, PE_HALF_DIM


def sinusoidal_encoding(positions: Tensor, dim: int) -> Tensor:
    """Standard signed sinusoidal encoding, applied to arbitrary signed floats.

    positions: (...,) float/int tensor.
    Returns (..., dim): dim/2 sin components followed by dim/2 cos components.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_encoding requires an even dim, got {dim}")
    half = dim // 2
    device = positions.device
    freq_idx = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (freq_idx / half))
    angles = positions.unsqueeze(-1).to(torch.float32) * inv_freq  # (..., half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def build_position_encoding(
    anchor_rel_coord: Tensor,
    insertion_rank: Tensor,
    is_mut_only: Tensor,
) -> Tensor:
    """Args (all same leading shape, e.g. (B, A)):
        anchor_rel_coord: signed WT-relative coordinate (0 for mut_only slots
                           by convention -- see alignment.anchor_relative_coordinates).
        insertion_rank:   1-based rank within an inserted span, 0 elsewhere.
        is_mut_only:      1.0/0.0 mask -- only mut_only slots keep the rank half.

    Returns (..., PE_DIM).
    """
    first_half = sinusoidal_encoding(anchor_rel_coord, PE_HALF_DIM)
    second_half = sinusoidal_encoding(insertion_rank, PE_HALF_DIM)
    second_half = second_half * is_mut_only.unsqueeze(-1)
    return torch.cat([first_half, second_half], dim=-1)


assert PE_HALF_DIM * 2 == PE_DIM
