"""Shared trainable modules: content projections, layer embedding,
constant-query cosine-attention pooling, and the sequence head.

All three model modes (paired_delta / branched_projection /
unified_reference_delta) are assembled from these same building blocks --
only how ContentBuilder combines its inputs differs (see build_content_builder).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .schema import (
    DEFAULT_BOTTLENECK_DIM,
    FLAGS_DIM,
    MODEL_BRANCHED_PROJECTION,
    MODEL_MODES,
    MODEL_PAIRED_DELTA,
    MODEL_UNIFIED_REFERENCE_DELTA,
    SLOT_MUT_ONLY,
    SLOT_PAIRED,
    SLOT_WT_ONLY,
    TOKEN_DIM,
)


class ProjectionMLP(nn.Module):
    """Linear(in_dim -> bottleneck) -> GELU -> Linear(bottleneck -> out_dim).

    This exact shape (both Linears with bias) is what the task spec's
    parameter-count table (45,216 / 135,648 / 86,368) is derived from --
    see README_STAGE1.md for the arithmetic.
    """

    def __init__(self, in_dim: int, out_dim: int = TOKEN_DIM, bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _kind_mask(slot_kind: Tensor, kind: int) -> Tensor:
    """(B, A) int tensor -> (B, 1, A, 1) float mask broadcastable over (B, Lyr, A, D)."""
    return (slot_kind == kind).to(torch.float32).unsqueeze(1).unsqueeze(-1)


class PairedDeltaContentBuilder(nn.Module):
    """Model A: content = P_delta(delta) everywhere.

    WT-only / MUT-only slots still get a (masked-out-of-attention) content
    value here -- excluding them is an ATTENTION concern (attention_valid),
    not a content-computation concern for this model.
    """

    mode = MODEL_PAIRED_DELTA

    def __init__(self, d_esm: int, bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM) -> None:
        super().__init__()
        self.p_delta = ProjectionMLP(d_esm, TOKEN_DIM, bottleneck_dim)

    def forward(self, H_wt: Tensor, H_mut: Tensor, delta: Tensor, wt_present: Tensor,
                mut_present: Tensor, delta_valid: Tensor, slot_kind: Tensor) -> Tensor:
        return self.p_delta(delta)


class BranchedProjectionContentBuilder(nn.Module):
    """Model B: Δ-projection for paired slots, separate WT/MUT projections
    for one-sided slots. Three independent trainable projections."""

    mode = MODEL_BRANCHED_PROJECTION

    def __init__(self, d_esm: int, bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM) -> None:
        super().__init__()
        self.p_delta = ProjectionMLP(d_esm, TOKEN_DIM, bottleneck_dim)
        self.p_wt = ProjectionMLP(d_esm, TOKEN_DIM, bottleneck_dim)
        self.p_mut = ProjectionMLP(d_esm, TOKEN_DIM, bottleneck_dim)

    def forward(self, H_wt: Tensor, H_mut: Tensor, delta: Tensor, wt_present: Tensor,
                mut_present: Tensor, delta_valid: Tensor, slot_kind: Tensor) -> Tensor:
        c_delta = self.p_delta(delta)
        c_wt = self.p_wt(H_wt)
        c_mut = self.p_mut(H_mut)
        m_paired = _kind_mask(slot_kind, SLOT_PAIRED)
        m_wt_only = _kind_mask(slot_kind, SLOT_WT_ONLY)
        m_mut_only = _kind_mask(slot_kind, SLOT_MUT_ONLY)
        return m_paired * c_delta + m_wt_only * c_wt + m_mut_only * c_mut


class UnifiedReferenceDeltaContentBuilder(nn.Module):
    """Model C: one shared projection over [R ; delta ; flags].

    R = H_wt if wt_present else (H_mut if mut_present else 0).
    flags (6) = [wt_present, mut_present, delta_valid, is_paired, is_wt_only, is_mut_only].
    """

    mode = MODEL_UNIFIED_REFERENCE_DELTA

    def __init__(self, d_esm: int, bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM) -> None:
        super().__init__()
        self.p_shared = ProjectionMLP(2 * d_esm + FLAGS_DIM, TOKEN_DIM, bottleneck_dim)

    def forward(self, H_wt: Tensor, H_mut: Tensor, delta: Tensor, wt_present: Tensor,
                mut_present: Tensor, delta_valid: Tensor, slot_kind: Tensor) -> Tensor:
        wt_p = wt_present.unsqueeze(1).unsqueeze(-1)     # (B,1,A,1)
        mut_p = mut_present.unsqueeze(1).unsqueeze(-1)
        R = wt_p * H_wt + (1.0 - wt_p) * mut_p * H_mut

        is_paired = (slot_kind == SLOT_PAIRED).to(torch.float32)
        is_wt_only = (slot_kind == SLOT_WT_ONLY).to(torch.float32)
        is_mut_only = (slot_kind == SLOT_MUT_ONLY).to(torch.float32)
        flags = torch.stack(
            [wt_present, mut_present, delta_valid, is_paired, is_wt_only, is_mut_only], dim=-1
        )  # (B, A, 6)
        n_layers = H_wt.shape[1]
        flags = flags.unsqueeze(1).expand(-1, n_layers, -1, -1)  # (B, Lyr, A, 6)

        x = torch.cat([R, delta, flags], dim=-1)
        return self.p_shared(x)


_BUILDERS = {
    MODEL_PAIRED_DELTA: PairedDeltaContentBuilder,
    MODEL_BRANCHED_PROJECTION: BranchedProjectionContentBuilder,
    MODEL_UNIFIED_REFERENCE_DELTA: UnifiedReferenceDeltaContentBuilder,
}


def build_content_builder(mode: str, d_esm: int, bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM) -> nn.Module:
    if mode not in MODEL_MODES:
        raise ValueError(f"unknown model mode {mode!r}, expected one of {MODEL_MODES}")
    return _BUILDERS[mode](d_esm, bottleneck_dim)


def content_projection_param_count(mode: str, d_esm: int, bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM) -> int:
    """Closed-form parameter count for the content projection(s) only.

    Linear(in,b) has in*b+b params; Linear(b,128) has b*128+128 params.
    """
    def one(in_dim: int) -> int:
        return (in_dim * bottleneck_dim + bottleneck_dim) + (bottleneck_dim * TOKEN_DIM + TOKEN_DIM)

    if mode == MODEL_PAIRED_DELTA:
        return one(d_esm)
    if mode == MODEL_BRANCHED_PROJECTION:
        return 3 * one(d_esm)
    if mode == MODEL_UNIFIED_REFERENCE_DELTA:
        return one(2 * d_esm + FLAGS_DIM)
    raise ValueError(f"unknown model mode {mode!r}")


class ConstantQueryPooling(nn.Module):
    """Cosine-similarity attention pooling against one learned, sample-shared query.

    alpha_i = softmax_i( cos(q0, K_i) / tau ),  z = sum_i alpha_i * V_i

    tau is kept strictly positive via softplus; invalid tokens get -inf
    logits (zero weight, and excluded from softmax normalisation) rather
    than being zeroed out post-hoc.
    """

    def __init__(self, dim: int = TOKEN_DIM) -> None:
        super().__init__()
        self.q0 = nn.Parameter(torch.randn(dim) * 0.02)
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, K: Tensor, V: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """K, V: (B, T, D). valid_mask: (B, T) in {0,1} (float or bool).

        Returns (z: (B, D), weights: (B, T)).
        """
        q = F.normalize(self.q0, dim=-1)
        k = F.normalize(K, dim=-1, eps=1e-8)
        sim = torch.einsum("btd,d->bt", k, q)
        tau = F.softplus(self.log_tau) + 1e-4
        logits = sim / tau
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(valid_mask <= 0, neg_inf)
        weights = torch.softmax(logits, dim=-1)
        z = torch.einsum("bt,btd->bd", weights, V)
        return z, weights


class SequenceHead(nn.Module):
    """128 -> 256 -> 128 -> 1, pre-norm residual FFN + 2 LayerNorms.

    h = z + FFN(LayerNorm(z));  h = LayerNorm(h);  out = Linear(h) -> scalar.
    Regression only (no classification head in this Stage 1 build).
    """

    def __init__(self, dim: int = TOKEN_DIM, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 1)

    def forward(self, z: Tensor) -> Tensor:
        h = z + self.ffn(self.norm1(z))
        h = self.norm2(h)
        return self.out(h).squeeze(-1)
