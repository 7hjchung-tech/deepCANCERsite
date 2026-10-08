"""NeighborhoodFiLMModel: Stage1-pooled-hidden query -> residue-token cross-attention -> FiLM.

    h_base in R^D                 Stage1's own pooled hidden (SequenceHead's input, z_seq)
    s_i    in R^32  (i=1..9)      residue tokens from ResidueStructureTokenizer

    Q            = W_Q h_base                         [B, 32],   W_Q: D -> 32, no bias
    K_content_i  = W_K s_i                             [B, 9, 32], W_K: 32 -> 32, no bias
    V_i          = W_V s_i                             [B, 9, 32], W_V: 32 -> 32, no bias
    K_i = K_content_i + metadata_scale * ( E_seq(i-p) + E_dist(d_ip) + E_anchor(is_anchor_i) )
        E_seq:    deterministic signed sinusoidal encoding of the offset (no parameters)
        E_dist:   fixed-center/width Gaussian RBF of the raw 3D distance, then Linear(16,32,
                  bias=False) (the only trainable part of E_dist)
        E_anchor: Embedding(2, 32), anchor-flag lookup
        metadata_scale is a FIXED (not learned) small constant, identical for E1 and E2, so
        that the metadata terms never dominate the content term at initialisation.
    logits_i = Q . K_i / sqrt(32)         (NOT cosine-normalised, no learnable temperature)
    a        = masked_softmax(logits, over the residue axis)     padding weight exactly 0
    c_struct = sum_i a_i V_i                                     [B, 32]

    [gamma, beta] = FiLMGenerator(c_struct)      32 -> 32 -> 2D, last layer zero-init
    h_mod = (1 + gamma) * h_base + beta
    delta_y = CorrectionHead(concat(h_mod, c_struct))   (D+32) -> 32 -> 1, last layer zero-init
    y_final = y_base + delta_y

Metadata is added to K only, never to V, per the task spec. An all-masked attention input
(every residue slot invalid) is a hard error, not a silent zero -- this should only ever be
reachable from a synthetic/malformed batch, since the anchor slot is always valid by
construction (both E1 and E2 keep it valid; see WTNeighborStore.raw_for_anchors).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .schema import N_STRUCT_TOKENS, STRUCT_TOKEN_DIM

DIST_MAX = 30.0          # Å, RBF range -- out-of-range distances are NOT clipped (recorded instead)
N_DIST_CENTERS = 16
METADATA_SCALE = 0.1     # fixed, documented, identical for E1 and E2


def _masked_std(x: Tensor, valid: Tensor) -> Tensor:
    """Per-row std of x over valid entries only (0 when only 1 entry is valid, e.g. E1)."""
    n = valid.sum(-1, keepdim=True).clamp_min(1).to(x.dtype)
    vf = valid.to(x.dtype)
    mean = (x * vf).sum(-1, keepdim=True) / n
    var = (((x - mean) ** 2) * vf).sum(-1, keepdim=True) / n
    return var.clamp_min(0).sqrt().squeeze(-1)


class NeighborhoodFiLMModel(nn.Module):
    def __init__(self, d: int, struct_dim: int = STRUCT_TOKEN_DIM, d_a: int = 32,
                 n_dist_centers: int = N_DIST_CENTERS, dist_max: float = DIST_MAX,
                 metadata_scale: float = METADATA_SCALE, film_hidden: int = 32, head_hidden: int = 32) -> None:
        super().__init__()
        self.d, self.d_a, self.metadata_scale = d, d_a, metadata_scale
        self.w_q = nn.Linear(d, d_a, bias=False)
        self.w_k = nn.Linear(struct_dim, d_a, bias=False)
        self.w_v = nn.Linear(struct_dim, d_a, bias=False)
        centers = torch.linspace(0.0, dist_max, n_dist_centers)
        width = float(centers[1] - centers[0]) if n_dist_centers > 1 else 1.0
        self.register_buffer("rbf_centers", centers)
        self.rbf_width = width
        self.dist_proj = nn.Linear(n_dist_centers, d_a, bias=False)
        self.anchor_emb = nn.Embedding(2, d_a)
        self.film = nn.Sequential(nn.Linear(d_a, film_hidden), nn.GELU(), nn.Linear(film_hidden, 2 * d))
        self.head = nn.Sequential(nn.Linear(d + d_a, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
        for last in (self.film[-1], self.head[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _rbf(self, dist: Tensor) -> Tensor:
        """[B, 9] raw distances (Å) -> [B, 9, n_centers] Gaussian RBF. Not clipped; distances
        beyond dist_max simply fall on the tail of the last few centers' Gaussians."""
        diff = dist.unsqueeze(-1) - self.rbf_centers.view(1, 1, -1)
        return torch.exp(-(diff ** 2) / (2 * self.rbf_width ** 2))

    def forward(self, h_base: Tensor, s: Tensor, distance: Tensor, offset: Tensor,
               is_anchor: Tensor, valid: Tensor, y_base: Tensor) -> dict:
        """h_base [B,D]; s [B,9,32] residue tokens; distance/offset/is_anchor/valid [B,9]."""
        if s.shape[1] != N_STRUCT_TOKENS:
            raise ValueError(f"expected {N_STRUCT_TOKENS} residue slots, got {s.shape[1]}")
        if h_base.shape[-1] != self.d:
            raise ValueError(f"h_base dim {h_base.shape[-1]} != model d={self.d} (check Stage1 TOKEN_DIM)")
        if not valid.any(dim=-1).all():
            raise ValueError("an all-masked structure input was passed (no valid residue slot) -- "
                             "this should be unreachable from real data; the anchor slot is always valid")

        from src.stage1.positional import sinusoidal_encoding   # reused as-is, no duplicate implementation
        e_seq = sinusoidal_encoding(offset.float(), self.d_a)                  # [B, 9, 32], no params
        e_dist = self.dist_proj(self._rbf(distance))                           # [B, 9, 32]
        e_anchor = self.anchor_emb(is_anchor.long())                           # [B, 9, 32]
        meta = self.metadata_scale * (e_seq + e_dist + e_anchor)

        q = self.w_q(h_base)                                                   # [B, 32]
        k_content = self.w_k(s)                                                # [B, 9, 32]
        k = k_content + meta
        v = self.w_v(s)                                                        # [B, 9, 32]

        raw_logits = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(self.d_a)    # [B, 9], pre-mask (for diagnostics)
        neg_inf = torch.finfo(raw_logits.dtype).min
        masked_logits = raw_logits.masked_fill(~valid, neg_inf)
        weights = torch.softmax(masked_logits, dim=-1) * valid.to(raw_logits.dtype)   # padding weight exactly 0
        c_struct = torch.einsum("bn,bnd->bd", weights, v)                      # [B, 32]

        gamma, beta = self.film(c_struct).chunk(2, dim=-1)                     # [B, D] each
        h_mod = (1.0 + gamma) * h_base + beta
        delta = self.head(torch.cat([h_mod, c_struct], dim=-1)).squeeze(-1)

        n_valid = valid.sum(-1)
        ent = -(weights * weights.clamp_min(1e-12).log()).sum(-1)              # [B], 0 when N_valid==1
        ent_norm = torch.where(n_valid > 1, ent / n_valid.clamp_min(2).float().log(),
                               torch.full_like(ent, float("nan")))              # N/A (nan) when N_valid==1
        local_mask = (~is_anchor) & (offset.abs() <= 2) & valid
        nonlocal_mask = (~is_anchor) & (offset.abs() > 2) & valid
        anchor_mass = (weights * is_anchor.to(weights.dtype)).sum(-1)
        local_mass = (weights * local_mask.to(weights.dtype)).sum(-1)
        nonlocal_mass = (weights * nonlocal_mask.to(weights.dtype)).sum(-1)

        return {
            "pred": y_base + delta, "delta": delta, "c_struct": c_struct, "weights": weights,
            "gamma": gamma, "beta": beta, "entropy": ent, "entropy_norm": ent_norm, "n_valid": n_valid,
            "anchor_mass": anchor_mass, "local_mass": local_mass, "nonlocal_mass": nonlocal_mass,
            "key_content_norm": k_content.norm(dim=-1), "key_meta_norm": meta.norm(dim=-1),
            "e_seq_norm": e_seq.norm(dim=-1), "e_dist_norm": e_dist.norm(dim=-1), "e_anchor_norm": e_anchor.norm(dim=-1),
            "logit_std": _masked_std(raw_logits, valid),
        }

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
