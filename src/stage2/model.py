"""Stage 2 structure-conditioned residual model.

Both query modes build the exact same set of modules (shared structure adapter,
variant-type embedding, attention temperature, FiLM generator, residual head),
so trainable parameter counts are identical by construction and the only
difference is where the query comes from and where the pooling happens:

  single_query : q  = c               -> one pooled vector z
  nine_query   : q_j = U_j + e_type   -> nine pooled vectors, z = mean_j z_j

with c = mean_j(U_j) + e_type, U_j = A(S_j), A shared across the 9 tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import attention_entropy, masked_cosine_attention
from .schema import N_STRUCT_TOKENS, QUERY_MODES, STRUCT_TOKEN_DIM, VARIANT_TYPES


class Stage2Model(nn.Module):
    def __init__(self, query_mode: str, d: int = 128, struct_dim: int = STRUCT_TOKEN_DIM,
                 film_hidden: int = 32, head_hidden: int = 32, tau_init: float | None = None) -> None:
        super().__init__()
        if query_mode not in QUERY_MODES:
            raise ValueError(f"query_mode must be one of {QUERY_MODES}, got {query_mode!r}")
        self.query_mode = query_mode
        self.d = d
        self.adapter = nn.Linear(struct_dim, d)                      # shared affine A: 32 -> d
        self.type_emb = nn.Embedding(len(VARIANT_TYPES), d)          # e_type
        # tau = softplus(log_tau) + 1e-4 (same as Stage 1). tau_init=None keeps log_tau=0 (tau ~= 0.69).
        # Otherwise log_tau is set so that tau starts at tau_init (inverse softplus).
        init_log_tau = 0.0 if tau_init is None else math.log(math.expm1(tau_init - 1e-4))
        self.log_tau = nn.Parameter(torch.tensor(init_log_tau))
        self.film = nn.Sequential(                                   # FiLMGenerator(c): d -> 32 -> 2d
            nn.Linear(d, film_hidden), nn.GELU(), nn.Linear(film_hidden, 2 * d),
        )
        self.head = nn.Sequential(                                   # ResidualHead(z_mod): d -> 32 -> 1
            nn.Linear(d, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1),
        )
        for last in (self.film[-1], self.head[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def tau(self) -> Tensor:
        return F.softplus(self.log_tau) + 1e-4

    def forward(self, S: Tensor, K: Tensor, V: Tensor, valid: Tensor, type_id: Tensor, y1: Tensor) -> dict:
        """S: [B, 9, 32]; K, V: [B, N, d]; valid: [B, N]; type_id: [B] long; y1: [B] (z-score)."""
        if S.shape[1] != N_STRUCT_TOKENS:
            raise ValueError(f"expected {N_STRUCT_TOKENS} structure tokens, got {S.shape[1]}")
        U = self.adapter(S)                                          # [B, 9, d]
        e = self.type_emb(type_id)                                   # [B, d]
        c = U.mean(dim=1) + e                                        # [B, d] global condition
        if self.query_mode == "single_query":
            q = c.unsqueeze(1)                                       # [B, 1, d]
        else:
            q = U + e.unsqueeze(1)                                   # [B, 9, d]
        out, weights = masked_cosine_attention(q, K, V, valid, self.tau())   # out [B, Q, d]
        z = out.squeeze(1) if self.query_mode == "single_query" else out.mean(dim=1)   # [B, d]
        gamma, beta = self.film(c).chunk(2, dim=-1)
        z_mod = (1.0 + gamma) * z + beta
        delta = self.head(z_mod).squeeze(-1)
        return {
            "pred": y1 + delta, "delta": delta, "z": z, "c": c,
            "gamma": gamma, "beta": beta, "weights": weights,
            "entropy": attention_entropy(weights),                   # [B, Q]
        }

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
