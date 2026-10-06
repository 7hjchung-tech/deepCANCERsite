"""Masked cosine attention over Stage 1 ESM tokens with an arbitrary query count.

Same math as src/stage1/modules.py ConstantQueryPooling (cosine similarity,
temperature tau, -inf on invalid tokens), generalised from one shared query to
Q queries. The softmax is always over the token axis N.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_cosine_attention(q: Tensor, K: Tensor, V: Tensor, valid: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
    """q: [B, Q, d]; K, V: [B, N, d]; valid: [B, N] (bool or {0,1}); tau: scalar > 0.

    Returns (out [B, Q, d], weights [B, Q, N]). Padding weights are exactly 0.
    A query whose sample has no valid token returns zeros (no NaN, no fake uniform mass).
    V is used as-is (its norm is preserved).
    """
    qn = F.normalize(q, dim=-1, eps=1e-8)
    kn = F.normalize(K, dim=-1, eps=1e-8)
    logits = torch.einsum("bqd,bnd->bqn", qn, kn) / tau
    valid_b = valid.bool().unsqueeze(1).expand_as(logits)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~valid_b, neg_inf)
    weights = torch.softmax(logits, dim=-1) * valid_b.to(logits.dtype)
    has_valid = valid_b.any(dim=-1, keepdim=True)
    weights = torch.where(has_valid, weights, torch.zeros_like(weights))
    out = torch.einsum("bqn,bnd->bqd", weights, V)
    return out, weights


def attention_entropy(weights: Tensor) -> Tensor:
    """Shannon entropy (nats) of each query's distribution. Padding weights are 0 and contribute 0. Returns [B, Q]."""
    return -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
