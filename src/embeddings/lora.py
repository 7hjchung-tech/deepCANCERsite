"""LoRA (Low-Rank Adaptation) injection for ESM-2.

Wraps q_proj and v_proj in each TransformerLayer with a low-rank trainable
update:

    output = W·x  +  (α/r) · B·A·x

        A ∈ R^{r × d_in}   initialised with Kaiming-uniform
        B ∈ R^{d_out × r}  initialised to zero
        → LoRA output is exactly zero at init (preserves pretrained behaviour)

ESM-2 always uses rotary embeddings, so self.rot_emb is truthy in every
MultiheadAttention instance.  The fast-path condition `not self.rot_emb` is
therefore always False — the slow path (explicit q = self.q_proj(query) etc.)
is the only path taken.  Replacing self_attn.{q,v}_proj with LoRALinear
is sufficient; no additional bypass is needed.

Reference: Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoRALinear(nn.Module):
    """nn.Linear with a frozen base weight and a trainable low-rank update.

    forward(x) = base(x)  +  scale · B·(A·x)

    Properties mirror nn.Linear (in_features, out_features) so the module can
    be used as a drop-in replacement wherever those attributes are read.

    Args:
        linear: Source linear layer (must be frozen before this is called).
        rank:   LoRA rank r.  Paper suggests r ∈ {4, 8, 16}.
        alpha:  Scaling constant.  Effective scale = alpha / rank.
                Setting alpha = rank gives scale = 1 (no extra scaling).
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        in_features  = linear.in_features
        out_features = linear.out_features

        # Keep the frozen base as a sub-module so state_dict works correctly
        self.base  = linear
        self.scale = alpha / rank

        # A: random init (non-zero so gradients flow through A from the start)
        # B: zero init (ensures LoRA output = 0 at step 0)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        # F.linear(x, W) computes x @ W.T; works for any leading batch dims
        # e.g. (T, B, D) used by ESM-2's attention slow path
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.base(x) + self.scale * lora_out

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_layers: Optional[list] = None,
) -> None:
    """Replace q_proj and v_proj in ESM-2 layers with LoRALinear wrappers.

    Call this AFTER freezing all model parameters so that only the newly
    created lora_A / lora_B tensors have requires_grad=True.

    Args:
        model:         ESM2 model instance (model.layers is an nn.ModuleList).
        rank:          LoRA rank.
        alpha:         LoRA scaling factor (scale = alpha / rank).
        target_layers: 0-indexed layer indices to inject.
                       None → every layer (0 .. num_layers-1).
    """
    layers  = model.layers
    indices: range | list = range(len(layers)) if target_layers is None else target_layers

    for i in indices:
        attn = layers[i].self_attn          # MultiheadAttention
        attn.q_proj = LoRALinear(attn.q_proj, rank, alpha)
        attn.v_proj = LoRALinear(attn.v_proj, rank, alpha)


def lora_param_count(model: nn.Module) -> int:
    """Count trainable parameters (i.e. LoRA parameters after injection)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
