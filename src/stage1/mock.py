"""A deliberately trivial consumer of the Stage 1 interface.

THIS IS NOT A MODEL. It exists for exactly one reason: to prove that tensors
can flow

    frozen cache -> loader -> Stage 1 input contract -> scalar prediction

and that the masks are honoured on the way through. It is not trained, not
tuned, not compared, and its outputs are never a result. Nothing it produces
may be called an F-seq baseline or reported as performance.

The real token-level Stage 1 is an external dependency; when it arrives it
replaces this file and nothing else.

WHY IT IS SHAPED THE WAY IT IS
------------------------------
Masked mean over `token_valid` slots, then one Linear. Two properties make it
useful as a plumbing test and they are the only reasons for the design:

  * it reads `token_valid`, so a padding slot cannot influence the output --
    `test_padding_cannot_change_the_output` perturbs the padded region and
    asserts the prediction is bit-identical;
  * it reads gaps too, so an indel `wt_only` / `mut_only` slot DOES influence
    the output -- proving gaps are not being silently treated as padding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .interface import Stage1Batch

MOCK_IS_NOT_A_SCIENTIFIC_MODEL = True


def masked_mean_over_tokens(x: torch.Tensor, token_valid: torch.Tensor) -> torch.Tensor:
    """[B,L,A,D] -> [B,L,D], averaging only over token_valid slots.

    Padding is excluded from both numerator and denominator. Gaps are included:
    they are real slots with a real representation on the side that exists.
    """
    m = token_valid.unsqueeze(1).unsqueeze(-1).to(x.dtype)      # [B,1,A,1]
    total = (x * m).sum(dim=2)                                   # [B,L,D]
    count = m.sum(dim=2).clamp_min(1.0)                          # [B,1,1]
    return total / count


class MockTokenConsumer(nn.Module):
    """Masked mean of delta_H over valid tokens -> Linear -> scalar. Plumbing only."""

    def __init__(self, n_layers: int = 3, embed_dim: int = 1280) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.head = nn.Linear(n_layers * embed_dim, 1)

    def forward(self, batch: Stage1Batch) -> torch.Tensor:
        pooled = masked_mean_over_tokens(batch.delta_H, batch.token_valid)  # [B,L,D]
        flat = pooled.reshape(pooled.shape[0], -1)                          # [B,L*D]
        return self.head(flat).squeeze(-1)                                  # [B]
