"""BottleneckMLP: project a difference embedding to a lower-dimensional space.

Architecture: Linear(in_dim → out_dim) → GELU → LayerNorm(out_dim)

LayerNorm is applied after the activation so that the downstream
classification head receives a scale-normalised representation.

Dropout is intentionally omitted at this stage and will be added when the
training loop is wired up.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class BottleneckMLP(nn.Module):
    """Single-layer bottleneck projection.

    Args:
        cfg: config["bottleneck"] dict with keys:
             in_dim  (int, default 1280)
             out_dim (int, default 256)

    Input:  (in_dim,) or (B, in_dim)
    Output: (out_dim,) or (B, out_dim)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        in_dim  = int(cfg.get("in_dim",  1280))
        out_dim = int(cfg.get("out_dim", 256))

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
