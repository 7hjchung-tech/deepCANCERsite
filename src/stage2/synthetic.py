"""SMOKE-TEST ONLY structure tokenizer. Not the real StructureTokenizer.

It maps each raw Block A feature to one of the 9 token slots with a fixed
per-feature affine map and a secondary-structure embedding. It exists so the
Stage 2 pipeline can be exercised end to end before the real tokenizer lands.
Results produced with it must never be reported as Stage 2 performance.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .schema import N_STRUCT_TOKENS, STRUCT_TOKEN_DIM
from .structure import StructureTokenizer


class SyntheticSmokeTokenizer(StructureTokenizer):
    def __init__(self, dim: int = STRUCT_TOKEN_DIM) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(8, dim) * 0.1)
        self.shift = nn.Parameter(torch.zeros(8, dim))
        self.ss_emb = nn.Embedding(3, dim)
        self.fit_calls: list[int] = []

    def fit_preprocessing(self, train_raw: dict) -> None:
        self.fit_calls.append(int(train_raw["continuous"].shape[0]))

    def forward(self, raw: dict) -> torch.Tensor:
        x = raw["continuous"]                                  # [B, 8]
        cont = x.unsqueeze(-1) * self.scale.unsqueeze(0) + self.shift.unsqueeze(0)   # [B, 8, 32]
        ss = self.ss_emb(raw["ss"]).unsqueeze(1)               # [B, 1, 32]
        S = torch.cat([cont, ss], dim=1)
        assert S.shape[1] == N_STRUCT_TOKENS
        return S


def make_synthetic_tokenizer(cfg: dict) -> SyntheticSmokeTokenizer:
    return SyntheticSmokeTokenizer()
