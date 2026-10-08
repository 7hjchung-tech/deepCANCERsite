"""WT 3D-neighbor structure store + residue-level StructureTokenizer.

Tensor axes are named explicitly everywhere below because this module introduces a THIRD axis
(residue) on top of the existing field axis that structure_tokenizer/tokenizer.py already
encodes -- attention here runs over the RESIDUE axis (9 WT positions: anchor + 8 nearest in
3D), never over the field axis (still 9 fields per residue, encoded exactly as before).

    raw per anchor:      continuous [9 residues, 8 fields], ss [9 residues]
    tokenizer output:    T          [B, 9 residues, 9 fields, 32]
    after mean(field):               [B, 9 residues, 32]
    after A_res (shared):  s         [B, 9 residues, 32]           <- residue tokens
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .schema import CONTINUOUS_COLUMNS, N_STRUCT_TOKENS, STRUCT_TOKEN_DIM


class WTNeighborStore:
    """Wraps data/structure/results/wt_neighbor_cache.npz (built by build_wt_neighbor_cache.py).

    raw_for_anchors(anchor_positions, mode) returns the SAME 9-slot content for mode in
    {"e1","e2"} -- only the `valid` mask differs (E1: anchor slot only; E2: anchor + 8
    neighbors), per the task spec's "E1/E2는 동일한 모델을 사용하고 valid mask만 다르게 한다".
    """

    def __init__(self, cache_path: str | Path):
        d = np.load(cache_path, allow_pickle=True)
        self.meta = json.loads(str(d["meta_json"]))
        self.positions = d["positions"]
        self.pos2row = {int(p): i for i, p in enumerate(self.positions)}
        self.continuous = d["continuous"].astype(np.float32)      # [376, 8]
        self.ss = d["ss"].astype(np.int64)                        # [376]
        self.neighbor_idx = d["neighbor_idx"].astype(np.int64)    # [376, 9] row into the 376 array
        self.distance = d["distance"].astype(np.float32)          # [376, 9]
        self.offset = d["offset"].astype(np.int64)                # [376, 9]
        self.is_anchor = d["is_anchor"].astype(bool)               # [376, 9]
        self.valid_full = d["valid"].astype(bool)                  # [376, 9] (all True in this structure)
        if self.continuous.shape[1] != len(CONTINUOUS_COLUMNS):
            raise ValueError("cache continuous-feature width does not match CONTINUOUS_COLUMNS")

    def raw_for_anchors(self, anchor_positions: list[int], mode: str) -> dict:
        if mode not in ("e1", "e2"):
            raise ValueError(f"mode must be 'e1' or 'e2', got {mode!r}")
        missing = [p for p in anchor_positions if p not in self.pos2row]
        if missing:
            raise ValueError(f"{len(missing)} anchor position(s) not in the WT neighbor cache "
                             f"(first 5): {missing[:5]}")
        rows = np.array([self.pos2row[p] for p in anchor_positions], dtype=np.int64)
        idx = self.neighbor_idx[rows]                              # [B, 9]
        cont = self.continuous[idx]                                 # [B, 9, 8]
        ss = self.ss[idx]                                           # [B, 9]
        dist = self.distance[rows]                                  # [B, 9]
        off = self.offset[rows]                                     # [B, 9]
        anchor_flag = self.is_anchor[rows]                          # [B, 9]
        valid = self.valid_full[rows].copy()                       # [B, 9]
        if mode == "e1":
            valid = valid.copy()
            valid[:, 1:] = False
        if not valid[:, 0].all():
            raise ValueError("anchor slot (0) is invalid for some sample -- anchor mapping failed upstream")
        return {
            "continuous": torch.as_tensor(cont), "ss": torch.as_tensor(ss),
            "distance": torch.as_tensor(dist), "offset": torch.as_tensor(off),
            "is_anchor": torch.as_tensor(anchor_flag), "valid": torch.as_tensor(valid),
        }

    def fit_positions_for_train_anchors(self, train_anchor_positions: list[int]) -> list[int]:
        """Union of every WT position that appears in ANY slot (anchor or neighbor) of any
        train anchor's 9-slot set, deduplicated -- the "공정한 비교를 위한 공통 fit 집합"."""
        rows = np.array([self.pos2row[p] for p in train_anchor_positions], dtype=np.int64)
        idx = self.neighbor_idx[rows]                               # [n_train, 9]
        valid = self.valid_full[rows]
        used_rows = sorted(set(int(r) for r in idx[valid]))
        return [int(self.positions[r]) for r in used_rows]

    def raw_for_positions(self, positions: list[int]) -> dict:
        rows = np.array([self.pos2row[p] for p in positions], dtype=np.int64)
        return {"continuous": torch.as_tensor(self.continuous[rows]), "ss": torch.as_tensor(self.ss[rows])}


class ResidueStructureTokenizer(nn.Module):
    """s_i = A_res(mean_field(T_i)), A_res shared across all 9 residue slots (anchor and
    neighbors alike -- no per-residue or per-slot-kind weights)."""

    out_residues: int = N_STRUCT_TOKENS
    out_dim: int = STRUCT_TOKEN_DIM

    def __init__(self, d_s: int = STRUCT_TOKEN_DIM, n_bins: int = 4) -> None:
        super().__init__()
        self.d_s, self.n_bins = d_s, n_bins
        self.inner = None
        self.a_res = nn.Linear(d_s, d_s)   # shared residue adapter, Linear(32,32) per the task spec

    def fit_preprocessing(self, train_raw_fit_positions: dict) -> None:
        """train_raw_fit_positions: {"continuous":[M,8], "ss":[M]} for the M positions from
        WTNeighborStore.fit_positions_for_train_anchors (train anchors' full kNN union, not
        just the anchors themselves)."""
        from structure_tokenizer.tokenizer import StructureTokenizer as QkModule, fit_qk_bins
        cont = train_raw_fit_positions["continuous"].detach().cpu().numpy()
        self.inner = QkModule(fit_qk_bins(cont, self.n_bins), self.d_s)

    def forward(self, raw: dict) -> torch.Tensor:
        if self.inner is None:
            raise RuntimeError("ResidueStructureTokenizer.fit_preprocessing(...) must run before forward")
        cont, ss = raw["continuous"], raw["ss"]                      # [B, 9 residues, 8], [B, 9 residues]
        B, R, F = cont.shape
        T = self.inner(cont.reshape(B * R, F), ss.reshape(B * R))    # [B*R, 9 fields, 32]
        T = T.view(B, R, T.shape[1], T.shape[2])                     # [B, 9 residues, 9 fields, 32]
        s = T.mean(dim=2)                                            # mean over the FIELD axis -> [B, 9 residues, 32]
        return self.a_res(s)                                         # shared adapter -> residue tokens [B, 9, 32]
