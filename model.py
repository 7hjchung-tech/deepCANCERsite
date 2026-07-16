"""
model.py — RAD51C variant-effect model, 4 comparable configurations (M1-M4).

Modality streams feeding the shared residual backbone:
    diff_emb  (1280 -> BottleneckMLP -> bottleneck.out_dim)   WT-difference ESM embedding
    mag       (4    -> MagMLP        -> mag_dim)              magnitude scalars of the diff
    struct    (22, or 11 if use_block_b=False)                Block A[+B] structure features
    meta      (6)                                             variant-type one-hot + continuous

concat_dim = bottleneck.out_dim + mag_dim + struct_dim + meta_dim
No dimension above is hardcoded into the model: struct_dim/meta_dim are read
from the actual feature files' shapes (see compute_dims()), and
bottleneck.out_dim/mag_dim come from config.

Two ESM paths (see src/embeddings/diff_embedder.py):
    esm_mode="cached" — DiffEmbedder looks up a precomputed diff_emb_raw.pt.
                        Used by M1/M2 (frozen backbone; never trained, so
                        the cache never goes stale).
    esm_mode="e2e"    — DiffEmbedder holds a live (optionally LoRA) ESMEncoder
                        and computes the diff on every forward. Required for
                        M3/M4 since gradients must reach the LoRA weights.

Block A occupies columns [0:11] and Block B occupies columns [11:22] of
rad51c_X.npy (verified against rad51c_struct_features.csv column names, not
assumed) -- use_block_b=False slices to struct_x[:, :11].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.embeddings.bottleneck import BottleneckMLP
from src.embeddings.diff_embedder import DiffEmbedder

STRUCT_BLOCK_A_DIM = 11   # columns [0:11]  in rad51c_X.npy  (verified via column names)
STRUCT_BLOCK_B_DIM = 11   # columns [11:22] in rad51c_X.npy
MAG_DIM = 4
MAG_HIDDEN_DIM = 16


class MagMLP(nn.Module):
    """Project the 4 diff-magnitude scalars, mirroring BottleneckMLP's shape.

    Architecture: Linear(in_dim -> out_dim) -> GELU -> LayerNorm(out_dim)
    """

    def __init__(self, in_dim: int = MAG_DIM, out_dim: int = MAG_HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Pre-norm residual FFN block: x + FFN(LayerNorm(x))."""

    def __init__(self, dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(self.norm(x))


def compute_dims(cfg: dict, struct_x_path: str, meta_x_path: str) -> dict:
    """Read the actual feature files to determine struct_dim/meta_dim/concat_dim.

    Never hardcode these -- if the struct or meta pipeline output changes
    shape, this recomputes automatically and the assertions below catch any
    mismatch with what the config *thinks* the dims are.
    """
    struct_full_dim = int(np.load(struct_x_path, mmap_mode="r").shape[1])
    meta_dim = int(np.load(meta_x_path, mmap_mode="r").shape[1])

    expected_full = STRUCT_BLOCK_A_DIM + STRUCT_BLOCK_B_DIM
    assert struct_full_dim == expected_full, (
        f"rad51c_X.npy has {struct_full_dim} struct cols, expected "
        f"{expected_full} (Block A {STRUCT_BLOCK_A_DIM} + Block B {STRUCT_BLOCK_B_DIM})"
    )

    use_block_b = bool(cfg["use_block_b"])
    struct_dim = struct_full_dim if use_block_b else STRUCT_BLOCK_A_DIM

    bottleneck_out = int(cfg["bottleneck"]["out_dim"])
    mag_hidden = int(cfg.get("mag_hidden_dim", MAG_HIDDEN_DIM))
    concat_dim = bottleneck_out + mag_hidden + struct_dim + meta_dim

    return {
        "struct_full_dim": struct_full_dim,
        "struct_dim": struct_dim,
        "meta_dim": meta_dim,
        "bottleneck_out_dim": bottleneck_out,
        "mag_hidden_dim": mag_hidden,
        "concat_dim": concat_dim,
    }


class RAD51CModel(nn.Module):
    def __init__(
        self,
        cfg: dict,
        dims: dict,
        manifest: dict | None = None,
        wt_seq: str | None = None,
        cache_path: str | None = None,
    ) -> None:
        super().__init__()
        self.use_block_b = bool(cfg["use_block_b"])
        self.dims = dims

        esm_mode = cfg["esm_mode"]
        lora_cfg = cfg.get("lora") if cfg.get("use_lora") else None
        self.diff_embedder = DiffEmbedder(
            mode=esm_mode,
            pooling_cfg=cfg["pooling"],
            cache_path=cache_path,
            manifest=manifest,
            wt_seq=wt_seq,
            esm_cfg=cfg.get("esm"),
            lora_cfg=lora_cfg,
        )
        self.bottleneck = BottleneckMLP(cfg["bottleneck"])
        self.mag_mlp = MagMLP(MAG_DIM, dims["mag_hidden_dim"])

        hidden = int(cfg["hidden_dim"])
        ffn = int(cfg["ffn_dim"])
        n_blocks = int(cfg["n_blocks"])
        dropout = float(cfg["dropout"])

        self.input_proj = nn.Linear(dims["concat_dim"], hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, ffn, dropout) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.reg_head = nn.Linear(hidden, 1)   # z_score regression only (no cls head yet)

    def forward(self, var_ids: list[str], struct_x: Tensor, meta_x: Tensor) -> Tensor:
        de = self.diff_embedder(var_ids)              # {"diff": (B,1280), "mag": (B,4)}
        diff_z = self.bottleneck(de["diff"])           # (B, bottleneck_out_dim)
        mag_z = self.mag_mlp(de["mag"])                # (B, mag_hidden_dim)

        struct = struct_x if self.use_block_b else struct_x[:, :STRUCT_BLOCK_A_DIM]

        x = torch.cat([diff_z, mag_z, struct, meta_x], dim=-1)
        assert x.shape[-1] == self.dims["concat_dim"], (
            f"concat produced {x.shape[-1]}, expected {self.dims['concat_dim']}"
        )
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        return self.reg_head(h).squeeze(-1)            # (B,)


def build_model(
    cfg: dict,
    struct_x_path: str,
    meta_x_path: str,
    manifest: dict | None = None,
    wt_seq: str | None = None,
    cache_path: str | None = None,
) -> tuple[RAD51CModel, dict]:
    dims = compute_dims(cfg, struct_x_path, meta_x_path)
    model = RAD51CModel(cfg, dims, manifest=manifest, wt_seq=wt_seq, cache_path=cache_path)
    return model, dims
