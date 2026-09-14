"""Stage1Model: assembles the shared modules around one mode-specific
content builder, per the task spec's shared-conditions section (§5/§8).

ESM itself is NOT part of this module -- callers hand it already-extracted
(cached, frozen) H_wt/H_mut/delta tensors via the batch dict produced by
Stage1Dataset.stage1_collate. This is what keeps ESM adaptation (LoRA/Mixout/
future fine-tuning) decoupled from these three model definitions: retraining
Stage 1 against an adapted ESM only means rebuilding the raw cache, not
touching this file.
"""

from __future__ import annotations

import zlib
from typing import Optional

import torch
import torch.nn as nn

from .metadata import raw_meta_features  # noqa: F401  (re-exported for convenience)
from .modules import ConstantQueryPooling, ProjectionMLP, SequenceHead, build_content_builder
from .positional import build_position_encoding
from .schema import (
    DEFAULT_BOTTLENECK_DIM,
    DEFAULT_D_ESM,
    MODEL_MODES,
    RAW_META_DIM,
    TOKEN_DIM,
)


def _seed_offset(name: str, base_seed: int) -> int:
    return (base_seed + zlib.crc32(name.encode("utf-8"))) & 0x7FFFFFFF


def _build_seeded(name: str, base_seed: Optional[int], builder_fn):
    """Build `builder_fn()` under a component-specific deterministic seed,
    then restore the global RNG state -- so which mode is built first/second
    never changes any other mode's initial weights (see task spec §5).
    """
    if base_seed is None:
        return builder_fn()
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(_seed_offset(name, base_seed))
        return builder_fn()
    finally:
        torch.random.set_rng_state(state)


class Stage1Model(nn.Module):
    def __init__(
        self,
        mode: str,
        d_esm: int = DEFAULT_D_ESM,
        bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM,
        layers: tuple[int, ...] = (33,),
        head_hidden: int = 256,
        dropout: float = 0.1,
        meta_raw_dim: int = RAW_META_DIM,
        init_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in MODEL_MODES:
            raise ValueError(f"unknown model mode {mode!r}, expected one of {MODEL_MODES}")
        self.mode = mode
        self.d_esm = d_esm
        self.layers = list(layers)
        self.register_buffer("layer_ids", torch.tensor(self.layers, dtype=torch.long), persistent=False)

        # Content builder is mode-specific in shape, so it cannot share init
        # with the other two modes bit-for-bit, but IS seeded deterministically
        # on its own so re-running the same mode is reproducible.
        self.content_builder = _build_seeded(
            "content_builder", init_seed, lambda: build_content_builder(mode, d_esm, bottleneck_dim)
        )

        # Everything below is IDENTICAL in shape across all three modes, so it
        # is seeded with the SAME per-component seed regardless of `mode` --
        # comparing A/B/C is not confounded by which mode happened to consume
        # RNG state first during construction.
        max_layer = max(self.layers)
        self.layer_embedding = _build_seeded(
            "layer_embedding", init_seed, lambda: nn.Embedding(max_layer + 1, TOKEN_DIM)
        )
        self.metadata_encoder = _build_seeded(
            "metadata_encoder", init_seed,
            lambda: ProjectionMLP(meta_raw_dim, TOKEN_DIM, bottleneck_dim),
        )
        self.pooling = _build_seeded("pooling", init_seed, lambda: ConstantQueryPooling(TOKEN_DIM))
        self.head = _build_seeded(
            "head", init_seed, lambda: SequenceHead(TOKEN_DIM, head_hidden, dropout)
        )

        self._frozen_for_stage2 = False

    # ------------------------------------------------------------------
    def forward(self, batch: dict, return_extras: bool = False) -> dict:
        H_wt, H_mut, delta = batch["H_wt"], batch["H_mut"], batch["delta"]
        wt_present, mut_present = batch["wt_present"], batch["mut_present"]
        delta_valid, slot_kind = batch["delta_valid"], batch["slot_kind"]
        attention_valid = batch["attention_valid"]
        B, Lyr, A, _ = H_wt.shape

        content = self.content_builder(H_wt, H_mut, delta, wt_present, mut_present, delta_valid, slot_kind)

        pe = build_position_encoding(
            batch["anchor_rel_coord"], batch["insertion_rank"],
            (slot_kind == 2).to(torch.float32),  # SLOT_MUT_ONLY
        )  # (B, A, 128)
        pe = pe.unsqueeze(1).expand(B, Lyr, A, TOKEN_DIM)

        layer_ids = batch.get("layer_ids_tensor")
        if layer_ids is None:
            layer_ids = torch.as_tensor(batch["layers"], device=H_wt.device, dtype=torch.long)
        layer_emb = self.layer_embedding(layer_ids)          # (Lyr, 128)
        layer_emb = layer_emb.view(1, Lyr, 1, TOKEN_DIM)

        K_res = content + pe + layer_emb
        V_res = content
        K_res = K_res.reshape(B, Lyr * A, TOKEN_DIM)
        V_res = V_res.reshape(B, Lyr * A, TOKEN_DIM)
        attn_valid_res = attention_valid.unsqueeze(1).expand(B, Lyr, A).reshape(B, Lyr * A)

        meta_content = self.metadata_encoder(batch["meta_raw"])   # (B, 128) -- no PE/layer add
        K = torch.cat([K_res, meta_content.unsqueeze(1)], dim=1)
        V = torch.cat([V_res, meta_content.unsqueeze(1)], dim=1)
        meta_valid = torch.ones(B, 1, device=H_wt.device)
        valid = torch.cat([attn_valid_res, meta_valid], dim=1)

        z_seq, weights = self.pooling(K, V, valid)
        pred = self.head(z_seq)

        out = {"pred": pred, "z_seq": z_seq}
        if return_extras:
            out.update({
                "attn_weights": weights, "K": K, "V": V, "attention_valid": valid,
                "slot_kind": slot_kind, "meta_token_index": Lyr * A,
                "layer_ids": layer_ids,
            })
        return out

    # ------------------------------------------------------------------
    def freeze_for_stage2(self) -> None:
        """Freeze every Stage 1 parameter and force eval mode permanently.

        After this call, .train(True) on this module (e.g. from an upstream
        Stage 2 model's train() sweep) is a no-op -- see the train() override
        below -- so Stage 1's dropout never silently re-activates.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self._frozen_for_stage2 = True

    def train(self, mode: bool = True):  # type: ignore[override]
        if self._frozen_for_stage2:
            return super().train(False)
        return super().train(mode)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_stage1_model(mode: str, cfg: dict, init_seed: Optional[int] = None) -> Stage1Model:
    return Stage1Model(
        mode=mode,
        d_esm=int(cfg.get("d_esm", DEFAULT_D_ESM)),
        bottleneck_dim=int(cfg.get("bottleneck_dim", DEFAULT_BOTTLENECK_DIM)),
        layers=tuple(cfg.get("layers", [33])),
        head_hidden=int(cfg.get("head_hidden", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
        meta_raw_dim=int(cfg.get("meta_raw_dim", RAW_META_DIM)),
        init_seed=init_seed if init_seed is not None else cfg.get("init_seed"),
    )
