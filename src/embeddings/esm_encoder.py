"""ESM-2 encoder with optional LoRA adaptation.

[1a] Loads esm2_t33_650M_UR50D with all parameters frozen and exposes a
     single `encode()` method that tokenises protein sequences and returns
     per-residue hidden states for the requested transformer layers.

[1b] Optionally injects LoRA (rank-r update on Q and V projections of every
     TransformerLayer).  Pass lora_cfg=config["lora"] to activate.
     LoRA parameters (lora_A, lora_B) are the only trainable tensors.

BOS and EOS tokens are stripped so that output index i corresponds exactly
to residue i (0-indexed) in the input sequence.

Assumption: all sequences in a single call must be the same length
(the natural case for WT / single-substitution MUT pairs).

Note: encode() is decorated with @torch.no_grad() and is intended for
inference.  When training LoRA, call self.model directly inside a grad context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import torch
import yaml

# Ensure deepRAD51C/ is importable when this file is run as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import esm  # noqa: E402  (comes after sys.path fix)


def _load_config() -> dict:
    path = Path(__file__).parent.parent / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ESMEncoder:
    """ESM-2 encoder with frozen backbone and optional LoRA adaptation.

    Args:
        cfg:      config["esm"] dict — device, repr_layer.
        lora_cfg: config["lora"] dict — enabled, rank, alpha, target_layers.
                  None (default) → fully frozen, no LoRA.
    """

    def __init__(self, cfg: dict, lora_cfg: Optional[dict] = None) -> None:
        self.device = torch.device(cfg.get("device", "cpu"))

        default_rl = cfg.get("repr_layer", 33)
        self._default_layers: list[int] = (
            [default_rl] if isinstance(default_rl, int) else list(default_rl)
        )

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().to(self.device)
        for param in model.parameters():
            param.requires_grad_(False)

        # LoRA injection (after freeze so only lora_A/lora_B are trainable)
        if lora_cfg and lora_cfg.get("enabled", False):
            from .lora import inject_lora
            inject_lora(
                model,
                rank=int(lora_cfg.get("rank", 8)),
                alpha=float(lora_cfg.get("alpha", 16.0)),
                target_layers=lora_cfg.get("target_layers", None),
            )

        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    @torch.no_grad()
    def encode(
        self,
        sequences: list[str],
        repr_layers: Union[int, list[int], None] = None,
    ) -> dict[int, torch.Tensor]:
        """Tokenise and forward-pass sequences through frozen ESM-2.

        Args:
            sequences: Protein sequences (equal-length). Index 0 → WT,
                       index 1 → MUT in the typical two-sequence call.
            repr_layers: Layer indices to extract.
                         0  = token embedding (before first transformer layer)
                         1-33 = transformer layer outputs
                         None → use cfg default (typically [33]).
                         Pass list(range(34)) to extract all layers at once.

        Returns:
            {layer_idx: Tensor(N, L, 1280)}
            N = len(sequences), L = sequence length (BOS/EOS stripped).
        """
        if repr_layers is None:
            layers = self._default_layers
        elif isinstance(repr_layers, int):
            layers = [repr_layers]
        else:
            layers = list(repr_layers)

        L = len(sequences[0])
        data = [(f"seq_{i}", s) for i, s in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        result = self.model(tokens, repr_layers=layers)

        # tokens layout: [BOS, res_0, ..., res_{L-1}, EOS]
        # slice [:, 1:L+1, :] → residues 0..L-1, shape (N, L, 1280)
        return {
            layer: result["representations"][layer][:, 1 : L + 1, :]
            for layer in layers
        }


if __name__ == "__main__":
    from src.embeddings.lora import inject_lora, lora_param_count

    cfg = _load_config()
    wt  = "MKTAYIAKQR"   # 10-residue dummy
    mut = "MKTAYIAKQK"   # R→K at position 9 (0-indexed)

    # ── [1a] frozen encoder ──────────────────────────────────────────────────
    encoder = ESMEncoder(cfg["esm"])
    reps = encoder.encode([wt, mut], repr_layers=[0, 33])
    for layer, h in sorted(reps.items()):
        print(f"[1a] layer {layer:2d}: {tuple(h.shape)}")   # (2, 10, 1280)

    reps_base_33 = reps[33].detach().clone()

    # ── [1b] inject LoRA into the already-loaded frozen model ────────────────
    inject_lora(encoder.model, rank=8, alpha=16.0)

    reps_lora = encoder.encode([wt, mut], repr_layers=[33])
    print(f"\n[1b] LoRA output shape: {tuple(reps_lora[33].shape)}")

    n_lora  = lora_param_count(encoder.model)
    n_total = sum(p.numel() for p in encoder.model.parameters())
    print(f"[1b] Trainable LoRA params : {n_lora:,}")
    print(f"[1b] Total params          : {n_total:,}")
    print(f"[1b] LoRA ratio            : {100 * n_lora / n_total:.3f}%")

    # B=0 at init → LoRA adds nothing → output must match pre-injection
    max_diff = (reps_lora[33] - reps_base_33).abs().max().item()
    print(f"[1b] Max diff vs frozen    : {max_diff:.2e}  (expect ~0)")
    assert max_diff < 1e-5, "LoRA changed output at init — B is not zero!"
    print("encode + LoRA OK")
