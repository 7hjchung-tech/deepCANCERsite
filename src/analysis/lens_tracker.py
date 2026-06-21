"""Orchestration for per-layer representation convergence tracking.

Intended use: call snapshot(model, epoch) after each fine-tuning epoch to
record how each transformer layer's MUT representation compares to the final
layer (33) representation.

IMPORTANT — diagnostic only
────────────────────────────
This module is a monitoring tool; its output MUST NOT be used as a training
signal or added to the loss function in any way.  The convergence curves it
produces are meaningful only when interpreted alongside task performance metrics
(e.g. LFC validation correlation).  A decreasing layer-distance does not by
itself imply better fine-tuning quality.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# Ensure project root is importable when this module is loaded directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.embeddings.diff_pooling import window_pool  # noqa: E402
from src.analysis.layer_metrics import mahalanobis  # noqa: E402
from src.analysis.covariance import CovarianceStore  # noqa: E402

logger = logging.getLogger(__name__)

_ALL_LAYERS: list[int] = list(range(34))   # 0 .. 33
_OUTPUT_FIELDS = ["epoch", "layer_index", "maha_dist", "mean_norm", "variance", "eff_rank"]


class LensTracker:
    """Track per-layer MUT representation convergence toward layer 33.

    Reproducibility definitions
    ───────────────────────────
    h_l    = window_pool(H_MUT_l, pos, W)   — MUT pooled vector at layer l.
    h_norm = h_33                            — per-variant layer-33 reference.
    Sigma_l is estimated once from the pretrained (LoRA B=0) activations of the
    reference sequence set so that Mahalanobis distances remain comparable
    across all epochs (freeze_at="pretrained").

    Args:
        encoder:             ESMEncoder instance (tokenisation infrastructure
                             is extracted; the live model is NOT stored here).
        reference_variants:  List of (wt_seq, mut_seq, pos) tuples. pos is
                             0-indexed mutation site.  Training/validation data
                             is not required — a handful of synthetic or known
                             RAD51C substitutions is sufficient.
        cfg:                 config["analysis"] dict.
    """

    def __init__(
        self,
        encoder,
        reference_variants: list[tuple],
        cfg: dict,
    ) -> None:
        # Tokenisation infrastructure from the encoder
        self._batch_converter = encoder.batch_converter
        self._device = encoder.device

        self._variants = reference_variants
        self._W    = int(cfg.get("pooling_W", 10))
        self._mode = str(cfg.get("pooling_mode", "uniform"))
        self._tau  = float(cfg.get("pooling_tau", 5.0))

        # Output file
        out_dir = Path(cfg.get("output_dir", "runs/layer_analysis"))
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = cfg.get("output_format", "jsonl")
        self._fmt = fmt
        self._out_path = out_dir / f"layer_metrics.{fmt}"
        if fmt == "csv" and not self._out_path.exists():
            with open(self._out_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=_OUTPUT_FIELDS).writeheader()

        # Covariance store
        self._cov_cfg = cfg.get("cov", {})
        self._cov_store = CovarianceStore(self._cov_cfg)
        self._freeze_at = self._cov_cfg.get("freeze_at", "pretrained")

        # Fit covariance on the pretrained (LoRA B=0) baseline
        if self._freeze_at == "pretrained":
            with torch.no_grad():
                layer_reps = self._collect_residue_reps(encoder.model)
            self._cov_store.fit(layer_reps)

    # ── public API ───────────────────────────────────────────────────────────

    def snapshot(self, model: nn.Module, epoch: int) -> list[dict]:
        """Evaluate layer-33 convergence for the current model state.

        Run model in eval + no_grad mode over reference_variants.  Computes
        Mahalanobis distance and health stats for layers 0..32.

        Call this AFTER the epoch's optimizer step, never inside a loss
        computation.

        Args:
            model: The nn.Module being fine-tuned (may have updated LoRA
                   weights).  Tokenisation uses the stored batch_converter.
            epoch: Current epoch index written to every output record.

        Returns:
            List of 33 dicts (layer_index 0..32):
            {epoch, layer_index, maha_dist, mean_norm, variance, eff_rank}
        """
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                if self._freeze_at == "per_epoch":
                    layer_reps = self._collect_residue_reps(model)
                    self._cov_store.fit(layer_reps)

                # Per-variant, per-layer pooled MUT vectors
                # variant_pooled[v][l] = h_l Tensor(D,)
                variant_pooled: list[dict[int, Tensor]] = [
                    self._pool_variant(model, mut_seq, pos)
                    for _, mut_seq, pos in self._variants
                ]
        finally:
            if was_training:
                model.train()

        records: list[dict] = []
        for l in range(33):
            P_l     = torch.stack([vp[l] for vp in variant_pooled])   # (K, D)
            h_norms = [vp[33] for vp in variant_pooled]

            Sigma_inv_l = self._cov_store.get_inv(l)
            maha_vals = [
                mahalanobis(P_l[i], h_norms[i], Sigma_inv_l)
                for i in range(len(self._variants))
            ]
            maha_mean = float(sum(maha_vals) / len(maha_vals))

            record: dict = {
                "epoch": epoch,
                "layer_index": l,
                "maha_dist": maha_mean,
                **_health_stats(P_l),
            }
            records.append(record)

        self._append_records(records)
        return records

    # ── internal helpers ─────────────────────────────────────────────────────

    def _collect_residue_reps(self, model: nn.Module) -> dict[int, Tensor]:
        """Forward all reference MUT sequences; stack per-layer residue reps.

        Returns {layer_idx: Tensor(N_total_residues, D)} for all 34 layers.
        Used exclusively for covariance fitting.
        """
        per_layer: dict[int, list[Tensor]] = {l: [] for l in _ALL_LAYERS}

        for _, mut_seq, _ in self._variants:
            L = len(mut_seq)
            _, _, tokens = self._batch_converter([("seq", mut_seq)])
            tokens = tokens.to(self._device)
            reps = model(tokens, repr_layers=_ALL_LAYERS)["representations"]
            for l in _ALL_LAYERS:
                # (1, L+2, D) → strip BOS/EOS → (L, D)
                per_layer[l].append(reps[l][0, 1 : L + 1, :].cpu())

        return {l: torch.cat(per_layer[l], dim=0) for l in _ALL_LAYERS}

    def _pool_variant(
        self, model: nn.Module, mut_seq: str, pos: int
    ) -> dict[int, Tensor]:
        """Forward one MUT sequence; return {layer: pooled (D,)} for all layers."""
        L = len(mut_seq)
        _, _, tokens = self._batch_converter([("seq", mut_seq)])
        tokens = tokens.to(self._device)
        reps = model(tokens, repr_layers=_ALL_LAYERS)["representations"]

        pooled: dict[int, Tensor] = {}
        for l in _ALL_LAYERS:
            H_l = reps[l][0, 1 : L + 1, :]                     # (L, D)
            pooled[l] = window_pool(H_l, pos, self._W, self._mode, self._tau).cpu()
        return pooled

    def _append_records(self, records: list[dict]) -> None:
        if self._fmt == "jsonl":
            with open(self._out_path, "a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
        else:
            with open(self._out_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_OUTPUT_FIELDS)
                writer.writerows(records)


# ── standalone health-stat helper ────────────────────────────────────────────

def _health_stats(P: Tensor) -> dict:
    """Representation health statistics from K stacked pooled vectors.

    Args:
        P: (K, D) tensor of pooled MUT representations for one layer.

    Returns:
        mean_norm: Mean ‖p_i‖ across K variants.
        variance:  Trace of element-wise sample covariance (correction=0).
                   Equals P.var(dim=0, correction=0).sum() — safe for K=1 (→ 0.0).
        eff_rank:  Participation ratio (Σs²)² / Σs⁴ from SVD of centred P.
                   Ranges from 1 (rank-1 collapse) to K or D.  Returns 1.0 for K < 2.
    """
    mean_norm = float(P.float().norm(dim=1).mean())

    if P.shape[0] < 2:
        return {"mean_norm": mean_norm, "variance": 0.0, "eff_rank": 1.0}

    variance = float(P.float().var(dim=0, correction=0).sum())

    H = P.float() - P.float().mean(dim=0)
    s = torch.linalg.svdvals(H)
    s2 = s ** 2
    denom = float((s2 ** 2).sum())
    eff_rank = float((s2.sum() ** 2) / denom) if denom > 1e-12 else 1.0

    return {"mean_norm": mean_norm, "variance": variance, "eff_rank": eff_rank}
