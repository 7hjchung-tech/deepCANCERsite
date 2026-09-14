"""Raw frozen-ESM hidden-state cache shared by all three Stage 1 model modes.

Only ONE extraction pass is ever done: full WT sequence once, full MUT
sequence per variant (grouped by length, mirroring the WT-broadcast trick in
src/embeddings/diff_embedder.py). All three token builders slice their own
window/content out of these same per-residue hidden states -- nothing here
is pooled or window-clipped, unlike the legacy diff_emb_raw.pt cache, which
is why that cache is NOT reusable for Stage 1 (see README_STAGE1.md).

No real extraction is executed by this task (would require downloading the
650M ESM-2 checkpoint). `build_cache_from_manifest` is fully implemented and
unit-tested, but only ever called in tests/dry-run with a FakeFrozenEncoder
that needs no download.
"""

from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .alignment import wt_sequence_hash  # noqa: E402
from .schema import CACHE_SCHEMA_VERSION  # noqa: E402


class FrozenEncoderProtocol(Protocol):
    def encode(self, sequences: list[str], repr_layers) -> dict[int, Tensor]:
        ...


class FakeFrozenEncoder:
    """Deterministic, download-free stand-in for ESMEncoder used by tests/dry-run.

    Same .encode(sequences, repr_layers) -> {layer: Tensor(N, L, D)} contract
    as src.embeddings.esm_encoder.ESMEncoder, so cache.py's real code path is
    exercised end-to-end without ever loading the 650M checkpoint.

    Values are a deterministic hash of (sequence, position, layer) so that:
      * WT == MUT sequence  -> identical hidden states (delta exactly 0)
      * same residue letter at different positions -> generally different
        vectors (position matters, like a real contextual encoder)
    """

    def __init__(self, hidden_dim: int = 32, seed: int = 0) -> None:
        self.hidden_dim = hidden_dim
        self.seed = seed

    def encode(self, sequences: list[str], repr_layers) -> dict[int, Tensor]:
        layers = [repr_layers] if isinstance(repr_layers, int) else list(repr_layers)
        out: dict[int, Tensor] = {}
        for layer in layers:
            rows = []
            for seq in sequences:
                residue_rows = []
                for i, aa in enumerate(seq):
                    key = f"{aa}|{i}|{layer}|{self.seed}".encode("utf-8")
                    g = torch.Generator().manual_seed(zlib.crc32(key))
                    residue_rows.append(torch.randn(self.hidden_dim, generator=g))
                rows.append(torch.stack(residue_rows) if residue_rows else torch.zeros(0, self.hidden_dim))
            out[layer] = torch.stack(rows)
        return out


@dataclass
class RawStage1Cache:
    schema_version: str
    esm_checkpoint: str
    layers: list[int]
    hidden_dim: int
    wt_seq: str
    wt_hash: str
    coordinate_convention: str
    H_wt: dict[int, Tensor]                      # layer -> (L_wt, D)
    mut_entries: dict[str, dict] = field(default_factory=dict)   # var_id -> {"H_mut": {layer: (L_mut,D)}, "mut_len": int}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": self.schema_version,
                "esm_checkpoint": self.esm_checkpoint,
                "layers": self.layers,
                "hidden_dim": self.hidden_dim,
                "wt_seq": self.wt_seq,
                "wt_hash": self.wt_hash,
                "coordinate_convention": self.coordinate_convention,
                "H_wt": {ly: h.cpu() for ly, h in self.H_wt.items()},
                "mut_entries": {
                    vid: {"H_mut": {ly: h.cpu() for ly, h in e["H_mut"].items()}, "mut_len": e["mut_len"]}
                    for vid, e in self.mut_entries.items()
                },
            },
            path,
        )

    @staticmethod
    def load(path: str | Path) -> "RawStage1Cache":
        raw = torch.load(path, weights_only=False)
        if raw.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: cache schema_version {raw.get('schema_version')!r} != "
                f"expected {CACHE_SCHEMA_VERSION!r}. Rebuild the cache."
            )
        return RawStage1Cache(**raw)

    def get_mut(self, var_id: str, layer: int) -> Tensor:
        return self.mut_entries[var_id]["H_mut"][layer]

    def get_wt(self, layer: int) -> Tensor:
        return self.H_wt[layer]


def build_cache_from_manifest(
    manifest_rows: list[dict],
    wt_seq: str,
    layers: list[int],
    encoder: FrozenEncoderProtocol,
    esm_checkpoint: str = "esm2_t33_650M_UR50D",
    batch_size: int = 8,
) -> RawStage1Cache:
    """Build a RawStage1Cache. `encoder` must expose the ESMEncoder.encode contract.

    manifest_rows: dicts with at least {"var_id", "mut_seq"}.
    """
    layers = sorted(set(layers))
    H_wt = {ly: h[0] for ly, h in encoder.encode([wt_seq], layers).items()}
    hidden_dim = int(next(iter(H_wt.values())).shape[-1])

    mut_entries: dict[str, dict] = {}
    by_len: dict[int, list[dict]] = {}
    for row in manifest_rows:
        by_len.setdefault(len(row["mut_seq"]), []).append(row)

    for _, group in by_len.items():
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            seqs = [r["mut_seq"] for r in batch]
            H_mut_batch = encoder.encode(seqs, layers)
            for i, row in enumerate(batch):
                mut_entries[row["var_id"]] = {
                    "H_mut": {ly: H_mut_batch[ly][i] for ly in layers},
                    "mut_len": len(row["mut_seq"]),
                }

    return RawStage1Cache(
        schema_version=CACHE_SCHEMA_VERSION,
        esm_checkpoint=esm_checkpoint,
        layers=layers,
        hidden_dim=hidden_dim,
        wt_seq=wt_seq,
        wt_hash=wt_sequence_hash(wt_seq),
        coordinate_convention="1-based protein position; sentinel 0 = absent",
        H_wt=H_wt,
        mut_entries=mut_entries,
    )


def main() -> None:
    """Real entry point (NOT executed this session -- requires the 650M ESM-2
    checkpoint). Mirrors dump_diff_emb.py's CLI shape for a Stage 1 raw cache.
    """
    import argparse

    import pandas as pd

    from src.embeddings.esm_encoder import ESMEncoder

    ap = argparse.ArgumentParser(description="Build the Stage 1 raw ESM hidden-state cache.")
    ap.add_argument("--manifest", default="data/split_manifest.csv")
    ap.add_argument("--wt-seq", default="data/wt_sequence.txt")
    ap.add_argument("--out", default="data/stage1/raw_cache.pt")
    ap.add_argument("--layers", type=int, nargs="+", default=[33])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    wt_seq = Path(args.wt_seq).read_text().strip()
    manifest = pd.read_csv(args.manifest).to_dict("records")
    encoder = ESMEncoder({"repr_layer": args.layers, "device": args.device})
    cache = build_cache_from_manifest(manifest, wt_seq, args.layers, encoder, batch_size=args.batch_size)
    cache.save(args.out)
    print(f"[stage1.cache] saved {len(cache.mut_entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
