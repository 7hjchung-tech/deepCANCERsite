"""
dump_diff_emb.py — P2: precompute the WT-difference embedding cache.

For every variant in data/split_manifest.csv, forwards WT + MUT through the
FROZEN ESM-2 backbone (no LoRA -- a frozen encoder gives the same output
every time, so precomputing once is strictly cheaper than recomputing on
every training step) and saves the window-pooled diff vector + magnitude
scalars.

The diff/mag computation itself is shared with the end-to-end LoRA path via
src.embeddings.diff_compute.compute_diff_and_mag -- so a cached-encoder model
(M1/M2) and an end-to-end LoRA model (M3/M4) are computing the exact same
kind of feature, just with different (frozen vs fine-tuned) weights behind it.

OUTPUT
------
    data/diff_emb_raw.pt   {var_id: {"diff": Tensor(1280,), "mag": Tensor(4,)}}

HOW TO RUN
----------
    python dump_diff_emb.py                   # full 5,887-variant dump
    python dump_diff_emb.py --limit 8 --out data/diff_emb_raw_smoketest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.embeddings.diff_compute import compute_diff_and_mag  # noqa: E402
from src.embeddings.esm_encoder import ESMEncoder  # noqa: E402


def _load_config() -> dict:
    with open(_ROOT / "src" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/split_manifest.csv")
    ap.add_argument("--wt-seq", default="data/wt_sequence.txt")
    ap.add_argument("--out", default="data/diff_emb_raw.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N rows (smoke test)")
    args = ap.parse_args()

    cfg = _load_config()
    wt_seq = Path(args.wt_seq).read_text().strip()

    m = pd.read_csv(args.manifest)
    if args.limit is not None:
        m = m.head(args.limit).copy()
    print(f"[dump_diff_emb] {len(m)} variants to encode")

    encoder = ESMEncoder(cfg["esm"])           # frozen, no LoRA
    layer = int(cfg["esm"]["repr_layer"])
    pooling_cfg = cfg["pooling"]

    # WT forwarded ONCE for the whole run (same sequence for every variant).
    with torch.no_grad():
        H_wt = encoder.encode([wt_seq], repr_layers=layer)[layer][0]

    out: dict[str, dict[str, torch.Tensor]] = {}
    rows = m.to_dict("records")
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        # group by mut_seq length (indels change length; a batched forward
        # call needs equal-length inputs within itself)
        by_len: dict[int, list[dict]] = {}
        for r in batch:
            by_len.setdefault(len(r["mut_seq"]), []).append(r)

        for _, group in by_len.items():
            seqs = [r["mut_seq"] for r in group]
            with torch.no_grad():
                H_mut = encoder.encode(seqs, repr_layers=layer)[layer]
            for i, r in enumerate(group):
                p = int(r["pp"]) - 1     # 1-based pp -> 0-based residue index
                res = compute_diff_and_mag(H_wt, H_mut[i], p, pooling_cfg)
                out[r["var_id"]] = {"diff": res["diff"], "mag": res["mag"]}

        done = min(start + args.batch_size, len(rows))
        print(f"[dump_diff_emb] {done}/{len(rows)} done", end="\r")

    print()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[dump_diff_emb] saved {len(out)} entries -> {args.out}")


if __name__ == "__main__":
    main()
