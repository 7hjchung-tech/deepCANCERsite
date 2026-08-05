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
    python dump_diff_emb.py --device cuda --batch-size 32     # GPU box

The cache is always saved as CPU tensors regardless of --device, so a dump
produced on a GPU node loads unchanged on a CPU-only machine (DiffEmbedder's
cached path calls torch.load without map_location).
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

from src.embeddings.diff_compute import mag_from_pooled  # noqa: E402
from src.embeddings.diff_pooling import window_pool  # noqa: E402
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
    ap.add_argument("--device", default=None,
                    help="cuda | cpu. Overrides src/config.yaml's esm.device. "
                         "Default: cuda when available, else cpu.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="sweep mode: extract these repr layers. ESM-2 returns every "
                         "requested layer from ONE forward pass, so N layers cost the "
                         "same GPU time as one.")
    ap.add_argument("--windows", type=int, nargs="+", default=None,
                    help="sweep mode: pool with these window radii. Pooling is cheap "
                         "post-processing on hidden states already in memory.")
    ap.add_argument("--concat-sets", nargs="+", default=None,
                    help="sweep mode: comma-separated layer groups to CONCATENATE, e.g. "
                         "--concat-sets 30,33 27,30,33. The pooled WT/MUT vectors are "
                         "joined across layers before the diff, so the cache is "
                         "1280*k wide and train.py sizes the bottleneck to match.")
    ap.add_argument("--pool", nargs="+", default=["uniform"],
                    help="sweep mode: pooling specs. 'uniform', or 'exp:<tau>' for "
                         "exponentially decaying weights exp(-|i-p|/tau) that favour "
                         "residues near the mutation. e.g. --pool uniform exp:2 exp:5")
    ap.add_argument("--out-dir", default="data/sweep",
                    help="sweep mode: writes <out-dir>/diff_emb_<tag>.pt")
    args = ap.parse_args()

    cfg = _load_config()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg["esm"]["device"] = device
    print(f"[dump_diff_emb] device={device}")

    sweep = any(x is not None for x in (args.layers, args.windows, args.concat_sets)) \
        or args.pool != ["uniform"]

    def parse_pool(spec):
        if spec == "uniform":
            return ("uniform", 1.0, "uniform")
        if spec.startswith("exp:"):
            tau = float(spec.split(":", 1)[1])
            return ("exp_decay", tau, f"exp{spec.split(':', 1)[1]}")
        raise SystemExit(f"알 수 없는 --pool 값: {spec} ('uniform' 또는 'exp:<tau>')")

    pools = [parse_pool(s) for s in args.pool]
    windows = args.windows or [int(cfg["pooling"]["window_W"])]
    # Each entry is (layer_tuple, window, mode, tau, tag). A single layer and a
    # concatenated group go through the same code path; only the tuple differs.
    groups = [((ly,), f"L{ly}") for ly in (args.layers or [int(cfg["esm"]["repr_layer"])])]
    for spec in (args.concat_sets or []):
        lys = tuple(int(x) for x in spec.split(","))
        groups.append((lys, "Lcat" + "-".join(str(x) for x in lys)))

    combos = [(lys, w, mode, tau, f"{gtag}_W{w}_{ptag}")
              for lys, gtag in groups for w in windows for mode, tau, ptag in pools]
    layers = sorted({ly for lys, _ in groups for ly in lys})
    if sweep:
        print(f"[dump_diff_emb] sweep: layers={layers} windows={windows} "
              f"pools={[c[4].split('_')[-1] for c in combos[:len(pools)]]} "
              f"-> {len(combos)} caches from one pass")

    wt_seq = Path(args.wt_seq).read_text().strip()

    m = pd.read_csv(args.manifest)
    if args.limit is not None:
        m = m.head(args.limit).copy()
    print(f"[dump_diff_emb] {len(m)} variants to encode")

    encoder = ESMEncoder(cfg["esm"])           # frozen, no LoRA

    # WT forwarded ONCE for the whole run (same sequence for every variant).
    with torch.no_grad():
        H_wt = {ly: h[0] for ly, h in encoder.encode([wt_seq], repr_layers=layers).items()}

    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {c[4]: {} for c in combos}
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
                H_mut = encoder.encode(seqs, repr_layers=layers)
            for i, r in enumerate(group):
                p = int(r["pp"]) - 1     # 1-based pp -> 0-based residue index
                for lys, w, mode, tau, tag in combos:
                    # Pool each layer, then join across layers BEFORE the diff, so a
                    # multi-layer cache uses the same diff/magnitude definition.
                    pw = torch.cat([window_pool(H_wt[ly], p, w, mode, tau) for ly in lys])
                    pm = torch.cat([window_pool(H_mut[ly][i], p, w, mode, tau)
                                    for ly in lys])
                    res = mag_from_pooled(pw, pm)
                    # .cpu() so the cache is device-independent (see module docstring)
                    out[tag][r["var_id"]] = {
                        "diff": res["diff"].cpu(), "mag": res["mag"].cpu()
                    }

        done = min(start + args.batch_size, len(rows))
        print(f"[dump_diff_emb] {done}/{len(rows)} done", end="\r")

    print()
    if not sweep:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out[combos[0][4]], args.out)
        print(f"[dump_diff_emb] saved {len(out[combos[0][4]])} entries -> {args.out}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, d in out.items():
        path = out_dir / f"diff_emb_{tag}.pt"
        torch.save(d, path)
        print(f"[dump_diff_emb] saved {len(d)} entries -> {path}")


if __name__ == "__main__":
    main()
