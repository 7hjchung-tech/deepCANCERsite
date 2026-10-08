"""train_stage2.py -- Stage 2 entry point (structure tokens + cross-attention + FiLM).

Compares two query designs on top of a frozen Stage 1 unified_reference_delta:
    --query-mode single_query   one query from the averaged structure tokens
    --query-mode nine_query     one query per structure token, then mean-pooled

Real training needs the learned StructureTokenizer (--tokenizer module:factory).
It is checked BEFORE the large cache is loaded, so a missing tokenizer fails fast.
--synthetic-tokenizer is for smoke tests only and its results are not valid.

Examples:
    python train_stage2.py --query-mode single_query --stage1-ckpt <best.pt> --tokenizer pkg.mod:make
    python train_stage2.py --query-mode nine_query --stage1-ckpt <best.pt> --synthetic-tokenizer --smoke-n 64 --max-epochs 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage1.cache import RawStage1Cache  # noqa: E402
from src.stage1.checkpoint import default_reference  # noqa: E402
from src.stage1.dataset import build_cohort, join_cohort_with_cache  # noqa: E402
from src.stage1.engine import set_all_seeds  # noqa: E402
from src.stage2.engine import make_loader, train_stage2  # noqa: E402
from src.stage2.model import Stage2Model  # noqa: E402
from src.stage2.schema import QUERY_MODES, TRAIN_MODES  # noqa: E402
from src.stage2.stage1_adapter import load_stage1_handle  # noqa: E402
from src.stage2.structure import load_structure_store, load_tokenizer  # noqa: E402
from src.stage2.synthetic import make_synthetic_tokenizer  # noqa: E402

BASE_CONFIG = _ROOT / "configs" / "stage2" / "base.yaml"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-mode", choices=QUERY_MODES, required=True)
    ap.add_argument("--train-mode", choices=TRAIN_MODES, default="frozen_stage1")
    ap.add_argument("--stage1-ckpt", required=True, help="explicit Stage 1 best.pt path (no automatic selection)")
    ap.add_argument("--init-stage2-ckpt", default=None, help="joint_l2sp only: frozen_stage1 Stage 2 checkpoint")
    ap.add_argument("--tokenizer", default="src.stage2.structure:make_qk_tokenizer",
                    help="module:factory returning a StructureTokenizer (default: Qk tokenizer)")
    ap.add_argument("--synthetic-tokenizer", action="store_true", help="SMOKE TEST ONLY")
    ap.add_argument("--structure-table", default="data/structure/results/rad51c_struct_features.csv")
    ap.add_argument("--manifest", default="data/split_manifest.csv")
    ap.add_argument("--wt-seq", default="data/wt_sequence.txt")
    ap.add_argument("--cache", default="data/stage1/raw_cache.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--unfreeze", nargs="+", default=None, help="joint_l2sp: Stage 1 submodules to unfreeze")
    ap.add_argument("--lambda-sp", type=float, default=None)
    ap.add_argument("--smoke-n", type=int, default=None, help="SMOKE TEST ONLY: keep N entries per split")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg["query_mode"] = args.query_mode
    cfg["train_mode"] = args.train_mode
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.lambda_sp is not None:
        cfg["lambda_sp"] = args.lambda_sp

    if args.synthetic_tokenizer:
        if args.smoke_n is None:
            raise SystemExit("--synthetic-tokenizer is allowed only together with --smoke-n (smoke tests).")
        tokenizer = make_synthetic_tokenizer(cfg)
        print("[stage2] WARNING: synthetic smoke tokenizer in use -- results are not Stage 2 performance.")
    else:
        try:
            tokenizer = load_tokenizer(args.tokenizer, cfg)
        except Exception as e:
            raise SystemExit(f"[stage2] cannot start: {e}") from e

    cache = RawStage1Cache.load(args.cache)
    manifest = pd.read_csv(args.manifest)
    wt_seq = Path(args.wt_seq).read_text().strip()
    cohort = join_cohort_with_cache(build_cohort(manifest.to_dict("records"), wt_seq), cache)
    by_split = {"train": [], "val": [], "test": []}
    for e in cohort.supported:
        by_split[e["row"]["split"]].append(e)
    manifest_split = {e["var_id"]: e["row"]["split"] for e in cohort.supported}
    store = load_structure_store(args.structure_table, [e["var_id"] for e in cohort.supported], manifest_split)

    expected_ref = default_reference(cfg.get("esm_checkpoint", "esm2_t33_650M_UR50D"), cache.wt_hash)
    handle = load_stage1_handle(args.stage1_ckpt, expected_ref, args.device)
    window = int(handle.cfg["window_radius"])
    layers = list(handle.cfg["layers"])
    print(f"[stage2] stage1 ckpt={args.stage1_ckpt} window={window} layers={layers}")

    if args.smoke_n is not None:
        for k in by_split:
            by_split[k] = by_split[k][: args.smoke_n]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = {
        k: make_loader(v, cache, window, layers, handle, store, "unified_reference_delta",
                       batch_size=int(cfg["batch_size"]), shuffle=(k == "train"))
        for k, v in by_split.items()
    }
    print(f"[stage2] split sizes: {[(k, len(v)) for k, v in by_split.items()]}")

    device = args.device
    set_all_seeds(args.seed)
    tokenizer.fit_preprocessing(store.raw([e["var_id"] for e in by_split["train"]]))
    stage2 = Stage2Model(args.query_mode, tau_init=cfg.get("tau_init"))
    print(f"[stage2] stage2 trainable params (head modules): {stage2.num_trainable_params():,}")

    init_state = None
    if args.init_stage2_ckpt:
        if args.train_mode != "joint_l2sp":
            raise SystemExit("--init-stage2-ckpt is only meaningful with --train-mode joint_l2sp")
        init = torch.load(args.init_stage2_ckpt, map_location=device, weights_only=False)
        if init["query_mode"] != args.query_mode:
            raise SystemExit(f"init checkpoint query_mode={init['query_mode']} != {args.query_mode}")
        init_state = {"stage2": init["stage2"], "tokenizer": init["tokenizer"]}

    result = train_stage2(
        stage2=stage2, tokenizer=tokenizer, handle=handle, loaders=loaders, store=store, cfg=cfg,
        device=device, out_dir=out_dir, seed=args.seed, train_mode=args.train_mode,
        unfreeze=args.unfreeze, init_state=init_state,
    )

    best = result["best_state"]
    torch.save({
        "stage2": best["stage2"], "tokenizer": best["tokenizer"], "query_mode": args.query_mode,
        "train_mode": args.train_mode, "cfg": cfg, "stage1_ckpt": args.stage1_ckpt,
        "stage1_reference": handle.reference, "stage1_unfrozen": best["stage1_unfrozen"],
        "y_mean": handle.y_mean, "y_std": handle.y_std,
    }, out_dir / "best_stage2.pt")

    hist = result["history"]
    (out_dir / "history.json").write_text(json.dumps({"history": hist, "best": result["best"],
                                                       "stop_reason": result["stop_reason"]}, indent=2, default=str))
    pd.DataFrame([{k: v for k, v in h.items() if not isinstance(v, dict)} for h in hist]).to_csv(
        out_dir / "history.csv", index=False)
    test = {k: v for k, v in result["test"].items() if k not in ("preds_stage2", "labels", "groups")}
    (out_dir / "test_metrics.json").write_text(json.dumps(test, indent=2, default=str))
    n_tok = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    n_s1 = sum(p.numel() for _, p in result["stage1_named"])
    (out_dir / "config.json").write_text(json.dumps({
        "cfg": cfg, "args": vars(args), "seed": args.seed,
        "trainable_params": {"stage2_head": stage2.num_trainable_params(), "tokenizer": n_tok,
                              "stage1_unfrozen": n_s1},
        "stop_reason": result["stop_reason"], "best_epoch": result["best"]["epoch"],
    }, indent=2, default=str))
    print(f"[stage2] done: best_epoch={result['best']['epoch']} stop={result['stop_reason']} "
          f"test_stage2_subset={test['stage2']['subset']:.4f} test_stage1_subset={test['stage1']['subset']:.4f}")
    print(f"[stage2] saved -> {out_dir}")


if __name__ == "__main__":
    main()
