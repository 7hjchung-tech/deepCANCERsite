"""train_stage2_r1_joint_ablation.py -- R1 fusion x {frozen/joint Stage1} x {real/fixed structure}.

Four conditions, same Stage1 checkpoint (seed44 W10), same Stage2 seed (42), same split and
(per condition type) the same batch order:

    A  Stage1 frozen,       real structure     (= the existing R1 run, frozen_stage1)
    B  Stage1 jointly fit,  real structure      (joint_l2sp, content_builder+pooling+head unfrozen)
    C  Stage1 jointly fit,  FIXED structure     (same unfreeze; structure replaced by train median/mode)
    D  Stage1 frozen,       FIXED structure

Joint training (B/C) unfreezes content_builder/pooling/head only (ESM and the rest of Stage1
stay frozen), at lr = Stage2 lr * stage1_lr_ratio (0.1, base.yaml), anchored by
lambda_sp * mean((theta-theta_ref)**2) with lambda_sp given on the command line (default 1.0,
reduction='mean' -- see src/stage2/engine.l2sp_penalty; the shipped default elsewhere in this
repo is reduction='sum', untuned lambda_sp=0.0, i.e. effectively off).

Each condition is reseeded (src.stage1.engine.set_all_seeds) right before its own tokenizer
fit / model build, so all four start from the same Stage2 init and the same train DataLoader
shuffle order. Stage2's residual head is zero-init (pred(epoch0) == y1 == Stage1). Epoch 0 is
evaluated and is an explicit candidate for "best": if no later epoch beats it (validation
subset Spearman, same selection rule, strict >), epoch 0 wins and the saved "best" checkpoint
is the untouched initial state. Both best and last-epoch checkpoints are saved (last epoch
weights were not previously tracked by src.stage2.engine.train_stage2; this run adds that).

Real ESM features are never re-run (frozen, cached raw_cache.pt); forward_batch always
recomputes Stage1's content_builder/pooling from those cached raw hidden states on every call
(no projected-K/V cache), so when Stage1 submodules are unfrozen the autograd graph already
reaches them -- see stage1_outputs()/forward_batch() in src/stage2/engine.py.

No OOF, no ESM fine-tuning, no sweep. Validation only; test is never loaded.

    python train_stage2_r1_joint_ablation.py --out-root runs/stage2_joint_ablation/seed42
    python train_stage2_r1_joint_ablation.py --conditions B C --lambda-sp 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage1.checkpoint import default_reference  # noqa: E402
from src.stage1.engine import set_all_seeds  # noqa: E402
from src.stage2.diagnostics import DEFAULT_STAGE1, load_setup, loader_for  # noqa: E402
from src.stage2.engine import evaluate, train_stage2, unfreeze_stage1  # noqa: E402
from src.stage2.reverse import ReverseFiLMModel  # noqa: E402
from src.stage2.stage1_adapter import load_stage1_handle  # noqa: E402
from src.stage2.structure import make_fixed_structure_store, make_qk_tokenizer  # noqa: E402

CONDITIONS = {
    "A": {"stage1": "frozen", "structure": "real"},
    "B": {"stage1": "joint", "structure": "real"},
    "C": {"stage1": "joint", "structure": "fixed"},
    "D": {"stage1": "frozen", "structure": "fixed"},
}
UNFREEZE = ["content_builder", "pooling", "head"]


def state_dist(a: dict, b: dict) -> float:
    if not a:
        return 0.0
    return float(np.sqrt(sum(float((a[k].float() - b[k].float()).pow(2).sum()) for k in a)))


def run_condition(cond: str, args, st) -> dict:
    spec = CONDITIONS[cond]
    out_dir = Path(args.out_root) / cond
    if (out_dir / "best_stage2.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already has results; pass --overwrite or remove it first")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(st.cfg)
    cfg["query_mode"] = "r1_seq_query"
    cfg["unfreeze"] = UNFREEZE
    cfg["lambda_sp"] = args.lambda_sp
    cfg["lambda_sp_reduction"] = args.lambda_sp_reduction
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    bs = int(cfg["batch_size"])
    train_mode = "joint_l2sp" if spec["stage1"] == "joint" else "frozen_stage1"
    unfreeze = UNFREEZE if train_mode == "joint_l2sp" else None

    # A fresh Stage1Handle per condition: B/C mutate Stage1's weights in place during joint
    # training, and st is shared across all four conditions in main() -- reusing st.handle
    # across conditions would leak one condition's trained-on Stage1 weights into the next
    # condition's "frozen checkpoint" starting point. This is cheap (just the small .pt file,
    # no ESM, no cache reload).
    ref = default_reference(st.cfg.get("esm_checkpoint", "esm2_t33_650M_UR50D"), st.cache.wt_hash)
    handle = load_stage1_handle(args.stage1_ckpt, ref, args.device)

    train_ids = [e["var_id"] for e in st.by_split["train"]]
    store = st.store if spec["structure"] == "real" else make_fixed_structure_store(st.store, train_ids)

    set_all_seeds(args.seed)         # same seed -> same train DataLoader shuffle order every condition
    loaders = {"train": loader_for(st, st.by_split["train"], bs, shuffle=True, store_override=store, handle_override=handle),
               "val": loader_for(st, st.by_split["val"], bs, shuffle=False, store_override=store, handle_override=handle)}

    set_all_seeds(args.seed)         # reseed so model init matches across conditions regardless of loader setup
    tok = make_qk_tokenizer(cfg)
    tok.fit_preprocessing(st.store.raw(train_ids))     # ALWAYS fit on the real distribution (§ docstring above)
    model = ReverseFiLMModel("r1_seq_query")
    model.to(args.device); tok.to(args.device)
    init_state = {"stage2": {k: v.detach().clone() for k, v in model.state_dict().items()},
                 "tokenizer": {k: v.detach().clone() for k, v in tok.state_dict().items()}}

    stage1_init = {}
    if train_mode == "joint_l2sp":
        named0 = unfreeze_stage1(handle, unfreeze)
        stage1_init = {n: p.detach().clone() for n, p in named0}

    ep0 = evaluate(model, tok, handle, loaders["val"], args.device)
    score0 = float(ep0["stage2"]["subset"])
    ep0_row = {"epoch": 0, "val_stage2_subset": score0, "val_stage1_subset": ep0["stage1"]["subset"],
               "val_stage2_spearman": ep0["stage2"]["spearman"], "val_stage2_rmse": ep0["stage2"]["rmse"],
               "val_stage2_mae": ep0["stage2"]["mae"], "val_by_group_spearman": ep0["stage2"]["by_group"],
               "val_by_group_n": ep0["stage2"]["by_group_n"], "attention_entropy_mean": ep0["attention_entropy_mean"],
               "delta_abs_mean": ep0["delta_abs_mean"]}
    print(f"[joint-ablation:{cond}] epoch0 val subset {score0:.4f} (stage1 {ep0_row['val_stage1_subset']:.4f})",
          flush=True)

    t0 = time.time()
    res = train_stage2(stage2=model, tokenizer=tok, handle=handle, loaders=loaders, store=store, cfg=cfg,
                       device=args.device, out_dir=out_dir, seed=args.seed, train_mode=train_mode,
                       unfreeze=unfreeze, init_state=init_state, evaluate_test=False)
    train_seconds = round(time.time() - t0, 1)

    # epoch 0 is an explicit candidate for "best" (strict >, matching min_delta=0's convention)
    if score0 > res["best"]["score"]:
        best = {"score": score0, "epoch": 0}
        best_state = {"stage2": init_state["stage2"], "tokenizer": init_state["tokenizer"],
                      "stage1_unfrozen": stage1_init}
        epoch0_won = True
    else:
        best, best_state, epoch0_won = res["best"], res["best_state"], False

    last_state = res["last_state"]
    last_epoch = res["history"][-1]["epoch"] if res["history"] else 0
    hist = [ep0_row] + res["history"]

    torch.save({"stage2": best_state["stage2"], "tokenizer": best_state["tokenizer"], "mode": "r1_seq_query",
                "condition": cond, "epoch": best["epoch"], "epoch0_won": epoch0_won, "cfg": cfg,
                "stage1_ckpt": args.stage1_ckpt, "stage1_unfrozen": best_state["stage1_unfrozen"],
                "stage1_reference": handle.reference}, out_dir / "best_stage2.pt")
    torch.save({"stage2": last_state["stage2"], "tokenizer": last_state["tokenizer"], "mode": "r1_seq_query",
                "condition": cond, "epoch": last_epoch, "cfg": cfg, "stage1_ckpt": args.stage1_ckpt,
                "stage1_unfrozen": last_state["stage1_unfrozen"], "stage1_reference": handle.reference},
               out_dir / "last_stage2.pt")
    if train_mode == "joint_l2sp":
        torch.save({"stage1_init": stage1_init, "unfreeze": unfreeze}, out_dir / "stage1_init.pt")

    (out_dir / "history.json").write_text(json.dumps(
        {"history": hist, "best": best, "epoch0_won": epoch0_won, "stop_reason": res["stop_reason"],
         "last_epoch": last_epoch}, indent=2, default=str))
    pd.DataFrame([{k: v for k, v in h.items() if not isinstance(v, (dict, list))} for h in hist]).to_csv(
        out_dir / "history.csv", index=False)

    stage1_change_best = state_dist(best_state["stage1_unfrozen"], stage1_init) if train_mode == "joint_l2sp" else None
    stage1_change_last = state_dist(last_state["stage1_unfrozen"], stage1_init) if train_mode == "joint_l2sp" else None
    n_s1_unfrozen = sum(p.numel() for _, p in res["stage1_named"]) if res["stage1_named"] else 0
    (out_dir / "config.json").write_text(json.dumps({
        "condition": cond, "spec": spec, "seed": args.seed, "stage1_ckpt": args.stage1_ckpt,
        "unfreeze": unfreeze, "lambda_sp": args.lambda_sp, "lambda_sp_reduction": args.lambda_sp_reduction,
        "stage1_lr_ratio": cfg["stage1_lr_ratio"], "best_epoch": best["epoch"], "epoch0_won": epoch0_won,
        "last_epoch": last_epoch, "stop_reason": res["stop_reason"], "train_seconds": train_seconds,
        "n_stage1_unfrozen_params": n_s1_unfrozen,
        "stage1_weight_change_l2_best_vs_init": stage1_change_best,
        "stage1_weight_change_l2_last_vs_init": stage1_change_last,
        "trainable_params": {"stage2": model.num_trainable_params(), "tokenizer": sum(p.numel() for p in tok.parameters())},
    }, indent=2, default=str))
    print(f"[joint-ablation:{cond}] done: best_epoch={best['epoch']} (epoch0_won={epoch0_won}) "
          f"val_subset={best['score']:.4f} last_epoch={last_epoch} stop={res['stop_reason']} "
          f"s1_change(best)={stage1_change_best}", flush=True)
    return {"condition": cond, "best": best, "last_epoch": last_epoch}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1)
    ap.add_argument("--out-root", default="runs/stage2_joint_ablation/seed42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--lambda-sp", type=float, default=1.0)
    ap.add_argument("--lambda-sp-reduction", default="mean", choices=("sum", "mean"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    st = load_setup(args.stage1_ckpt, args.device, splits=("train", "val"))
    for cond in args.conditions:
        run_condition(cond, args, st)


if __name__ == "__main__":
    main()
