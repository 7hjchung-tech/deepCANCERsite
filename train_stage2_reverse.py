"""train_stage2_reverse.py -- validation-only Stage 2 runs (R0 / R1 / R2 reverse direction, and the
forward single/nine models), optionally on out-of-fold Stage 1 residual targets (exploratory).

  --mode r0_struct_mean      structure mean -> token-wise FiLM on Stage 1 content tokens
  --mode r1_seq_query        sequence query -> structure K/V (softmax over 9 tokens) -> token-wise FiLM
  --mode r2_multihead_query  R1 strengthened: multi-head cross-attention + learnable temperature
                             (see src/stage2/reverse.py docstring)
  --mode single_query | nine_query   the forward Stage2Model (tau_init from base.yaml)
  --y1-override oof_y1.csv           train rows use y1_oof (from make_stage1_oof.py) as the residual
                                     base; val rows keep the frozen Stage 1 prediction. K/V are unchanged.

Same frozen Stage 1 checkpoint, tokenizer, data, seed handling, optimizer, loss, early stopping
and checkpoint selection as train_stage2.py (src.stage2.engine.train_stage2 is reused).
Validation only: the test split is never loaded or evaluated. Epoch 0 (no update) is recorded.

    python train_stage2_reverse.py --mode r1_seq_query --out runs/stage2_reverse/r1_seq_query/seed42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage1.engine import set_all_seeds  # noqa: E402
from src.stage2.diagnostics import DEFAULT_STAGE1, load_setup, loader_for  # noqa: E402
from src.stage2.engine import evaluate, train_stage2  # noqa: E402
from src.stage2.model import Stage2Model  # noqa: E402
from src.stage2.schema import QUERY_MODES  # noqa: E402
from src.stage2.reverse import REVERSE_MODES, build_reverse_model  # noqa: E402
from src.stage2.structure import make_qk_tokenizer  # noqa: E402


def module_norms(model, tok) -> dict:
    out = {f"stage2.{n}": float(p.detach().norm()) for n, p in model.named_parameters()}
    out.update({f"tokenizer.{n}": float(p.detach().norm()) for n, p in tok.named_parameters()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=REVERSE_MODES + QUERY_MODES, required=True)
    ap.add_argument("--y1-override", default=None, help="oof_y1.csv from make_stage1_oof.py (train rows only)")
    ap.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "best_stage2.pt").exists():
        raise SystemExit(f"{out} already has results; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    st = load_setup(args.stage1_ckpt, args.device, splits=("train", "val"))
    cfg = dict(st.cfg)
    cfg["query_mode"] = args.mode
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    bs = int(cfg["batch_size"])
    override = None
    if args.y1_override:
        oof = pd.read_csv(args.y1_override)
        train_ids = {e["var_id"] for e in st.by_split["train"]}
        if set(oof["var_id"]) != train_ids:
            raise SystemExit("y1 override must cover exactly the train split")
        override = dict(zip(oof["var_id"], oof["y1_oof"].astype(float)))
    loaders = {"train": loader_for(st, st.by_split["train"], bs, shuffle=True, y1_override=override),
               "val": loader_for(st, st.by_split["val"], bs, shuffle=False)}

    set_all_seeds(args.seed)
    tok = make_qk_tokenizer(cfg)
    tok.fit_preprocessing(st.store.raw([e["var_id"] for e in st.by_split["train"]]))
    model = (build_reverse_model(args.mode, tau_init=cfg.get("tau_init")) if args.mode in REVERSE_MODES
             else Stage2Model(args.mode, tau_init=cfg.get("tau_init")))
    n_params = model.num_trainable_params()
    print(f"[reverse] {args.mode}: stage2 trainable params {n_params:,}; tokenizer "
          f"{sum(p.numel() for p in tok.parameters()):,}")
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    init_tok = {k: v.detach().clone() for k, v in tok.state_dict().items()}

    model.to(args.device); tok.to(args.device)
    ep0 = evaluate(model, tok, st.handle, loaders["val"], args.device)    # no update, no RNG use
    ep0_row = {"epoch": 0, "val_stage2_subset": ep0["stage2"]["subset"], "val_stage1_subset": ep0["stage1"]["subset"],
               "val_stage2_spearman": ep0["stage2"]["spearman"], "val_stage2_rmse": ep0["stage2"]["rmse"],
               "attention_entropy_mean": ep0["attention_entropy_mean"], "delta_abs_mean": ep0["delta_abs_mean"]}
    print(f"[reverse] epoch 0 val subset {ep0_row['val_stage2_subset']:.4f} (stage1 {ep0_row['val_stage1_subset']:.4f})")

    res = train_stage2(stage2=model, tokenizer=tok, handle=st.handle, loaders=loaders, store=st.store, cfg=cfg,
                       device=args.device, out_dir=out, seed=args.seed, train_mode="frozen_stage1",
                       evaluate_test=False)
    best = res["best_state"]
    # movement of every trainable tensor between init and the selected checkpoint (gradient-flow evidence)
    moved = {f"stage2.{k}": float((best["stage2"][k].cpu() - init_state[k]).norm()) for k in init_state}
    moved.update({f"tokenizer.{k}": float((best["tokenizer"][k].float().cpu() - init_tok[k].float()).norm())
                  for k in init_tok if init_tok[k].is_floating_point()})
    torch.save({"stage2": best["stage2"], "tokenizer": best["tokenizer"], "mode": args.mode, "cfg": cfg,
                "stage1_ckpt": args.stage1_ckpt, "stage1_reference": st.handle.reference}, out / "best_stage2.pt")
    hist = [ep0_row] + res["history"]
    (out / "history.json").write_text(json.dumps({"history": hist, "best": res["best"], "stop_reason": res["stop_reason"]},
                                                 indent=2, default=str))
    pd.DataFrame([{k: v for k, v in h.items() if not isinstance(v, (dict, list))} for h in hist]).to_csv(
        out / "history.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "cfg": cfg, "args": vars(args), "trainable_params": {"stage2": n_params,
                                                             "tokenizer": sum(p.numel() for p in tok.parameters())},
        "param_breakdown": {n: p.numel() for n, p in model.named_parameters()},
        "best_epoch": res["best"]["epoch"], "stop_reason": res["stop_reason"], "test_evaluated": False,
        "y1_override": args.y1_override,
        "param_change_init_to_best": moved,
    }, indent=2, default=str))
    print(f"[reverse] done: best_epoch={res['best']['epoch']} val_subset={res['best']['score']:.4f} stop={res['stop_reason']}")


if __name__ == "__main__":
    main()
