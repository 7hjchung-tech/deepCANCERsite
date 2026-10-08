"""train_stage2_neighborhood.py -- E0 (Stage1 alone) / E1 (anchor-only correction) /
E2 (anchor + 8 nearest-in-3D-neighbor correction), same fusion model for E1/E2 (only the
structure valid-mask differs).

    python train_stage2_neighborhood.py --condition E0 --out runs/stage2_neighborhood/E0/seed42
    python train_stage2_neighborhood.py --condition E1 --out runs/stage2_neighborhood/E1/seed42
    python train_stage2_neighborhood.py --condition E2 --out runs/stage2_neighborhood/E2/seed42

E1 and E2 do NOT share a Stage1Handle, model, or tokenizer instance (each condition reloads
the Stage1 checkpoint fresh). When --init-from-condition is given (used by the E2 run so it
starts from literally the same weights E1 did), the model/tokenizer state_dict is copied from
that condition's freshly-built (pre-training) init, not from a trained checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage1.engine import group_of, set_all_seeds  # noqa: E402
from src.stage2.diagnostics import DEFAULT_STAGE1, load_setup  # noqa: E402
from src.stage2.neighbor_engine import evaluate, make_loader, train_neighborhood  # noqa: E402
from src.stage2.neighbor_model import NeighborhoodFiLMModel  # noqa: E402
from src.stage2.neighbor_structure import ResidueStructureTokenizer, WTNeighborStore  # noqa: E402
from src.stage2.stage1_adapter import stage1_outputs  # noqa: E402

DEFAULT_CACHE = "data/structure/results/wt_neighbor_cache.npz"
DEFAULT_CONFIG = "configs/stage2/neighborhood.yaml"


def resolve_cfg() -> dict:
    import yaml
    return yaml.safe_load(Path(DEFAULT_CONFIG).read_text())


def dump_val_predictions(val_preds_by_epoch: dict, out_dir: Path) -> None:
    rows = []
    for epoch, d in val_preds_by_epoch.items():
        for i in range(len(d["var_id"])):
            rows.append({"epoch": epoch, "var_id": d["var_id"][i], "label": d["label"][i],
                        "pred": d["pred"][i], "stage1_pred": d["stage1_pred"][i], "group": d["group"][i]})
    pd.DataFrame(rows).to_csv(out_dir / "val_predictions_by_epoch.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=("E0", "E1", "E2"))
    ap.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--init-state-from", default=None,
                    help="path to another condition's out dir; copy its PRE-training init_state.pt "
                         "(not a trained checkpoint) so this run starts from identical weights")
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "history.json").exists():
        raise SystemExit(f"{out} already has results; remove it first")
    out.mkdir(parents=True, exist_ok=True)
    cfg = resolve_cfg()
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs

    st = load_setup(args.stage1_ckpt, args.device, splits=("train", "val"))   # handle reloaded fresh below anyway
    from src.stage1.checkpoint import default_reference
    from src.stage2.stage1_adapter import load_stage1_handle
    ref = default_reference(st.cfg.get("esm_checkpoint", "esm2_t33_650M_UR50D"), st.cache.wt_hash)
    handle = load_stage1_handle(args.stage1_ckpt, ref, args.device)   # fresh handle, not shared with any other condition

    bs = int(cfg["batch_size"])

    if args.condition == "E0":
        loaders = {"val": make_loader(st.by_split["val"], st.cache, st.window, st.layers, handle,
                                      WTNeighborStore(args.cache), "e1", bs, shuffle=False)}
        v = evaluate_stage1_only(handle, loaders["val"])
        (out / "history.json").write_text(json.dumps({
            "history": [{"epoch": 0, "val_stage2_subset": v["subset"], "val_stage1_subset": v["subset"],
                        "val_stage2_spearman": v["spearman"], "val_stage2_rmse": v["rmse"],
                        "val_by_group_spearman": v["by_group"], "val_by_group_n": v["by_group_n"]}],
            "best": {"score": v["subset"], "epoch": 0}, "last_epoch": 0, "stop_reason": "n/a (E0 has no training)"},
            indent=2, default=str))
        (out / "config.json").write_text(json.dumps({"condition": "E0", "note": "Stage1 alone, no Stage2 model"}, indent=2))
        print(f"[neighborhood:E0] Stage1-alone val subset = {v['subset']:.4f}")
        return

    store = WTNeighborStore(args.cache)
    mode = "e1" if args.condition == "E1" else "e2"
    train_anchors = [int(e["row"]["pp"]) for e in st.by_split["train"]]
    fit_positions = store.fit_positions_for_train_anchors(train_anchors)
    (out / "fit_positions.json").write_text(json.dumps({
        "n_train_anchors": len(set(train_anchors)), "n_fit_positions_union": len(fit_positions),
        "rule": "union of every WT position in any slot of any train anchor's 9-slot neighbor set "
                "(deduplicated); same rule and same fit set for E1 and E2",
    }, indent=2))

    loaders = {"train": make_loader(st.by_split["train"], st.cache, st.window, st.layers, handle, store, mode, bs, shuffle=True),
              "val": make_loader(st.by_split["val"], st.cache, st.window, st.layers, handle, store, mode, bs, shuffle=False)}

    set_all_seeds(args.seed)
    tokenizer = ResidueStructureTokenizer(d_s=int(cfg.get("d_s", 32)), n_bins=int(cfg.get("n_bins", 4)))
    tokenizer.fit_preprocessing(store.raw_for_positions(fit_positions))
    # peek one batch to read h_base's real dimension from the checkpoint, not a hardcoded 128
    probe_batch = next(iter(loaders["val"]))
    d_model = stage1_outputs(handle, probe_batch, grad=False)["h_base"].shape[-1]
    model = NeighborhoodFiLMModel(d=int(d_model))

    if args.init_state_from:
        src_dir = Path(args.init_state_from)
        init = torch.load(src_dir / "init_state.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(init["model"])
        tokenizer.load_state_dict(init["tokenizer"])
        print(f"[neighborhood:{args.condition}] copied initial state from {src_dir}/init_state.pt")

    res = train_neighborhood(model=model, tokenizer=tokenizer, handle=handle, loaders=loaders, cfg=cfg,
                             device=args.device, out_dir=out, seed=args.seed)

    torch.save(res["init_state"], out / "init_state.pt")
    torch.save({"model": res["best_state"]["model"], "tokenizer": res["best_state"]["tokenizer"],
                "epoch": res["best"]["epoch"], "d_model": int(d_model), "condition": args.condition,
                "stage1_ckpt": args.stage1_ckpt, "stage1_reference": handle.reference}, out / "best.pt")
    torch.save({"model": res["last_state"]["model"], "tokenizer": res["last_state"]["tokenizer"],
                "epoch": res["last_epoch"], "d_model": int(d_model), "condition": args.condition,
                "stage1_ckpt": args.stage1_ckpt, "stage1_reference": handle.reference}, out / "last.pt")
    (out / "history.json").write_text(json.dumps(
        {"history": res["history"], "best": res["best"], "last_epoch": res["last_epoch"],
         "stop_reason": res["stop_reason"]}, indent=2, default=str))
    pd.DataFrame([{k: v for k, v in h.items() if not isinstance(v, (dict, list))} for h in res["history"]]).to_csv(
        out / "history.csv", index=False)
    dump_val_predictions(res["val_preds_by_epoch"], out)

    n_grad_step1 = sum(res["grad_coverage_step1"].values())
    cov2 = res["grad_coverage_epoch2_step1"]
    n_grad_ep2 = sum(cov2.values()) if cov2 else None
    n_total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in tokenizer.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
        sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    (out / "config.json").write_text(json.dumps({
        "condition": args.condition, "mode": mode, "seed": args.seed, "stage1_ckpt": args.stage1_ckpt,
        "d_model": int(d_model), "cfg": cfg, "best_epoch": res["best"]["epoch"], "last_epoch": res["last_epoch"],
        "stop_reason": res["stop_reason"], "n_total_params": n_total_params, "n_trainable_params": n_trainable,
        "n_params_with_nonzero_grad_step1": n_grad_step1, "n_params_with_nonzero_grad_epoch2_step1": n_grad_ep2,
        "grad_coverage_step1": res["grad_coverage_step1"], "grad_coverage_epoch2_step1": cov2,
        "init_state_from": args.init_state_from,
    }, indent=2, default=str))
    print(f"[neighborhood:{args.condition}] done: best_epoch={res['best']['epoch']} "
          f"val_subset={res['best']['score']:.4f} last_epoch={res['last_epoch']} stop={res['stop_reason']} "
          f"grad_nonzero_step1={n_grad_step1}/{len(res['grad_coverage_step1'])} "
          f"grad_nonzero_epoch2step1={n_grad_ep2}/{len(cov2) if cov2 else 0}")


def evaluate_stage1_only(handle, loader) -> dict:
    import numpy as np
    from src.stage2.neighbor_engine import _summarise
    y, p1, groups = [], [], []
    with torch.no_grad():
        for batch in loader:
            s1 = stage1_outputs(handle, batch, grad=False)
            y.append(batch["label"].numpy()); p1.append(s1["y1"].cpu().numpy())
            groups += [group_of(t) for t in batch["edit_type"]]
    return _summarise(np.concatenate(y), np.concatenate(p1), np.array(groups))


if __name__ == "__main__":
    main()
