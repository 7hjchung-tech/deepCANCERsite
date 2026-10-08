"""make_stage1_oof.py -- out-of-fold Stage 1 predictions for the TRAIN split.

Why: Stage 2 learns the residual y - y1. On train rows the shipped Stage 1 checkpoint is
in-sample, so its residuals differ from the validation residuals Stage 2 is judged on.
Here every train row gets a y1 from a Stage 1 model that never saw that row's position.

Folds: train anchor positions (`pp`, the shipped split's own grouping unit) are shuffled
with --fold-seed and assigned greedily to K folds, balancing row counts. Fold k's model is
trained on the other K-1 folds with exactly the settings of run_stage1_sweep.run_one
(same resolve_config, init_seed = seed, train-only meta scaler / target scaling, early
stopping on the shipped validation split, patience 10, max 120 epochs). The test split is
never loaded.

Outputs (out-root/fold{k}/: best.pt, history.json, config.json, val_metrics.json) and
out-root/oof_y1.csv with var_id, fold, pp, type, label, y1_oof, y1_full (shipped checkpoint).

    python make_stage1_oof.py --k 5 --seed 44 --out-root runs/stage1_oof/unified_reference_delta/W10/seed44
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
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage1.cache import RawStage1Cache  # noqa: E402
from src.stage1.checkpoint import default_reference, load_checkpoint, save_checkpoint  # noqa: E402
from src.stage1.config import resolve_config  # noqa: E402
from src.stage1.dataset import Stage1Dataset, build_cohort, join_cohort_with_cache, make_collate_fn  # noqa: E402
from src.stage1.engine import evaluate, fit_meta_scaler_from_entries, group_of, train_one_run  # noqa: E402
from src.stage1.metadata import MetaScaler  # noqa: E402
from src.stage1.model import build_stage1_model  # noqa: E402

MODE = "unified_reference_delta"


def assign_folds(entries: list[dict], k: int, seed: int) -> dict:
    """pp -> fold. Positions shuffled, then each goes to the fold with the fewest rows so far."""
    counts = pd.Series([int(e["row"]["pp"]) for e in entries]).value_counts()
    rng = np.random.default_rng(seed)
    order = rng.permutation(counts.index.to_numpy())
    load = np.zeros(k, dtype=int)
    fold_of = {}
    for pp in order:
        f = int(np.argmin(load))
        fold_of[int(pp)] = f
        load[f] += int(counts[pp])
    return fold_of


@torch.no_grad()
def predict(model, entries, cache, cfg, meta_scaler, y_mean, y_std, device) -> np.ndarray:
    ds = Stage1Dataset(entries, cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler)
    ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=make_collate_fn(MODE))
    _, preds = evaluate(model, ld, y_mean, y_std, device)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=44, help="Stage 1 training/init seed (shipped checkpoint: 44)")
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--full-ckpt", default="runs/stage1_v2/unified_reference_delta/W10/shipped_split/seed44/best.pt")
    ap.add_argument("--out-root", default="runs/stage1_oof/unified_reference_delta/W10/seed44")
    ap.add_argument("--manifest", default="data/split_manifest.csv")
    ap.add_argument("--wt-seq", default="data/wt_sequence.txt")
    ap.add_argument("--cache", default="data/stage1/raw_cache.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds to (re)run")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cache = RawStage1Cache.load(args.cache)
    man = pd.read_csv(args.manifest)
    cohort = join_cohort_with_cache(build_cohort(man.to_dict("records"), Path(args.wt_seq).read_text().strip()), cache)
    train = [e for e in cohort.supported if e["row"]["split"] == "train"]
    val = [e for e in cohort.supported if e["row"]["split"] == "val"]
    fold_of = assign_folds(train, args.k, args.fold_seed)
    fold_rows = [fold_of[int(e["row"]["pp"])] for e in train]
    (out_root / "folds.json").write_text(json.dumps({
        "k": args.k, "fold_seed": args.fold_seed, "unit": "pp (anchor position)",
        "rows_per_fold": np.bincount(fold_rows, minlength=args.k).tolist(),
        "positions_per_fold": np.bincount(list(fold_of.values()), minlength=args.k).tolist(),
        "pp_to_fold": {str(k_): v for k_, v in sorted(fold_of.items())}}, indent=2))
    print(f"[oof] rows per fold {np.bincount(fold_rows, minlength=args.k).tolist()}")

    cfg = resolve_config(MODE, window_radius=args.window, layers=[33])
    oof = np.full(len(train), np.nan)
    for k in range(args.k):
        fdir = out_root / f"fold{k}"
        fdir.mkdir(exist_ok=True)
        tr_k = [e for e, f in zip(train, fold_rows) if f != k]
        ho_k = [e for e, f in zip(train, fold_rows) if f == k]
        assert not ({int(e["row"]["pp"]) for e in tr_k} & {int(e["row"]["pp"]) for e in ho_k})
        if (fdir / "best.pt").exists() and (args.folds is None or k not in args.folds):
            ck = load_checkpoint(fdir / "best.pt", map_location=args.device)
            model = build_stage1_model(MODE, ck["cfg"], init_seed=args.seed)
            model.load_state_dict(ck["model_state"] if "model_state" in ck else ck["state_dict"])
            model.to(args.device)
            meta_scaler, y_mean, y_std = MetaScaler.from_state_dict(ck["meta_scaler"]), ck["y_mean"], ck["y_std"]
            print(f"[oof] fold {k}: reusing {fdir / 'best.pt'}")
        else:
            t0 = time.time()
            meta_scaler = fit_meta_scaler_from_entries(tr_k)
            tr_ds = Stage1Dataset(tr_k, cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler)
            va_ds = Stage1Dataset(val, cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler)
            model = build_stage1_model(MODE, cfg, init_seed=args.seed)
            res = train_one_run(model, tr_ds, va_ds, cfg, device=args.device, max_epochs=args.epochs,
                                patience=args.patience, select_on=cfg["select_on"], seed=args.seed, min_delta=0.0,
                                resume_path=str(fdir / "resume.pt"))
            model, y_mean, y_std = res["model"], res["y_mean"], res["y_std"]
            save_checkpoint(fdir / "best.pt", model, cfg, meta_scaler.state_dict(), y_mean, y_std,
                            default_reference(cfg["esm_checkpoint"], cache.wt_hash))
            (fdir / "history.json").write_text(json.dumps({"history": res["history"], "best": res["best"],
                                                           "stop_reason": res["stop_reason"]}, indent=2, default=str))
            (fdir / "config.json").write_text(json.dumps({
                "fold": k, "seed": args.seed, "n_train_rows": len(tr_k), "n_heldout_rows": len(ho_k),
                "resolved_config": cfg, "best_epoch": res["best"]["epoch"], "stop_reason": res["stop_reason"],
                "train_seconds": round(time.time() - t0, 1)}, indent=2, default=str))
            (fdir / "resume.pt").unlink(missing_ok=True)
        va_ld = DataLoader(Stage1Dataset(val, cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler),
                           batch_size=64, shuffle=False, collate_fn=make_collate_fn(MODE))
        vm, _ = evaluate(model, va_ld, y_mean, y_std, args.device, edit_types=[e["edit"].edit_type for e in val])
        (fdir / "val_metrics.json").write_text(json.dumps(vm, indent=2, default=str))
        idx = [i for i, f in enumerate(fold_rows) if f == k]
        oof[idx] = predict(model, ho_k, cache, cfg, meta_scaler, y_mean, y_std, args.device)
        print(f"[oof] fold {k}: val subset {vm['subset']:.4f}, held-out rows {len(idx)}", flush=True)

    # shipped full-train checkpoint on the same train rows (in-sample reference)
    ck = load_checkpoint(args.full_ckpt, map_location=args.device)
    full = build_stage1_model(MODE, ck["cfg"], init_seed=args.seed)
    full.load_state_dict(ck["model_state"] if "model_state" in ck else ck["state_dict"])
    full.to(args.device)
    y1_full = predict(full, train, cache, ck["cfg"], MetaScaler.from_state_dict(ck["meta_scaler"]),
                      ck["y_mean"], ck["y_std"], args.device)
    assert np.isfinite(oof).all()
    df = pd.DataFrame({"var_id": [e["var_id"] for e in train], "fold": fold_rows,
                       "pp": [int(e["row"]["pp"]) for e in train],
                       "type": [group_of(e["edit"].edit_type) for e in train],
                       "label": [float(e["row"]["z_score_D4_D14"]) for e in train],
                       "y1_oof": oof, "y1_full": y1_full})
    df.to_csv(out_root / "oof_y1.csv", index=False)
    print(f"[oof] wrote {out_root / 'oof_y1.csv'}")


if __name__ == "__main__":
    main()
