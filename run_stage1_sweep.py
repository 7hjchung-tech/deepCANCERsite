"""run_stage1_sweep.py -- Stage 1 multi-seed sweep runner.

Runs (model x window x seed) combinations sequentially on ONE GPU, reusing
train_stage1.py's own building blocks (src/stage1/{cache,dataset,config,model,
engine,checkpoint}.py) with NO changes to split loading, target normalization,
metric implementation, or model architecture. The only engine.py changes this
depends on are additive (see its docstrings): seeding, min_delta, richer
per-epoch history, last/fixed-budget checkpoint snapshots, and resume support.

Why a separate script instead of train_stage1.py --execute-plan: that flag is
deliberately hard-refused (see cmd_plan / test_execute_plan_flag_is_refused_
this_session) as a leftover guard from when real training was out of scope.
This script does not touch that flag or its guard. It also loads the ~11GB
raw cache ONCE for the whole sweep instead of once per run (see cache.py's
mmap patch + this session's own timing: an eager per-process reload cost
~37 min, ~11 min even with mmap -- paid 45 times that would dominate total
wall-clock and make the speed comparison meaningless).

Usage:
    python run_stage1_sweep.py --dry-run
    python run_stage1_sweep.py --dry-run --models paired_delta --windows 5 --seeds 42
    python run_stage1_sweep.py --device cuda                       # full 45-run sweep
    python run_stage1_sweep.py --device cuda --models paired_delta --windows 10 --seeds 42 43

Safe to re-run: completed runs (history.json + test_metrics.json + best.pt +
last.pt all present) are skipped; a run that has a resume.pt but is not
complete is resumed from its last saved epoch; everything else starts fresh.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage1.cache import RawStage1Cache
from src.stage1.checkpoint import default_reference, save_checkpoint
from src.stage1.config import resolve_config
from src.stage1.dataset import Stage1Dataset, build_cohort, join_cohort_with_cache, make_collate_fn
from src.stage1.engine import evaluate, fit_meta_scaler_from_entries, train_one_run
from src.stage1.model import build_stage1_model
from src.stage1.schema import MODEL_MODES
from torch.utils.data import DataLoader

DEFAULT_MODELS = list(MODEL_MODES)
DEFAULT_WINDOWS = [5, 10, 20]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_CHECKPOINT_EPOCHS = {10, 20, 30, 40, 60, 80}
DEFAULT_OUT_ROOT = "runs/stage1_v2"
LEGACY_ROOT = "runs/stage1"  # never written to by this script


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
class Tee:
    """Writes to both the real stream and a per-run log file."""

    def __init__(self, real, log_fh):
        self.real, self.log_fh = real, log_fh

    def write(self, s):
        self.real.write(s)
        self.log_fh.write(s)

    def flush(self):
        self.real.flush()
        self.log_fh.flush()


def git_commit_hash() -> dict:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT, capture_output=True, text=True, timeout=10)
        return {"commit": h.stdout.strip(), "dirty": bool(dirty.stdout.strip()),
                "dirty_files": dirty.stdout.strip().splitlines()}
    except Exception as e:  # pragma: no cover -- best-effort provenance only
        return {"commit": None, "error": str(e)}


def gpu_info() -> dict:
    info = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["cuda_version"] = torch.version.cuda
    return info


def legacy_reuse_decision(model: str, window: int, seed: int) -> dict:
    """Programmatically check whether an existing runs/stage1/.../seed<seed>
    result satisfies THIS sweep's schema (richer per-epoch history + seeded
    shuffle/dropout). Never returns True for a legacy run -- kept as a real
    check (not a hardcoded refusal) so the reasons are auditable per run.
    """
    legacy_dir = Path(LEGACY_ROOT) / model / f"W{window}" / "shipped_split" / f"seed{seed}"
    metrics_path = legacy_dir / "metrics.json"
    if not metrics_path.exists():
        return {"reusable": False, "path": str(legacy_dir), "reason": "no legacy metrics.json"}

    required_epoch_fields = {"lr", "patience_counter", "best_epoch_so_far", "best_score_so_far",
                              "monitor_metric", "val_by_group_n"}
    try:
        old = json.loads(metrics_path.read_text())
        old_epoch0 = old["history"][0] if old.get("history") else {}
    except Exception as e:
        return {"reusable": False, "path": str(legacy_dir), "reason": f"unreadable: {e}"}

    missing_fields = sorted(required_epoch_fields - set(old_epoch0.keys()))
    reasons = []
    if missing_fields:
        reasons.append(f"legacy history entries are missing required fields: {missing_fields}")
    reasons.append(
        "legacy run predates engine.py's seeding fix -- its DataLoader shuffle order was drawn from "
        "whatever the ambient global torch RNG state was (see set_all_seeds' docstring), not from "
        "--seed, so it is not a reproducible/comparable sample for this seed-variance sweep"
    )
    return {"reusable": False, "path": str(legacy_dir), "reasons": reasons}


def run_status(out_dir: Path) -> str:
    complete = all((out_dir / f).exists() for f in
                   ("history.json", "test_metrics.json", "best.pt", "last.pt"))
    if complete:
        return "complete"
    if (out_dir / "resume.pt").exists():
        return "resumable"
    return "fresh"


# ---------------------------------------------------------------------------
# one (model, window, seed) run
# ---------------------------------------------------------------------------
def run_one(model_mode: str, window: int, seed: int, cache, by_split: dict, out_root: Path,
            device: str, epochs: int, patience: int, min_delta: float,
            checkpoint_epochs: set, provenance: dict) -> dict:
    out_dir = out_root / model_mode / f"W{window}" / "shipped_split" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "resume.pt"

    t_start = time.time()
    cfg = resolve_config(model_mode, window_radius=window, layers=[33])
    meta_scaler = fit_meta_scaler_from_entries(by_split["train"])
    train_ds = Stage1Dataset(by_split["train"], cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler)
    val_ds = Stage1Dataset(by_split["val"], cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler)

    model = build_stage1_model(model_mode, cfg, init_seed=seed)
    result = train_one_run(
        model, train_ds, val_ds, cfg, device=device, max_epochs=epochs, patience=patience,
        select_on=cfg["select_on"], seed=seed, min_delta=min_delta,
        checkpoint_epochs=checkpoint_epochs, resume_path=str(resume_path),
    )
    t_train = time.time() - t_start

    best_state = {k: v.clone() for k, v in result["model"].state_dict().items()}
    reference = default_reference(cfg["esm_checkpoint"], cache.wt_hash)

    save_checkpoint(out_dir / "best.pt", result["model"], cfg, meta_scaler.state_dict(),
                     result["y_mean"], result["y_std"], reference)

    result["model"].load_state_dict(result["last_state"])
    save_checkpoint(out_dir / "last.pt", result["model"], cfg, meta_scaler.state_dict(),
                     result["y_mean"], result["y_std"], reference)

    for ep, state in result["fixed_budget_states"].items():
        result["model"].load_state_dict(state)
        save_checkpoint(out_dir / f"epoch{ep:03d}.pt", result["model"], cfg, meta_scaler.state_dict(),
                         result["y_mean"], result["y_std"], reference)

    # restore best for the (single, val-selected) test evaluation below
    result["model"].load_state_dict(best_state)

    history_json = {"history": result["history"], "best": result["best"], "stop_reason": result["stop_reason"],
                     "seed": seed}
    with open(out_dir / "history.json", "w") as f:
        json.dump(history_json, f, indent=2, default=str)
    # legacy-shaped metrics.json kept alongside for any code that still reads {history, best}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"history": result["history"], "best": result["best"]}, f, indent=2, default=str)

    flat_rows = []
    for h in result["history"]:
        row = {k: v for k, v in h.items() if k != "val" and k != "val_by_group_spearman" and k != "val_by_group_n"}
        for g in ("missense", "synonymous", "indel"):
            row[f"val_spearman_{g}"] = h["val_by_group_spearman"].get(g)
            row[f"val_n_{g}"] = h["val_by_group_n"].get(g)
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(out_dir / "history.csv", index=False)

    # test-split evaluation -- ONCE, on the val-selected best checkpoint, never used for selection
    test_ds = Stage1Dataset(by_split["test"], cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler)
    collate = make_collate_fn(result["model"].mode)
    test_loader = DataLoader(test_ds, batch_size=int(cfg.get("batch_size", 32)), shuffle=False, collate_fn=collate)
    test_edit_types = [e["edit"].edit_type for e in test_ds.entries]
    test_m, _ = evaluate(result["model"], test_loader, result["y_mean"], result["y_std"], device,
                          edit_types=test_edit_types)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2, default=str)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config_record = {
        "model_mode": model_mode, "window_radius": window, "seed": seed,
        "resolved_config": cfg, "epochs_requested": epochs, "patience": patience,
        "min_delta": min_delta, "checkpoint_epochs_requested": sorted(checkpoint_epochs),
        "checkpoint_epochs_reached": sorted(result["fixed_budget_states"].keys()),
        "device": device, "n_trainable_params": n_trainable,
        "n_epochs_completed": len(result["history"]), "stop_reason": result["stop_reason"],
        "best_epoch": result["best"]["epoch"], "train_seconds": round(t_train, 1),
        **provenance,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2, default=str)

    resume_path.unlink(missing_ok=True)  # run completed -- resume file no longer meaningful

    return {"out_dir": str(out_dir), "stop_reason": result["stop_reason"],
            "n_epochs": len(result["history"]), "train_seconds": round(t_train, 1),
            "best_epoch": result["best"]["epoch"], "test_subset": test_m.get("subset")}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    ap.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--manifest", default="data/split_manifest.csv")
    ap.add_argument("--wt-seq", default="data/wt_sequence.txt")
    ap.add_argument("--cache", default="data/stage1/raw_cache.pt")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=120,
                     help="max epochs; base.yaml's cfg['max_epochs']=100 is currently NOT wired into "
                          "cmd_train (it always uses --epochs, default 100) -- 120 gives >40 epochs of "
                          "headroom past seed42's observed 76-epoch stop")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=0.0,
                     help="0.0 = bit-identical to the pre-existing (undocumented) min_delta=0 behavior")
    ap.add_argument("--checkpoint-epochs", type=int, nargs="+", default=sorted(DEFAULT_CHECKPOINT_EPOCHS))
    ap.add_argument("--dry-run", action="store_true", help="print the planned run list/status and exit")
    args = ap.parse_args()

    combos = [(m, w, s) for m in args.models for w in args.windows for s in args.seeds]
    out_root = Path(args.out_root)
    out_dirs = [out_root / m / f"W{w}" / "shipped_split" / f"seed{s}" for m, w, s in combos]
    assert len(set(out_dirs)) == len(out_dirs), "duplicate output directories planned -- refusing to run"

    print(f"[sweep] {len(combos)} planned runs "
          f"({len(args.models)} models x {len(args.windows)} windows x {len(args.seeds)} seeds)")
    print(f"[sweep] out_root={out_root}  device={args.device}  epochs={args.epochs}  "
          f"patience={args.patience}  min_delta={args.min_delta}")
    for (m, w, s), d in zip(combos, out_dirs):
        print(f"    {run_status(d):9s}  {m:24s} W{w:<3d} seed{s:<3d} -> {d}")

    legacy = {f"{m}_W{w}_seed{s}": legacy_reuse_decision(m, w, s)
              for m, w, s in combos if s == 42}
    if not any(v["reusable"] for v in legacy.values()) and legacy:
        print(f"\n[sweep] legacy runs/stage1/ seed42 results checked for reuse -- NOT reusable "
              f"(see out_root/sweep_manifest.json 'legacy_reuse_decision' for exact reasons per run). "
              f"All seed42 in this sweep will be regenerated fresh under {out_root}.")

    if args.dry_run:
        print("\n[sweep] --dry-run: no cache loaded, no training performed.")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    provenance = {"git": git_commit_hash(), "gpu": gpu_info()}
    print(f"\n[sweep] provenance: {provenance}")

    print(f"[sweep] loading cache from {args.cache} ...")
    t0 = time.time()
    cache = RawStage1Cache.load(args.cache)
    print(f"[sweep] cache loaded in {time.time()-t0:.1f}s")

    manifest = pd.read_csv(args.manifest)
    wt_seq = Path(args.wt_seq).read_text().strip()
    rows = manifest.to_dict("records")
    cohort = join_cohort_with_cache(build_cohort(rows, wt_seq), cache)
    by_split = {"train": [], "val": [], "test": []}
    for e in cohort.supported:
        by_split.setdefault(e["row"]["split"], []).append(e)
    print(f"[sweep] cohort: n_supported={len(cohort.supported)}, "
          f"split sizes={[(k, len(v)) for k, v in by_split.items()]}")

    checkpoint_epochs = set(args.checkpoint_epochs)
    sweep_start = time.time()
    results, failed = [], []

    for (model_mode, window, seed), out_dir in zip(combos, out_dirs):
        status = run_status(out_dir)
        run_id = f"{model_mode}__W{window}__shipped_split__seed{seed}"
        if status == "complete":
            print(f"\n[sweep] SKIP (already complete): {run_id}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = open(out_dir / "stdout.log", "a")
        stderr_log = open(out_dir / "stderr.log", "a")
        t_run_start = time.time()
        print(f"\n[sweep] {'RESUME' if status == 'resumable' else 'START'} [{time.strftime('%H:%M:%S')}] {run_id}")
        try:
            with contextlib.redirect_stdout(Tee(sys.stdout, stdout_log)), \
                 contextlib.redirect_stderr(Tee(sys.stderr, stderr_log)):
                r = run_one(model_mode, window, seed, cache, by_split, out_root, args.device,
                            args.epochs, args.patience, args.min_delta, checkpoint_epochs,
                            {**provenance, "run_started_at": t_run_start})
            elapsed = time.time() - t_run_start
            print(f"[sweep] DONE  {run_id}  epochs={r['n_epochs']} best_epoch={r['best_epoch']} "
                  f"stop={r['stop_reason']} test_subset={r['test_subset']:.4f} "
                  f"train_s={r['train_seconds']:.0f} wall_s={elapsed:.0f}")
            results.append({"run_id": run_id, **r, "wall_seconds": round(elapsed, 1)})
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[sweep] FAILED {run_id}: {e}\n{tb}", file=sys.stderr)
            stderr_log.write(tb)
            failed.append({"run_id": run_id, "error": str(e), "traceback": tb})
        finally:
            stdout_log.close()
            stderr_log.close()

    manifest_out = {
        "provenance": provenance,
        "args": vars(args),
        "combos_planned": len(combos),
        "results": results,
        "failed_runs": failed,
        "legacy_reuse_decision": legacy,
        "sweep_wall_seconds": round(time.time() - sweep_start, 1),
    }
    with open(out_root / "sweep_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2, default=str)

    print(f"\n[sweep] ALL DONE: {len(results)} succeeded, {len(failed)} failed, "
          f"{len(combos) - len(results) - len(failed)} skipped (already complete).")
    if failed:
        print(f"[sweep] failed run_ids: {[f['run_id'] for f in failed]}")


if __name__ == "__main__":
    main()
