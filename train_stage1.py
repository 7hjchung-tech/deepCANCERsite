"""train_stage1.py -- Stage 1 entry point (frozen-ESM representation comparison).

Importing this module, or running it with --help, never starts training or
downloads anything. Every action below is explicit:

    python train_stage1.py --dry-run                       # CPU synthetic forward/config check
    python train_stage1.py --dry-run --model paired_delta   # just one mode
    python train_stage1.py --audit-split                    # read-only split/edited-span audit
    python train_stage1.py --plan --windows 5 10 20          # write an experiment plan (no training)
    python train_stage1.py --model paired_delta --window 10 --train   # NOT run in this session
    python train_stage1.py --plan --execute-plan              # NOT run in this session (explicit sweep)

See README_STAGE1.md for the full command reference and current data/cache
requirements before any real --train call.
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

from src.stage1.cache import RawStage1Cache  # noqa: E402
from src.stage1.checkpoint import default_reference, save_checkpoint  # noqa: E402
from src.stage1.config import resolve_config  # noqa: E402
from src.stage1.dataset import Stage1Dataset, build_cohort, join_cohort_with_cache, make_collate_fn  # noqa: E402
from src.stage1.engine import fit_meta_scaler_from_entries, train_one_run  # noqa: E402
from src.stage1.experiment_plan import generate_experiment_plan, write_plan  # noqa: E402
from src.stage1.model import build_stage1_model  # noqa: E402
from src.stage1.modules import content_projection_param_count  # noqa: E402
from src.stage1.schema import MODEL_MODES  # noqa: E402
from src.stage1.split_audit import audit_split_overlap, build_position_to_split  # noqa: E402
from src.stage1.synthetic import make_synthetic_fixture  # noqa: E402

DEFAULT_MANIFEST = "data/split_manifest.csv"
DEFAULT_WT_SEQ = "data/wt_sequence.txt"
DEFAULT_CACHE = "data/stage1/raw_cache.pt"


def _print_param_counts(mode: str, cfg: dict) -> None:
    model = build_stage1_model(mode, cfg)
    content = content_projection_param_count(mode, cfg["d_esm"], cfg["bottleneck_dim"])
    print(f"    {mode:26s} content_projection={content:,}  total={model.num_total_params():,}  "
          f"trainable={model.num_trainable_params():,}")


def cmd_dry_run(args: argparse.Namespace) -> None:
    modes = [args.model] if args.model != "all" else list(MODEL_MODES)
    print(f"[dry-run] building synthetic CPU fixture (hidden_dim={args.dry_run_hidden_dim}) ...")
    fixture = make_synthetic_fixture(hidden_dim=args.dry_run_hidden_dim, layers=args.layers)
    print(f"[dry-run] cohort: {len(fixture.cohort_entries)} supported, {len(fixture.skipped)} skipped")
    for s in fixture.skipped:
        print(f"    skipped {s['var_id']}: {s['reason']}")

    for mode in modes:
        cfg = resolve_config(mode, window_radius=args.window, layers=args.layers,
                              overrides={"d_esm": args.dry_run_hidden_dim})
        ds = Stage1Dataset(fixture.cohort_entries, fixture.cache, cfg["window_radius"], cfg["layers"])
        collate = make_collate_fn(mode)
        batch = collate([ds[i] for i in range(len(ds))])

        model = build_stage1_model(mode, cfg)
        out = model(batch, return_extras=True)
        weights_sum = out["attn_weights"].sum(-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "attention weights must sum to 1"

        # connectivity check ONLY -- no optimizer.step() anywhere in --dry-run
        loss = out["pred"].pow(2).mean()
        loss.backward()
        n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)

        print(f"[dry-run] {mode}: batch pred shape={tuple(out['pred'].shape)}  "
              f"K shape={tuple(out['K'].shape)}  trainable_with_grad={n_grad}  "
              f"total_params={model.num_total_params():,}")

    print("\n[dry-run] parameter counts (bottleneck_dim=32, d_esm=1280, token_dim=128):")
    full_cfg_probe = resolve_config(MODEL_MODES[0])
    for mode in MODEL_MODES:
        _print_param_counts(mode, {**full_cfg_probe, "model_mode": mode})
    print("\n[dry-run] OK -- no training, no optimizer.step(), no ESM checkpoint download.")


def cmd_audit_split(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    wt_seq = Path(args.wt_seq).read_text().strip()
    rows = manifest.to_dict("records")
    cohort = build_cohort(rows, wt_seq)
    print(f"[audit-split] cohort: {len(cohort.supported)} supported, {len(cohort.skipped)} skipped")
    for s in cohort.skipped[:20]:
        print(f"    skipped {s['var_id']}: {s['reason']}")
    if len(cohort.skipped) > 20:
        print(f"    ... and {len(cohort.skipped) - 20} more")

    pos_to_split = build_position_to_split(rows)
    result = audit_split_overlap(cohort.supported, pos_to_split)
    print(f"[audit-split] indels audited: {result.n_indels_audited}")
    print(f"[audit-split] start-position overlaps (should be 0): {len(result.start_position_overlaps)}")
    print(f"[audit-split] edited-span overlaps: {len(result.edited_span_overlaps)}")
    print(f"[audit-split] edited-span positions with no recorded split: {len(result.unassigned_span_positions)}")
    for o in result.edited_span_overlaps:
        print(f"    OVERLAP {o}")
    if args.strict_edited_span and not args.edited_span_policy:
        raise SystemExit(
            "--strict-edited-span requires --edited-span-policy {exclude,group} -- "
            "no common grouping/exclusion policy has been chosen, so this run is blocked."
        )
    if args.strict_edited_span and not result.is_clean:
        print(f"[audit-split] STRICT mode: overlaps found, policy={args.edited_span_policy} "
              f"-- resolve manually before training with --strict-edited-span.")


def cmd_plan(args: argparse.Namespace) -> None:
    windows = args.windows or [10]
    modes = [args.model] if args.model != "all" else list(MODEL_MODES)
    folds = [None]
    if args.cv_folds:
        folds = [(args.cv_folds, i, args.cv_seed) for i in range(args.cv_folds)]
    plan = generate_experiment_plan(
        model_modes=modes, windows=windows, folds=folds, seeds=args.seeds or [42, 43, 44],
        out_root=args.out, layers=args.layers,
    )
    plan_path = Path(args.out) / "experiment_plan.json"
    write_plan(plan, plan_path)
    n_skip = sum(1 for p in plan if p["skip_existing"])
    print(f"[plan] {len(plan)} runs ({len(modes)} models x {len(windows)} windows x "
          f"{len(folds)} fold(s) x {len(args.seeds or [42, 43, 44])} seeds)")
    print(f"[plan] {n_skip} already have metrics.json and would be skipped (no overwrite)")
    print(f"[plan] wrote {plan_path}")
    if args.execute_plan:
        raise SystemExit(
            "--execute-plan is out of scope for this session (would run real training). "
            "Run individual `python train_stage1.py --model ... --window ... --train` commands "
            "yourself when you are ready."
        )


def cmd_train(args: argparse.Namespace) -> None:
    """Real training. NOT invoked anywhere in this task's execution."""
    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise SystemExit(
            f"{cache_path} not found. Build it first with:\n"
            f"    python dump_stage1_cache.py --manifest {args.manifest} --wt-seq {args.wt_seq} "
            f"--out {cache_path} --layers {' '.join(map(str, args.layers))}\n"
            f"(requires downloading the ESM-2 650M checkpoint -- not done by this task)"
        )
    cache = RawStage1Cache.load(cache_path)
    manifest = pd.read_csv(args.manifest)
    wt_seq = Path(args.wt_seq).read_text().strip()
    rows = manifest.to_dict("records")
    cohort = join_cohort_with_cache(build_cohort(rows, wt_seq), cache)

    by_split = {"train": [], "val": [], "test": []}
    for e in cohort.supported:
        by_split.setdefault(e["row"]["split"], []).append(e)

    cfg = resolve_config(args.model, window_radius=args.window, layers=args.layers)
    meta_scaler = fit_meta_scaler_from_entries(by_split["train"])
    train_ds = Stage1Dataset(by_split["train"], cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler)
    val_ds = Stage1Dataset(by_split["val"], cache, cfg["window_radius"], cfg["layers"], meta_scaler=meta_scaler)

    model = build_stage1_model(args.model, cfg, init_seed=args.seed)
    result = train_one_run(model, train_ds, val_ds, cfg, device=args.device,
                            max_epochs=args.epochs, patience=cfg["patience"], select_on=cfg["select_on"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out_dir / "best.pt", result["model"], cfg, meta_scaler.state_dict(),
        result["y_mean"], result["y_std"],
        default_reference(cfg["esm_checkpoint"], cache.wt_hash),
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": result["history"], "best": result["best"]}, f, indent=2, default=str)
    print(f"[train] saved -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=[*MODEL_MODES, "all"], default="all")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--windows", type=int, nargs="+", default=None, help="--plan only")
    ap.add_argument("--layers", type=int, nargs="+", default=[33])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--wt-seq", default=DEFAULT_WT_SEQ)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default="runs/stage1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="--plan only")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cv-folds", type=int, default=None)
    ap.add_argument("--cv-fold", type=int, default=0)
    ap.add_argument("--cv-seed", type=int, default=0)
    ap.add_argument("--strict-edited-span", action="store_true")
    ap.add_argument("--edited-span-policy", choices=["exclude", "group"], default=None)
    ap.add_argument("--dry-run-hidden-dim", type=int, default=16,
                     help="tiny D_esm used ONLY by --dry-run's synthetic fixture")

    ap.add_argument("--dry-run", action="store_true", help="CPU synthetic forward/config check")
    ap.add_argument("--audit-split", action="store_true", help="read-only split/edited-span audit")
    ap.add_argument("--plan", action="store_true", help="write an experiment plan (never trains)")
    ap.add_argument("--execute-plan", action="store_true",
                     help="EXPLICIT sweep-execution switch -- refused in this task's session")
    ap.add_argument("--train", action="store_true",
                     help="EXPLICIT single-run training switch -- not invoked by this task")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        cmd_dry_run(args)
    elif args.audit_split:
        cmd_audit_split(args)
    elif args.plan:
        cmd_plan(args)
    elif args.train:
        cmd_train(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
