"""
train.py — z-score regression training loop for M1-M4.

Regression only: model.py exposes a single reg_head, so this optimises MSE on
z_score_D4_D14 and reports Spearman / Pearson / RMSE. A classification head
(functional_classification) is not wired up yet.

HOW TO RUN
----------
    python build_meta_x.py                                   # once
    python dump_diff_emb.py --device cuda                    # once (M1/M2 cache)

    python train.py --config configs/m1.yaml                 # all 3 seeds
    python train.py --config configs/m3.yaml --epochs 10     # e2e LoRA, GPU
    python train.py --config configs/m1.yaml --seed 42       # single seed

OUTPUTS (per run)
-----------------
    runs/<MODEL_ID>/seed<N>/best.pt        trainable weights only, best val epoch
    runs/<MODEL_ID>/seed<N>/metrics.json   per-epoch history + final test metrics
    runs/<MODEL_ID>/summary.json           mean +/- std across seeds

DESIGN NOTES
------------
* Row alignment: struct_X / meta_X / y / split all come from files whose row
  order is rad51c_struct_features.csv's var_id order. This is asserted against
  split_manifest.csv at load time, not assumed.

* Feature standardisation: struct columns are standardised with mean/std fit on
  TRAIN ROWS ONLY (no leakage). Binary columns -- detected by their actual
  values being a subset of {0,1}, not hardcoded -- are left untouched, as is
  meta (one-hot) and diff_emb (already LayerNorm'd inside BottleneckMLP).
  The fitted statistics are saved next to the checkpoint so inference can
  reproduce them. Disable with --no-standardize.

* Target scaling: z_score spans roughly [-34, +5], so MSE on the raw target is
  dominated by a handful of extreme depleted variants. The loss is computed on
  a train-standardised target; predictions are converted back before any metric
  is reported, so every number in metrics.json is in original z-score units.
  Spearman is invariant to this either way.

* Checkpoints save trainable parameters only. For M3/M4 the full state_dict
  would include the frozen 650M ESM-2 backbone (~2.5 GB per seed); the base
  weights are reproducible from the pretrained checkpoint, only the LoRA and
  head weights are run-specific.
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
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import build_model  # noqa: E402
from src.config_loader import load_model_config  # noqa: E402

STRUCT_X_PATH = "data/structure/results/rad51c_X.npy"
STRUCT_FEATURES_PATH = "data/structure/results/rad51c_struct_features.csv"
META_X_PATH = "data/structure/results/rad51c_meta_X.npy"
Y_PATH = "data/structure/results/rad51c_y.npy"
MANIFEST_PATH = "data/split_manifest.csv"
WT_SEQ_PATH = "data/wt_sequence.txt"
DEFAULT_CACHE_PATH = "data/diff_emb_raw.pt"


# ======================================================================
# metrics (numpy-only: avoids a scipy dependency on the training box)
# ======================================================================
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- same convention as scipy.stats.rankdata."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": _pearson(_rankdata(y_true), _rankdata(y_pred)),
        "pearson": _pearson(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "n": int(len(y_true)),
    }


# ======================================================================
# data
# ======================================================================
def kfold_split(manifest: pd.DataFrame, n_folds: int, fold: int, cv_seed: int) -> np.ndarray:
    """Re-derive train/val/test by holding out whole RESIDUE POSITIONS.

    The shipped split puts 262 positions in train, 56 in val and 57 in test.
    882 test variants therefore sit on 57 positions, and variants sharing a
    position share their entire struct feature vector and nearly all of their
    ESM context -- the effective sample size is far closer to 57 than to 882.
    That is why val and test differ by a consistent ~0.08 in missense Spearman
    for every model and setting tried: the two position sets are simply not
    equally hard.

    Rotating which positions are held out turns "our score on these 57
    positions" into a mean and spread over folds, which is the only honest way
    to say whether a 0.03 gap to the LLR baseline means anything on a 376-
    residue protein.

    Splitting by position (never by variant) is what the shipped manifest does
    too -- it is what keeps the same residue out of both train and test.
    """
    positions = np.sort(manifest["pp"].unique())
    rng = np.random.default_rng(cv_seed)
    shuffled = rng.permutation(positions)
    groups = np.array_split(shuffled, n_folds)

    test_pos = set(groups[fold % n_folds].tolist())
    val_pos = set(groups[(fold + 1) % n_folds].tolist())

    pp = manifest["pp"].to_numpy()
    split = np.where(np.isin(pp, list(test_pos)), "test",
                     np.where(np.isin(pp, list(val_pos)), "val", "train"))
    print(f"[cv] fold {fold}/{n_folds} (cv_seed={cv_seed}): positions "
          f"train={len(positions) - len(test_pos) - len(val_pos)} "
          f"val={len(val_pos)} test={len(test_pos)}")
    return split


def load_data(cv: tuple[int, int, int] | None = None) -> dict:
    struct_features = pd.read_csv(STRUCT_FEATURES_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    if len(struct_features) != len(manifest) or not struct_features["var_id"].equals(
        manifest["var_id"]
    ):
        raise SystemExit(
            "var_id order differs between rad51c_struct_features.csv and "
            "split_manifest.csv -- every array below is indexed by row number, "
            "so this must match exactly."
        )

    meta_path = Path(META_X_PATH)
    if not meta_path.exists():
        raise SystemExit(
            f"{META_X_PATH} not found. Run:  python build_meta_x.py"
        )

    struct_x = np.load(STRUCT_X_PATH).astype(np.float32)
    meta_x = np.load(META_X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.float32)

    for name, arr in [("rad51c_X.npy", struct_x), ("rad51c_meta_X.npy", meta_x),
                      ("rad51c_y.npy", y)]:
        if len(arr) != len(manifest):
            raise SystemExit(f"{name} has {len(arr)} rows, manifest has {len(manifest)}")

    if np.isnan(y).any():
        raise SystemExit(f"{Y_PATH} contains {int(np.isnan(y).sum())} NaN targets")

    split = (kfold_split(manifest, *cv) if cv is not None
             else struct_features["split"].to_numpy())
    var_ids = struct_features["var_id"].tolist()

    manifest_lookup = {
        row["var_id"]: {"pp": int(row["pp"]), "mut_seq": row["mut_seq"]}
        for row in manifest.to_dict("records")
    }
    wt_seq = Path(WT_SEQ_PATH).read_text().strip()

    # Variant group per row, for group-wise metrics. missense vs synonymous
    # only exists in the manifest's slim_consequence (rad51c_meta.csv collapses
    # both into var_type="sav"), so take it from there.
    cons = manifest["slim_consequence"].to_numpy()
    group = np.where(cons == "missense", "missense",
                     np.where(cons == "synonymous", "synonymous", "indel"))

    idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
    print(f"[data] {len(manifest)} variants  "
          f"train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}")
    print(f"[data] struct {struct_x.shape}  meta {meta_x.shape}")

    return {
        "struct_x": struct_x, "meta_x": meta_x, "y": y, "group": group,
        "var_ids": var_ids, "idx": idx,
        "manifest": manifest_lookup, "wt_seq": wt_seq,
    }


def standardize_struct(struct_x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Z-score the non-binary struct columns using TRAIN-ONLY statistics.

    Binary columns are detected from the data (values ⊆ {0,1}) rather than
    hardcoded by index, so a change to the struct pipeline's column layout
    cannot silently standardise a one-hot column.
    """
    is_binary = np.array([
        np.isin(np.unique(struct_x[:, j]), (0.0, 1.0)).all()
        for j in range(struct_x.shape[1])
    ])
    cont = ~is_binary

    mean = struct_x[train_idx][:, cont].mean(axis=0)
    std = struct_x[train_idx][:, cont].std(axis=0)
    std[std == 0] = 1.0                       # constant column -> leave as-is

    out = struct_x.copy()
    out[:, cont] = (out[:, cont] - mean) / std
    print(f"[data] standardised {int(cont.sum())}/{len(cont)} struct columns "
          f"(train-only stats); {int(is_binary.sum())} binary columns untouched")
    return out, {
        "cont_mask": cont.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


# ======================================================================
# training
# ======================================================================
def make_param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """Two LRs: LoRA adapters vs everything else (head + backbone)."""
    lr_cfg = cfg.get("lr") or {}
    lr_head = float(lr_cfg.get("head", 1.0e-3))
    lr_lora = float(lr_cfg.get("lora", 1.0e-5))

    lora, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora if ("lora_A" in name or "lora_B" in name) else head).append(p)

    groups = [{"params": head, "lr": lr_head, "name": "head"}]
    msg = f"[opt] head: {sum(p.numel() for p in head):,} params @ lr={lr_head:g}"
    if lora:
        groups.append({"params": lora, "lr": lr_lora, "name": "lora"})
        msg += f"  |  lora: {sum(p.numel() for p in lora):,} params @ lr={lr_lora:g}"
    print(msg)
    return groups


@torch.no_grad()
def evaluate(model, data, rows, batch_size, device, y_mean, y_std) -> tuple[dict, np.ndarray]:
    model.eval()
    preds = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        b = rows[start:start + batch_size]
        var_ids = [data["var_ids"][i] for i in b]
        struct = torch.from_numpy(data["struct_x"][b]).to(device)
        meta = torch.from_numpy(data["meta_x"][b]).to(device)
        out = model(var_ids, struct, meta)
        preds[start:start + len(b)] = out.detach().float().cpu().numpy()
    preds = preds * y_std + y_mean                    # back to z-score units
    m = compute_metrics(data["y"][rows], preds)
    # MSE in the standardised space the loss was computed in: RMSE_std = RMSE/y_std.
    m["loss"] = float((m["rmse"] / y_std) ** 2)

    # Within-group Spearman. Pooled Spearman has a floor of 0.37 from
    # between-group separation alone; these do not.
    groups = data["group"][rows]
    per_group = {}
    for g in ("missense", "synonymous", "indel"):
        mask = groups == g
        if mask.sum() >= 2:
            per_group[g] = compute_metrics(data["y"][rows][mask], preds[mask])["spearman"]
    m["by_group"] = per_group

    # Selection target: the mean of the two groups the study is actually about.
    # synonymous is excluded on purpose -- its diff embedding is exactly zero
    # for every variant, so its within-group rho is noise around 0 and would
    # only add variance to model selection.
    scored = [per_group[g] for g in ("missense", "indel") if g in per_group]
    m["subset"] = float(np.mean(scored)) if scored else float("nan")
    return m, preds


def train_one_seed(cfg: dict, data: dict, seed: int, args, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device)
    cached = cfg["esm_mode"] == "cached"

    if cached:
        cache_path = args.cache
        if not Path(cache_path).exists():
            raise SystemExit(
                f"{cache_path} not found (esm_mode='cached' needs it). Run:\n"
                f"    python dump_diff_emb.py --device {args.device}"
            )
        model, dims = build_model(cfg, STRUCT_X_PATH, META_X_PATH, cache_path=cache_path)
        model.diff_embedder.cache_to(device)
    else:
        # e2e keeps ESM-2 on cfg["esm"]["device"]; keep the two consistent.
        cfg = {**cfg, "esm": {**cfg["esm"], "device": str(device)}}
        model, dims = build_model(
            cfg, STRUCT_X_PATH, META_X_PATH,
            manifest=data["manifest"], wt_seq=data["wt_seq"],
        )
    model = model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed {seed}] dims={dims}  trainable={n_trainable:,}")

    if cfg.get("gradient_checkpointing"):
        print(f"[seed {seed}] NOTE: config requests gradient_checkpointing, but this "
              f"ESM-2 implementation does not expose it -- running without. Lower "
              f"batch_size if you hit OOM.")

    train_idx, val_idx, test_idx = data["idx"]["train"], data["idx"]["val"], data["idx"]["test"]

    # target standardisation (train-only); metrics are always reported unscaled
    y_mean = float(data["y"][train_idx].mean())
    y_std = float(data["y"][train_idx].std()) or 1.0
    y_scaled = (data["y"] - y_mean) / y_std

    batch_size = int(args.batch_size or cfg.get("batch_size", 64))
    accum = int(args.accum or cfg.get("gradient_accumulation", 1))

    # Comparability across M1-M4 hinges on the EFFECTIVE batch (batch_size x
    # accumulation), not on batch_size alone. Halving batch_size to fit a
    # smaller GPU is free as long as accumulation is raised to match -- there
    # is no BatchNorm here, only LayerNorm, so accumulation reproduces the
    # large-batch gradient. Print both so a run's real setting is on the record.
    cfg_effective = int(cfg.get("batch_size", 64)) * int(cfg.get("gradient_accumulation", 1))
    effective = batch_size * accum
    if effective != cfg_effective:
        print(f"[seed {seed}] WARNING: effective batch {effective} "
              f"({batch_size}x{accum}) differs from the config's {cfg_effective} "
              f"({cfg.get('batch_size', 64)}x{cfg.get('gradient_accumulation', 1)}). "
              f"Runs compared against each other should match -- consider "
              f"--accum {max(1, cfg_effective // batch_size)}.")
    precision = args.precision or cfg.get("precision", "fp32")
    use_amp = (precision == "fp16") and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = torch.optim.AdamW(
        make_param_groups(model, cfg), weight_decay=float(cfg.get("weight_decay", 0.01))
    )
    loss_fn = nn.HuberLoss(delta=1.0) if args.loss == "huber" else nn.MSELoss()

    print(f"[seed {seed}] batch_size={batch_size} accum={accum} "
          f"(effective {effective}) precision={precision} amp={use_amp} "
          f"epochs<={args.epochs} patience={args.patience} select_on={args.select_on}")

    # Epoch selection. Neither obvious choice works here:
    #
    #   pooled val Spearman -- the split is 78/16/6% missense/synonymous/indel
    #     with group means -3.5/-0.2/-11.2, so predicting nothing but the group
    #     mean already scores 0.37. It barely moves when the within-group
    #     ranking improves, and every seed stopped at epoch 1-4.
    #   val loss -- z_score reaches -33.6 against a bulk near 0, so a handful of
    #     extreme variants dominate MSE. Selecting on it measurably COST rank
    #     quality: missense rho fell 0.589 -> 0.567 versus selecting on pooled
    #     Spearman, both below the 0.5935 LLR baseline.
    #
    # So select on the mean of the within-group Spearman of missense and indel:
    # aligned with how the models are judged, no between-group floor, and it
    # weights the two groups the ablation asks about equally instead of letting
    # missense (78% of rows) decide alone. synonymous is left out because its
    # diff embedding is identically zero, making its within-group rho noise.
    sign = -1.0 if args.select_on == "loss" else 1.0     # loss: lower is better
    history: list[dict] = []
    best = {"score": -np.inf, "epoch": -1}
    best_state: dict | None = None
    rng = np.random.default_rng(seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        running, n_batches = 0.0, 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, start in enumerate(range(0, len(order), batch_size)):
            b = order[start:start + batch_size]
            var_ids = [data["var_ids"][i] for i in b]
            struct = torch.from_numpy(data["struct_x"][b]).to(device)
            meta = torch.from_numpy(data["meta_x"][b]).to(device)
            target = torch.from_numpy(y_scaled[b]).to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(var_ids, struct, meta)
                loss = loss_fn(out.float(), target)

            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach())
            n_batches += 1

        # flush a trailing partial accumulation window
        if n_batches % accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running / max(n_batches, 1)
        val_m, _ = evaluate(model, data, val_idx, batch_size, device, y_mean, y_std)
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_m,
                        "seconds": round(time.time() - t0, 1)})
        bg = val_m["by_group"]
        print(f"[seed {seed}] epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val loss={val_m['loss']:.4f} rho={val_m['spearman']:.4f} "
              f"| mis={bg.get('missense', float('nan')):.4f} "
              f"indel={bg.get('indel', float('nan')):.4f} "
              f"subset={val_m['subset']:.4f}  ({time.time() - t0:.0f}s)")

        score = sign * val_m[args.select_on]
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "val": val_m}
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
                if k in {n for n, p in model.named_parameters() if p.requires_grad}
            }
        elif epoch - best["epoch"] >= args.patience:
            print(f"[seed {seed}] early stop: no val {args.select_on} improvement "
                  f"for {args.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    test_m, test_preds = evaluate(model, data, test_idx, batch_size, device, y_mean, y_std)
    val_m, _ = evaluate(model, data, val_idx, batch_size, device, y_mean, y_std)
    tg = test_m["by_group"]
    print(f"[seed {seed}] BEST epoch {best['epoch']}  test rho={test_m['spearman']:.4f} "
          f"| missense={tg.get('missense', float('nan')):.4f} "
          f"(LLR baseline 0.5935)  indel={tg.get('indel', float('nan')):.4f}")

    seed_dir = out_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"trainable_state_dict": best_state,
         "cfg": cfg, "dims": dims, "seed": seed,
         "best_epoch": best["epoch"], "select_on": args.select_on,
         "effective_batch": effective, "precision": precision,
         "y_mean": y_mean, "y_std": y_std,
         "struct_scaler": data.get("struct_scaler")},
        seed_dir / "best.pt",
    )
    result = {"seed": seed, "best_epoch": best["epoch"], "val": val_m, "test": test_m}
    with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({**result, "history": history}, f, indent=2)
    np.save(seed_dir / "test_predictions.npy", test_preds)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. configs/m1.yaml")
    ap.add_argument("--cache", default=DEFAULT_CACHE_PATH,
                    help="diff_emb cache (.pt) for esm_mode='cached'")
    ap.add_argument("--out", default="runs", help="output root directory")
    ap.add_argument("--seed", type=int, default=None,
                    help="run this seed only (default: every seed in the config)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="run exactly these seeds, e.g. --seeds 42 43 44 45 46. Use the "
                         "SAME list for every model you intend to compare.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10,
                    help="early stop after N epochs without val improvement")
    ap.add_argument("--select-on", choices=["subset", "loss", "spearman"], default="subset",
                    help="val metric picking the best epoch. Default 'subset': mean of "
                         "the within-group Spearman of missense and indel. 'spearman' "
                         "(pooled) has a 0.37 floor from group separation; 'loss' is "
                         "dominated by the z_score tail and costs rank quality.")
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse",
                    help="training objective on the standardised target. 'huber' caps "
                         "the influence of the extreme depleted variants (z down to "
                         "-33.6), which dominate MSE while the metric is rank-based.")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default=None,
                    help="overrides the config. Changes numerics, so use the SAME value "
                         "for every model you intend to compare.")
    ap.add_argument("--batch-size", type=int, default=None, help="overrides the config")
    ap.add_argument("--accum", type=int, default=None,
                    help="gradient accumulation steps, overrides the config. Raise this "
                         "when you lower --batch-size to fit a smaller GPU: what has to "
                         "match across compared runs is batch_size x accum.")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    # Capacity overrides for sweeping without editing base.yaml. They are part
    # of the settings record, so analyze_runs.py still catches a model swept to
    # a different capacity than the one it is compared against.
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--ffn-dim", type=int, default=None)
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    # LoRA capacity (M3/M4 only). 4,124 training rows against rank-8 adapters on
    # all 33 layers is 1.35M trainable adapter parameters; restricting the rank
    # or the layers is the standard small-data remedy and, unlike anything tuned
    # per gene, it is a property of the readout rather than of RAD51C.
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=float, default=None)
    ap.add_argument("--lora-last-k", type=int, default=None,
                    help="adapt only the last K transformer layers (default: all 33)")
    ap.add_argument("--lora-lr", type=float, default=None)
    ap.add_argument("--repr-layer", type=int, default=None,
                    help="ESM-2 layer to read (e2e path). For cached runs the layer is "
                         "whatever --cache was dumped with.")
    ap.add_argument("--window", type=int, default=None,
                    help="pooling window radius (e2e path); cached runs inherit it "
                         "from the cache file.")
    ap.add_argument("--no-standardize", action="store_true",
                    help="skip train-only standardisation of struct features")
    ap.add_argument("--cv-folds", type=int, default=None,
                    help="hold out whole residue positions in N folds instead of using "
                         "the shipped split. Measures how much of a score is the luck of "
                         "which positions landed in test.")
    ap.add_argument("--cv-fold", type=int, default=0,
                    help="which fold is the test set (0-indexed); fold+1 becomes val")
    ap.add_argument("--cv-seed", type=int, default=0,
                    help="shuffles positions before folding; keep it fixed across models")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_model_config(args.config)
    overrides = {
        "hidden_dim": args.hidden_dim, "ffn_dim": args.ffn_dim,
        "n_blocks": args.n_blocks, "dropout": args.dropout,
        "weight_decay": args.weight_decay,
    }
    applied = {k: v for k, v in overrides.items() if v is not None}
    if applied:
        cfg = {**cfg, **applied}

    # Nested overrides (lora.*, esm.repr_layer, pooling.window_W)
    n_layers = 33                     # ESM-2 650M
    lora_over = {}
    if args.lora_rank is not None:
        lora_over["rank"] = args.lora_rank
    if args.lora_alpha is not None:
        lora_over["alpha"] = args.lora_alpha
    if args.lora_last_k is not None:
        lora_over["target_layers"] = list(range(n_layers - args.lora_last_k, n_layers))
    if lora_over:
        cfg = {**cfg, "lora": {**cfg["lora"], **lora_over}}
        applied["lora"] = lora_over
    if args.lora_lr is not None:
        cfg = {**cfg, "lr": {**(cfg.get("lr") or {}), "lora": args.lora_lr}}
        applied["lr.lora"] = args.lora_lr
    if args.repr_layer is not None:
        cfg = {**cfg, "esm": {**cfg["esm"], "repr_layer": args.repr_layer}}
        applied["esm.repr_layer"] = args.repr_layer
    if args.window is not None:
        cfg = {**cfg, "pooling": {**cfg["pooling"], "window_W": args.window}}
        applied["pooling.window_W"] = args.window

    if applied:
        print(f"[cfg] overridden: {applied}  (use the SAME overrides for every "
              f"model you compare)")

    # A multi-layer cache concatenates k layers, so its diff vectors are 1280*k
    # wide. Read the width off the cache instead of requiring the caller to keep
    # --cache and bottleneck.in_dim manually in sync.
    if cfg["esm_mode"] == "cached" and Path(args.cache).exists():
        probe = torch.load(args.cache, weights_only=True)
        cache_dim = int(next(iter(probe.values()))["diff"].numel())
        del probe
        if cache_dim != int(cfg["bottleneck"]["in_dim"]):
            print(f"[cfg] cache diff dim {cache_dim} != bottleneck.in_dim "
                  f"{cfg['bottleneck']['in_dim']} -> 자동 조정")
            cfg = {**cfg, "bottleneck": {**cfg["bottleneck"], "in_dim": cache_dim}}

    model_id = cfg.get("model_id", Path(args.config).stem.upper())
    print(f"=== {model_id}  ({args.config})  device={args.device} ===")
    print(f"    esm_mode={cfg['esm_mode']}  use_lora={cfg['use_lora']}  "
          f"use_block_b={cfg['use_block_b']}")

    cv = ((args.cv_folds, args.cv_fold, args.cv_seed) if args.cv_folds else None)
    data = load_data(cv)
    if args.no_standardize:
        data["struct_scaler"] = None
        print("[data] standardisation disabled (--no-standardize)")
    else:
        data["struct_x"], data["struct_scaler"] = standardize_struct(
            data["struct_x"], data["idx"]["train"]
        )

    if args.seed is not None and args.seeds is not None:
        raise SystemExit("pass --seed or --seeds, not both")
    seeds = args.seeds or ([args.seed] if args.seed is not None else cfg.get("seed", [42]))
    if isinstance(seeds, int):
        seeds = [seeds]

    out_dir = Path(args.out) / model_id
    results = [train_one_seed(cfg, data, int(s), args, out_dir) for s in seeds]

    print(f"\n{'=' * 60}\n{model_id} SUMMARY ({len(results)} seed(s))\n{'=' * 60}")
    # Record the settings that have to match across compared models, so a
    # stale run in runs/ can be spotted instead of silently averaged in.
    summary = {
        "model_id": model_id, "config": args.config, "seeds": results, "test": {},
        "settings": {
            "lr": cfg.get("lr"), "weight_decay": cfg.get("weight_decay"),
            "effective_batch": int(args.batch_size or cfg.get("batch_size", 64))
            * int(args.accum or cfg.get("gradient_accumulation", 1)),
            "precision": args.precision or cfg.get("precision", "fp32"),
            "select_on": args.select_on, "loss": args.loss, "patience": args.patience,
            "cv": {"folds": args.cv_folds, "fold": args.cv_fold, "seed": args.cv_seed}
            if args.cv_folds else None,
            "max_epochs": args.epochs, "seed_list": [int(s) for s in seeds],
            "hidden_dim": cfg.get("hidden_dim"), "ffn_dim": cfg.get("ffn_dim"),
            "n_blocks": cfg.get("n_blocks"), "dropout": cfg.get("dropout"),
            "lora": cfg.get("lora"),
            "repr_layer": cfg.get("esm", {}).get("repr_layer"),
            "window_W": cfg.get("pooling", {}).get("window_W"),
            "cache": args.cache if cfg["esm_mode"] == "cached" else None,
        },
    }
    for metric in ("spearman", "pearson", "rmse"):
        vals = np.array([r["test"][metric] for r in results], dtype=np.float64)
        summary["test"][metric] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"  test {metric:<9} {vals.mean():.4f} +/- {vals.std():.4f}"
              f"   ({', '.join(f'{v:.4f}' for v in vals)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
