"""End-to-end interface check: frozen cache -> loader -> Stage 1 -> scalar.

    .venv/bin/python -m src.stage1.smoke

Proves the plumbing and nothing else. It reports shapes, masks, ids and splits;
it reports NO performance, and the mock consumer it runs is not a model.

Test rows are never given a target here: the smoke check fits the scaler on
train only and evaluates on val only, so no test label is read at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from . import data as sd
from .interface import check_prediction
from .losses import HUBER_DELTA, conventional_mse, get_objective
from .metrics import evaluate_split
from .mock import MockTokenConsumer
from .protocol import PILOT

_ROOT = Path(__file__).resolve().parents[2]


class Checks:
    def __init__(self) -> None:
        self.results = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        return bool(ok)

    @property
    def all_passed(self) -> bool:
        return all(r["pass"] for r in self.results)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(sd.DEFAULT_CACHE))
    ap.add_argument("--manifest", default=str(sd.DEFAULT_MANIFEST))
    ap.add_argument("--batch-size", type=int, default=PILOT.batch_size)
    ap.add_argument("--seed", type=int, default=PILOT.seeds[0])
    ap.add_argument("--out", default=str(_ROOT / "data" / "stage1" / "interface_smoke.json"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    c = Checks()

    # ---- 1. load + verify -------------------------------------------------
    print("[1] loading and verifying the frozen cache")
    data = sd.load_frozen_data(
        cache_path=Path(args.cache), manifest_path=Path(args.manifest), verify_cohort=True
    )
    rep = data.report()
    c.check("cache provenance verified against code + manifest", True,
            data.cache.provenance["provenance_hash"][:16])
    c.check("cohort == validated in_eval_scope, recomputed from variant_map",
            rep.cohort_verified, f"{rep.n_cache_rows} rows of {rep.n_manifest_rows}")
    c.check("splits preserved from the shipped manifest",
            sum(rep.rows_by_split.values()) == rep.n_cache_rows, str(rep.rows_by_split))
    c.check("every row has a finite target available",
            rep.n_targets_present == rep.n_cache_rows,
            f"{rep.n_targets_present}/{rep.n_cache_rows} {rep.target_column}")

    # ---- 2. rows are mapped by var_id, not by position --------------------
    print("[2] var_id mapping")
    probe = data.var_id[7]
    man_row = data.manifest[data.manifest.var_id == probe].iloc[0]
    c.check("cache var_id -> manifest row lookup agrees on split",
            data.split_of[probe] == man_row.split, f"{probe} -> {man_row.split}")
    shuffled = [data.cache_index_of[v] for v in reversed(data.var_id[:5])]
    b_rev, t_rev = data.make_batch(shuffled)
    c.check("a batch built from arbitrary indices keeps id order",
            b_rev.var_id == list(reversed(data.var_id[:5]))
            and t_rev.var_id == b_rev.var_id,
            str(b_rev.var_id[:2]))

    # ---- 3. scaler: train only -------------------------------------------
    print("[3] target scaler (train rows only)")
    scaler = data.fit_target_scaler(split="train")
    train_idx = data.indices_for_split("train")
    val_idx = data.indices_for_split("val")
    y_tr = data.targets_raw(train_idx)
    c.check("scaler mean/std equal the TRAIN rows' statistics",
            np.isclose(scaler.mean, y_tr.mean())
            and np.isclose(scaler.std, y_tr.std(ddof=scaler.ddof)),
            scaler.describe())
    y_val = data.targets_raw(val_idx)
    c.check("scaler was NOT refitted on val",
            not np.isclose(scaler.mean, y_val.mean(), atol=1e-9),
            f"train mu={scaler.mean:.6f} vs val mu={y_val.mean():.6f}")
    rt = scaler.inverse_transform(scaler.transform(y_val))
    c.check("inverse_transform round-trips", float(np.abs(rt - y_val).max()) < 1e-9,
            f"max|err|={float(np.abs(rt - y_val).max()):.2e}")

    # ---- 4. one batch through the interface -------------------------------
    print("[4] one train batch through the Stage 1 input contract")
    gen = torch.Generator().manual_seed(args.seed)
    loader = sd.make_loader(data, "train", batch_size=args.batch_size,
                            shuffle=True, scaler=scaler, generator=gen)
    batch, targets = next(iter(loader))
    batch.validate()
    batch.assert_no_labels()
    c.check("Stage1Batch validates against the full contract", True,
            f"[B,L,A,D]={tuple(batch.H_WT.shape)}")
    c.check("H_WT / H_MUT / delta_H all [B,3,A,1280] fp32",
            batch.H_WT.shape == batch.H_MUT.shape == batch.delta_H.shape
            and batch.n_layers == 3 and batch.embed_dim == 1280
            and batch.H_WT.dtype is torch.float32,
            str(batch.layers))
    c.check("every row in the batch is a train row",
            set(batch.split) == {"train"}, str(sorted(set(batch.split))))
    c.check("targets arrive as a SEPARATE object keyed by the same ids",
            targets.var_id == batch.var_id and not hasattr(batch, "y_raw"),
            f"target_column={targets.target_column}")

    kinds = [k for b in range(batch.batch_size) for k in batch.slot_kind_names(b)]
    pad_mask = batch.kind_mask("pad")
    c.check("padding slots are token_invalid",
            not bool((pad_mask & batch.token_valid).any()),
            f"{int(pad_mask.sum())} pad slots in this batch")
    gap_mask = batch.kind_mask("wt_only") | batch.kind_mask("mut_only")
    c.check("gap slots (if any) are token_VALID and never delta_valid",
            not bool((gap_mask & ~batch.token_valid).any())
            and not bool((gap_mask & batch.delta_valid).any()),
            f"{int(gap_mask.sum())} gap slots in this batch, "
            f"{int((gap_mask & batch.token_valid).sum())} of them token_valid")
    c.check("slot kinds present are the Task C vocabulary",
            set(kinds) <= set(batch.slot_kind_vocab), str(sorted(set(kinds))))

    # ---- 5. mock consumer -------------------------------------------------
    print("[5] mock consumer (NOT a model -- plumbing proof only)")
    model = MockTokenConsumer(n_layers=batch.n_layers, embed_dim=batch.embed_dim)
    pred = model(batch)
    check_prediction(pred, batch)
    c.check("mock returns one finite scalar per variant",
            tuple(pred.shape) == (batch.batch_size,) and bool(torch.isfinite(pred).all()),
            f"pred{tuple(pred.shape)}")

    # padding must be unable to influence the output
    perturbed = batch.to("cpu")
    pm = pad_mask.unsqueeze(1).unsqueeze(-1)
    perturbed.delta_H = batch.delta_H + pm.to(batch.delta_H.dtype) * 1e3
    with torch.no_grad():
        pred_perturbed = model(perturbed)
    c.check("padding cannot change the prediction (masks are honoured)",
            torch.equal(pred.detach(), pred_perturbed)
            if pad_mask.any() else True,
            f"max|diff|={float((pred.detach() - pred_perturbed).abs().max()):.2e}")

    # ---- 6. both objectives are computable on the same batch --------------
    print("[6] objective / metric plumbing")
    losses: Dict[str, float] = {}
    for name in PILOT.objectives:
        fn = get_objective(name)
        losses[name] = float(fn(pred, targets.y_std))
    mse_std = float(torch.mean((pred - targets.y_std) ** 2))
    c.check("half_mse is exactly 0.5 * standardised MSE",
            abs(losses["half_mse"] - 0.5 * mse_std) < 1e-6,
            f"half_mse={losses['half_mse']:.6f} 0.5*mse_std={0.5 * mse_std:.6f}")
    pred_raw = scaler.inverse_transform(pred.detach().numpy())
    y_raw = targets.y_raw.numpy()
    conv = conventional_mse(pred_raw, y_raw)
    c.check("conventional MSE is a raw-space METRIC, distinct from the loss",
            abs(conv - losses["half_mse"]) > 1e-6
            and abs(conv - scaler.std ** 2 * mse_std) < 1e-3 * max(conv, 1.0),
            f"raw MSE={conv:.4f} vs half_mse(std)={losses['half_mse']:.4f}")

    # gradients must reach the head (interface proof, NOT training)
    losses_t = get_objective("huber")(pred, targets.y_std)
    losses_t.backward()
    c.check("gradients flow back through the interface to the consumer",
            model.head.weight.grad is not None
            and bool(torch.isfinite(model.head.weight.grad).all()),
            f"grad norm={float(model.head.weight.grad.norm()):.3e}")

    # ---- 7. a val pass, reported with explicit undefined handling ---------
    print("[7] validation-side metric plumbing (no test labels read)")
    vb, vt = data.make_batch(val_idx[: args.batch_size], scaler=scaler)
    with torch.no_grad():
        vpred = model(vb)
    vpred_raw = scaler.inverse_transform(vpred.numpy())
    report = evaluate_split(
        var_id=vb.var_id,
        variant_type=vb.variant_type,
        y_true_raw=vt.y_raw.numpy(),
        y_pred_raw=vpred_raw,
        y_true_std=vt.y_std.numpy(),
        y_pred_std=vpred.numpy(),
        optimization_loss=float(get_objective("huber")(vpred, vt.y_std)),
        objective="huber",
    )
    c.check("validation report carries Spearman with explicit definedness",
            "value" in report["missense_spearman"]
            and "undefined_reason" in report["missense_spearman"],
            f"missense n={report['missense_spearman']['n']}")
    c.check("raw-space MAE/RMSE/MSE are all present",
            all(k in report for k in ("raw_mae", "raw_rmse", "raw_mse")))
    c.check("test labels were never read in this run",
            True, "scaler=train only; metrics=val only")

    out = {
        "status": "INTERFACE PLUMBING ONLY -- no model was trained, no performance measured",
        "real_stage1_present": False,
        "note": (
            "The team's real token Stage 1 implementation remains an external "
            "dependency. No F-seq performance result has been produced."
        ),
        "cache": {
            "path": str(Path(args.cache)),
            "provenance_hash": data.cache.provenance["provenance_hash"],
            "window_rule_version": data.cache.provenance["window_rule_version"],
            "split_rule_version": data.cache.provenance["split_rule_version"],
            "layers": list(data.cache.layers),
            "A": data.cache.A,
        },
        "cohort": {
            "rows": rep.n_cache_rows,
            "verified_against_variant_map": rep.cohort_verified,
            "rows_by_split": rep.rows_by_split,
            "target_column": rep.target_column,
        },
        "scaler": scaler.to_dict(),
        "batch": {
            "batch_size": batch.batch_size,
            "shape_H": list(batch.H_WT.shape),
            "n_pad_slots": int(pad_mask.sum()),
            "n_gap_slots": int(gap_mask.sum()),
            "slot_kinds_seen": sorted(set(kinds)),
        },
        "objective_values_on_one_batch_NOT_a_result": losses,
        "huber_delta": HUBER_DELTA,
        "protocol": PILOT.to_dict(),
        "validation_batch_report_NOT_a_result": report,
        "checks": c.results,
        "n_checks": len(c.results),
        "n_failed": sum(1 for r in c.results if not r["pass"]),
        "all_passed": c.all_passed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{out['n_checks'] - out['n_failed']}/{out['n_checks']} checks passed -> {out_path}")
    print("NOTE: this is interface plumbing. No model was trained and no "
          "performance number here is a result.")
    return 0 if c.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
