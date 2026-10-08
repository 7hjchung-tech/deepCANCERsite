"""instrument_stage2.py -- short instrumented reproduction of a Stage 2 frozen_stage1 run.

Replays train_stage2.py / src.stage2.engine.train_stage2 with the same data, seed, init order,
optimizer, loss, clipping and batch order, and records what the normal run does not:
  * per-module gradient norm (after backward, before clipping), clip events, post-clip norm
  * per-module update norm per step, and distance from the initial parameters per epoch
  * tau (and log_tau), delta_y / gamma / beta statistics, train vs val task loss next to the
    frozen Stage 1 loss on the same rows
  * Stage 1 weight hash before/after and its training flag
  * wall-clock split: data loading, Stage 1 forward, Stage 2 forward/backward/step, validation
    (CUDA-synchronised), peak GPU memory
Epoch 0 (before any update) is evaluated too. Test is never evaluated. Nothing is saved
except logs; existing run directories are not touched.

    python instrument_stage2.py --query-mode nine_query --epochs 11 --out runs/stage2_diag/instrumented/nine_query
"""
from __future__ import annotations

import argparse
import hashlib
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

from src.stage1.engine import set_all_seeds  # noqa: E402
from src.stage2.diagnostics import DEFAULT_STAGE1, build_initial, load_setup, loader_for  # noqa: E402
from src.stage2.engine import _move, evaluate  # noqa: E402
from src.stage2.stage1_adapter import stage1_outputs  # noqa: E402


def module_groups(stage2: nn.Module, tok: nn.Module) -> dict[str, list[nn.Parameter]]:
    g = {"tokenizer": list(tok.parameters()), "adapter": list(stage2.adapter.parameters()),
         "type_emb": list(stage2.type_emb.parameters()), "log_tau": [stage2.log_tau],
         "film_in": list(stage2.film[0].parameters()), "film_out": list(stage2.film[2].parameters()),
         "head_in": list(stage2.head[0].parameters()), "head_out": list(stage2.head[2].parameters())}
    all_ids = {id(p) for p in list(stage2.parameters()) + list(tok.parameters())}
    assert all_ids == {id(p) for ps in g.values() for p in ps}, "module groups do not cover all trainable params"
    return g


def gnorm(ps) -> float:
    gs = [p.grad.detach().pow(2).sum() for p in ps if p.grad is not None]
    return float(torch.stack(gs).sum().sqrt()) if gs else 0.0


def pnorm_diff(ps, ref) -> float:
    return float(torch.stack([(p.detach() - r).pow(2).sum() for p, r in zip(ps, ref)]).sum().sqrt())


def s1_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


@torch.no_grad()
def attention_sensitivity(stage2, tok, st, loader, dev) -> dict:
    """Validation: how much the prediction depends on the attention pattern.
    delta with learned attention vs delta with z replaced by the masked uniform mean of V."""
    from diagnose_stage2 import stage2_internals
    from src.stage2.diagnostics import stage1_components
    stage2.eval(); tok.eval()
    dd, d_all, hn, mp = [], [], [], []
    for batch in loader:
        comp = stage1_components(st.handle, batch)
        it = stage2_internals(stage2, tok, comp, batch, dev)
        dd.append((it["delta"] - it["delta_unif"]).abs().cpu()); d_all.append(it["delta"].cpu())
        nv = comp["valid"].sum(-1, keepdim=True)
        H = -(it["w"] * it["w"].clamp_min(1e-12).log()).sum(-1)
        hn.append((H / nv.log().clamp_min(1e-12)).cpu()); mp.append(it["w"].max(-1).values.cpu())
    dd, d_all = torch.cat(dd), torch.cat(d_all)
    return {"val_delta_rms": float(d_all.pow(2).mean().sqrt()), "val_abs_delta_attn_minus_unif_mean": float(dd.mean()),
            "val_attn_H_norm_mean": float(torch.cat([h.flatten() for h in hn]).mean()),
            "val_attn_maxp_mean": float(torch.cat([m.flatten() for m in mp]).mean())}


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-mode", required=True, choices=("single_query", "nine_query"))
    ap.add_argument("--epochs", type=int, default=11)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dense-steps", type=int, default=30, help="log every step for the first N steps")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    st = load_setup(args.stage1_ckpt, dev)
    cfg = st.cfg
    bs = int(cfg["batch_size"])
    # loaders are built before seeding, exactly as in train_stage2.py
    loaders = {"train": loader_for(st, st.by_split["train"], bs, shuffle=True),
               "train_eval": loader_for(st, st.by_split["train"], 128, shuffle=False),
               "val": loader_for(st, st.by_split["val"], bs, shuffle=False)}
    stage2, tok = build_initial(st, args.query_mode, args.seed, cfg.get("tau_init"))
    set_all_seeds(args.seed)                    # train_stage2() seeds again before the loop
    params = list(stage2.parameters()) + list(tok.parameters())
    opt = torch.optim.AdamW([{"params": params, "lr": cfg["lr"]}], weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.HuberLoss(delta=float(cfg["huber_delta"]))
    groups = module_groups(stage2, tok)
    init = {k: [p.detach().clone() for p in ps] for k, ps in groups.items()}
    s1_before, s1_training_before = s1_hash(st.handle.model), st.handle.model.training
    n_total = sum(p.numel() for p in params) + sum(p.numel() for p in st.handle.model.parameters())
    n_trainable = sum(p.numel() for p in params if p.requires_grad)

    def eval_split(name):
        r = evaluate(stage2, tok, st.handle, loaders[name], dev)
        y = np.array(r["labels"]); p2 = np.array(r["preds_stage2"])
        hub = lambda a: float(nn.functional.huber_loss(torch.tensor(a), torch.tensor(y), delta=1.0))  # noqa: E731
        # stage 1 predictions are y_final - delta; recover them from the same pass
        return r, hub(p2)

    step_rows, epoch_rows = [], []
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    step = 0
    for epoch in range(0, args.epochs + 1):
        t = {"data": 0.0, "stage1_fwd": 0.0, "stage2_fwd_bwd_step": 0.0}
        n_batches = 0
        if epoch > 0:
            stage2.train(); tok.train()
            sync(dev); t_end = time.perf_counter()
            for batch in loaders["train"]:
                sync(dev); t_data = time.perf_counter(); t["data"] += t_data - t_end
                s1 = stage1_outputs(st.handle, batch, grad=False)
                sync(dev); t_s1 = time.perf_counter(); t["stage1_fwd"] += t_s1 - t_data
                # identical to engine.forward_batch, split so the Stage 1 part can be timed separately
                S = tok(_move(batch["struct_raw"], dev))
                fwd = stage2(S, s1["K"], s1["V"], s1["valid"], batch["type_id"].to(dev), s1["y1"])
                y = batch["label"].to(dev)
                loss = loss_fn(fwd["pred"], y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                pre = {k: gnorm(ps) for k, ps in groups.items()}
                total_pre = float(torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], float(cfg["grad_clip"])))
                post_total = gnorm(params)
                before = {k: [p.detach().clone() for p in ps] for k, ps in groups.items()}
                opt.step()
                sync(dev); t_end = time.perf_counter(); t["stage2_fwd_bwd_step"] += t_end - t_s1
                step += 1; n_batches += 1
                if step <= args.dense_steps or step % args.log_every == 0:
                    row = {"step": step, "epoch": epoch, "loss": float(loss), "stage1_loss": float(loss_fn(s1["y1"], y)),
                           "grad_total_preclip": total_pre, "grad_total_postclip": post_total,
                           "clipped": total_pre > float(cfg["grad_clip"]), "tau": float(stage2.tau()),
                           "log_tau": float(stage2.log_tau)}
                    for k in groups:
                        row[f"grad_{k}"] = pre[k]
                        row[f"update_{k}"] = pnorm_diff(groups[k], before[k])
                    step_rows.append(row)
                elif total_pre > float(cfg["grad_clip"]):
                    step_rows.append({"step": step, "epoch": epoch, "clipped": True, "grad_total_preclip": total_pre})
        sync(dev); tv = time.perf_counter()
        val, val_hub = eval_split("val")
        sync(dev); t_val = time.perf_counter() - tv
        tr, tr_hub = eval_split("train_eval")
        row = {"epoch": epoch, "n_batches": n_batches, "steps_total": step, **{f"sec_{k}": round(v, 2) for k, v in t.items()},
               "sec_val": round(t_val, 2),
               "val_subset": val["stage2"]["subset"], "val_stage1_subset": val["stage1"]["subset"],
               "val_huber": val_hub, "train_subset": tr["stage2"]["subset"], "train_stage1_subset": tr["stage1"]["subset"],
               "train_huber": tr_hub, "tau": float(stage2.tau()), "val_entropy_mean": float(np.mean(val["attention_entropy_mean"])),
               "val_gamma_std": val["gamma_std"], "val_beta_std": val["beta_std"], "val_delta_abs_mean": val["delta_abs_mean"]}
        for k in groups:
            row[f"dist_from_init_{k}"] = pnorm_diff(groups[k], init[k])
        row.update(attention_sensitivity(stage2, tok, st, loaders["val"], dev))
        epoch_rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
               if k in ("epoch", "val_subset", "train_subset", "train_huber", "val_huber", "tau", "sec_data",
                        "sec_stage1_fwd", "sec_stage2_fwd_bwd_step", "sec_val")}, flush=True)

    # Stage 1 Huber on the same rows for reference (constant across epochs)
    s1_rows = {}
    for name in ("train_eval", "val"):
        ys, y1s = [], []
        with torch.no_grad():
            for batch in loaders[name]:
                ys.append(batch["label"]); y1s.append(stage1_outputs(st.handle, batch, grad=False)["y1"].cpu())
        s1_rows[name] = float(nn.functional.huber_loss(torch.cat(y1s), torch.cat(ys), delta=1.0))

    summary = {
        "query_mode": args.query_mode, "seed": args.seed, "stage1_ckpt": args.stage1_ckpt, "tau_init": cfg.get("tau_init"),
        "device": dev, "gpu": torch.cuda.get_device_name(0) if dev.startswith("cuda") else None,
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if dev.startswith("cuda") else None,
        "n_train": len(st.by_split["train"]), "n_val": len(st.by_split["val"]), "batch_size": bs,
        "batches_per_epoch": epoch_rows[1]["n_batches"] if len(epoch_rows) > 1 else None,
        "grad_accumulation": 1, "drop_last": False, "num_workers": 0,
        "optimizer_steps_total": step, "params_total_incl_stage1": n_total, "params_trainable": n_trainable,
        "stage1_hash_before": s1_before, "stage1_hash_after": s1_hash(st.handle.model),
        "stage1_training_flag_before": s1_training_before, "stage1_training_flag_after": st.handle.model.training,
        "stage1_requires_grad_any": any(p.requires_grad for p in st.handle.model.parameters()),
        "stage1_huber_train": s1_rows["train_eval"], "stage1_huber_val": s1_rows["val"],
        "esm_forward_in_loop": False, "esm_note": "Stage1Dataset reads cached ESM hidden states (raw_cache.pt, mmap); ESM is never run",
    }
    torch.save({"stage2": stage2.state_dict(), "tokenizer": tok.state_dict(), "query_mode": args.query_mode,
                "cfg": cfg, "epoch": args.epochs, "note": "LAST epoch state (not validation-selected)"},
               out / "last_state.pt")
    pd.DataFrame(step_rows).to_csv(out / "steps.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out / "epochs.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
