"""diagnose_stage2.py -- read-only diagnosis of the recorded Stage 2 runs.

Uses a fixed, type-stratified train/val diagnostic sample (IDs saved). Test rows are never
loaded. Existing run directories are only read. Output: analysis/stage2_diag/.

Sections
  audit      run configs, checkpoint hashes, split sizes, attention shapes, N_valid
  keys       (A) per-sample key/value diversity, K decomposition, z_attn vs z_uniform
  queries    (B) query norms/diversity, cos(Q,K) and logit spread, entropy, single vs nine
  tau        fixed Q/K/V, eval mode, tau overridden in {0.69, 0.1, 0.03, 0.01}
  residual   delta_y vs Stage 1 residual, Stage 1 loss on train vs val
  shuffle    position-level structure shuffle on the full validation split

    python diagnose_stage2.py --out analysis/stage2_diag
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage1.engine import group_of  # noqa: E402
from src.stage2.attention import masked_cosine_attention  # noqa: E402
from src.stage2.diagnostics import (  # noqa: E402
    DEFAULT_STAGE1, build_initial, centered_var_ratio, diag_sample, entropy_rows, load_setup,
    load_trained, loader_for, offdiag_cos, sha16, stage1_components,
)
from src.stage2.engine import _move, _summarise  # noqa: E402
from src.stage2.schema import VARIANT_TYPES  # noqa: E402

RUNS = {
    "tau0.69_single": "runs/stage2/single_query/seed42",
    "tau0.69_nine": "runs/stage2/nine_query/seed42",
    "tau0.1_single": "runs/stage2_tau0.1/single_query/seed42",
    "tau0.1_nine": "runs/stage2_tau0.1/nine_query/seed42",
}


def q(x, ps=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    d = {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}
    d.update({f"p{int(p * 100):02d}": float(np.quantile(x, p)) for p in ps})
    return d


@torch.no_grad()
def stage2_internals(m, tok, comp: dict, batch: dict, device: str, tau_override=None) -> dict:
    """Recomputes the Stage 2 forward step by step and checks it against Stage2Model.forward."""
    S = tok(_move(batch["struct_raw"], device))
    tid = batch["type_id"].to(device)
    K, V, valid = comp["K"], comp["V"], comp["valid"]
    U = m.adapter(S)
    e = m.type_emb(tid)
    c = U.mean(1) + e
    qv = c.unsqueeze(1) if m.query_mode == "single_query" else U + e.unsqueeze(1)
    tau = m.tau() if tau_override is None else torch.tensor(float(tau_override), device=device)
    cos = torch.einsum("bqd,bnd->bqn", F.normalize(qv, dim=-1, eps=1e-8), F.normalize(K, dim=-1, eps=1e-8))
    out, w = masked_cosine_attention(qv, K, V, valid, tau)
    z = out.squeeze(1) if m.query_mode == "single_query" else out.mean(1)
    vm = valid.unsqueeze(-1)
    z_unif = (V * vm).sum(1) / vm.sum(1).clamp_min(1)
    gamma, beta = m.film(c).chunk(2, -1)
    delta = m.head((1 + gamma) * z + beta).squeeze(-1)
    delta_unif = m.head((1 + gamma) * z_unif + beta).squeeze(-1)
    if tau_override is None:
        ref = m(S, K, V, valid, tid, comp["y1"])
        assert torch.allclose(ref["delta"], delta, atol=1e-5) and torch.allclose(ref["weights"], w, atol=1e-6)
    return {"S": S, "U": U, "e": e, "c": c, "q": qv, "cos": cos, "logits": cos / tau, "w": w, "tau": float(tau),
            "z": z, "z_unif": z_unif, "delta": delta, "delta_unif": delta_unif, "gamma": gamma, "beta": beta}


def per_sample_keys(comp: dict, batch: dict, rows: list, split: str) -> None:
    valid = comp["valid"].bool()
    n_res = comp["n_res"]
    for b, vid in enumerate(batch["var_id"]):
        idx = valid[b].nonzero().squeeze(-1)
        res_idx = idx[idx < n_res]
        K, V = comp["K"][b, idx], comp["V"][b, idx]
        Kr = comp["K"][b, res_idx]
        cK = offdiag_cos(K) if len(idx) > 1 else torch.tensor([float("nan")])
        cKr = offdiag_cos(Kr) if len(res_idx) > 1 else torch.tensor([float("nan")])
        cC = offdiag_cos(comp["content"][b, res_idx]) if len(res_idx) > 1 else torch.tensor([float("nan")])
        cP = offdiag_cos(comp["pe"][b, res_idx]) if len(res_idx) > 1 else torch.tensor([float("nan")])
        sw = comp["s1_weights"][b, idx].cpu().numpy()
        H1, H1n, p1, _ = entropy_rows(sw, len(idx))
        rows.append({
            "split": split, "var_id": vid, "type": VARIANT_TYPES[int(batch["type_id"][b])], "n_valid": len(idx),
            "key_norm_mean": float(K.norm(dim=-1).mean()), "value_norm_mean": float(V.norm(dim=-1).mean()),
            "key_cos_offdiag_mean": float(cK.mean()), "key_cos_offdiag_min": float(cK.min()),
            "reskey_cos_offdiag_mean": float(cKr.mean()), "reskey_cos_offdiag_min": float(cKr.min()),
            "content_cos_offdiag_mean": float(cC.mean()), "pe_cos_offdiag_mean": float(cP.mean()),
            "key_centered_var_ratio": centered_var_ratio(K), "value_centered_var_ratio": centered_var_ratio(V),
            "content_norm_mean": float(comp["content"][b, res_idx].norm(dim=-1).mean()),
            "pe_norm_mean": float(comp["pe"][b, res_idx].norm(dim=-1).mean()),
            "layer_emb_norm": float(comp["layer_emb"][b, 0].norm()),
            "meta_key_norm": float(comp["K"][b, n_res].norm()),
            "stage1_attn_H": H1, "stage1_attn_H_norm": H1n, "stage1_attn_maxp": p1,
            "stage1_attn_meta_weight": float(comp["s1_weights"][b, n_res]),
        })


def per_query_rows(run: str, it: dict, comp: dict, batch: dict, rows: list, split: str) -> None:
    valid = comp["valid"].bool()
    Qn = it["q"].shape[1]
    for b, vid in enumerate(batch["var_id"]):
        idx = valid[b].nonzero().squeeze(-1)
        nv = len(idx)
        qcos = offdiag_cos(it["q"][b]).cpu().numpy() if Qn > 1 else np.array([np.nan])
        scos = offdiag_cos(it["S"][b]).cpu().numpy()
        ucos = offdiag_cos(it["U"][b]).cpu().numpy()
        za, zu = it["z"][b], it["z_unif"][b]
        base = {"run": run, "split": split, "var_id": vid, "type": VARIANT_TYPES[int(batch["type_id"][b])],
                "n_valid": nv, "tau": it["tau"],
                "S_cos_offdiag_mean": float(scos.mean()), "U_cos_offdiag_mean": float(ucos.mean()),
                "query_cos_offdiag_mean": float(np.nanmean(qcos)) if Qn > 1 else float("nan"),
                "U_mean_norm": float(it["U"][b].mean(0).norm()), "e_type_norm": float(it["e"][b].norm()),
                "z_attn_minus_unif_abs": float((za - zu).norm()),
                "z_attn_minus_unif_rel": float((za - zu).norm() / (zu.norm() + 1e-6)),
                "delta": float(it["delta"][b]), "delta_unif": float(it["delta_unif"][b]),
                "delta_attn_minus_unif": float(it["delta"][b] - it["delta_unif"][b])}
        for j in range(Qn):
            cs = it["cos"][b, j, idx].cpu().numpy()
            lg = it["logits"][b, j, idx].cpu().numpy()
            H, Hn, pm, kl = entropy_rows(it["w"][b, j, idx].cpu().numpy(), nv)
            rows.append({**base, "query": j, "q_norm": float(it["q"][b, j].norm()),
                         "cos_mean": float(cs.mean()), "cos_std": float(cs.std()), "cos_min": float(cs.min()),
                         "cos_max": float(cs.max()), "cos_range": float(cs.max() - cs.min()),
                         "logit_std": float(lg.std()), "logit_range": float(lg.max() - lg.min()),
                         "H": H, "H_norm": Hn, "max_p": pm, "kl_uniform": kl})
        if Qn > 1:   # pairwise L1 distance between the 9 query distributions
            W = it["w"][b, :, idx]
            l1 = (W.unsqueeze(0) - W.unsqueeze(1)).abs().sum(-1)
            rows[-1]["nine_pairwise_L1_mean"] = float(l1[~torch.eye(Qn, dtype=torch.bool, device=l1.device)].mean())


def summarise(df: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for c in cols:
            if c in g:
                out.append({**dict(zip(by, key)), "metric": c, **q(g[c].to_numpy())})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/stage2_diag")
    ap.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1)
    ap.add_argument("--n-per-type", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shuffle-reps", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False

    st = load_setup(args.stage1_ckpt, args.device)
    dev = st.device
    report: dict = {}

    # ---------------- audit
    sizes = {s: {g: sum(group_of(e["edit"].edit_type) == g for e in es) for g in ("missense", "synonymous", "indel")}
             for s, es in st.by_split.items()}
    audit = {
        "stage1_ckpt": args.stage1_ckpt, "stage1_ckpt_sha16": sha16(args.stage1_ckpt),
        "stage1_cfg": {k: st.handle.cfg.get(k) for k in ("window_radius", "layers", "init_seed", "bottleneck_dim")},
        "stage1_tau": float(F.softplus(st.handle.model.pooling.log_tau) + 1e-4),
        "manifest_sha16": sha16("data/split_manifest.csv"),
        "structure_table_sha16": sha16("data/structure/results/rad51c_struct_features.csv"),
        "split_sizes_by_type": sizes, "runs": {},
    }
    for name, d in RUNS.items():
        c = json.loads((Path(d) / "config.json").read_text())
        audit["runs"][name] = {"dir": d, "ckpt_sha16": sha16(Path(d) / "best_stage2.pt"),
                               "query_mode": c["cfg"]["query_mode"], "tau_init_cfg": c["cfg"].get("tau_init"),
                               "seed": c["seed"], "stage1_ckpt": c["args"]["stage1_ckpt"],
                               "trainable_params": c["trainable_params"], "best_epoch": c["best_epoch"],
                               "stop_reason": c["stop_reason"]}
    # trainable parameter list of the model exactly as train_stage2 builds it
    m0, t0 = build_initial(st, "nine_query", 42, 0.1)
    audit["trainable_parameters"] = (
        [{"name": f"stage2.{n}", "shape": list(p.shape), "numel": p.numel()} for n, p in m0.named_parameters()]
        + [{"name": f"tokenizer.{n}", "shape": list(p.shape), "numel": p.numel()} for n, p in t0.named_parameters()])
    audit["stage1_trainable_after_freeze"] = sum(p.numel() for p in st.handle.model.parameters() if p.requires_grad)
    audit["stage1_training_flag"] = st.handle.model.training

    val_ids = diag_sample(st.by_split["val"], args.n_per_type, seed=0)
    tr_ids = diag_sample(st.by_split["train"], args.n_per_type, seed=0)
    pd.DataFrame([{"split": s, "var_id": e["var_id"], "type": group_of(e["edit"].edit_type), "pos": e["edit"].u + 1}
                  for s, es in (("train", tr_ids), ("val", val_ids)) for e in es]).to_csv(out / "diag_sample_ids.csv", index=False)

    # shapes on one batch
    b0 = next(iter(loader_for(st, val_ids[:8], batch_size=8)))
    comp0 = stage1_components(st.handle, b0)
    shp = {"K": list(comp0["K"].shape), "V": list(comp0["V"].shape), "valid": list(comp0["valid"].shape),
           "H_wt": list(b0["H_wt"].shape), "n_res_tokens": comp0["n_res"], "meta_token_index": comp0["n_res"]}
    for name in ("tau0.1_single", "tau0.1_nine"):
        m, t, _ = load_trained(st, Path(RUNS[name]) / "best_stage2.pt")
        it = stage2_internals(m, t, comp0, b0, dev)
        shp[f"{name}_weights"] = list(it["w"].shape)
        # softmax axis check: weights sum to 1 over the token axis for every (b, q)
        shp[f"{name}_weight_sum_over_N_maxdev"] = float((it["w"].sum(-1) - 1).abs().max())
    audit["shapes"] = shp

    # ---------------- (A) keys and (B) queries on the diagnostic sample
    trained = {n: load_trained(st, Path(d) / "best_stage2.pt") for n, d in RUNS.items()}
    inits = {"init_tau0.69_single": build_initial(st, "single_query", 42, None),
             "init_tau0.69_nine": build_initial(st, "nine_query", 42, None),
             "init_tau0.1_single": build_initial(st, "single_query", 42, 0.1),
             "init_tau0.1_nine": build_initial(st, "nine_query", 42, 0.1)}
    models = {**{k: (v[0], v[1]) for k, v in trained.items()}, **inits}
    for mm, tt in models.values():
        mm.eval(); tt.eval()
    key_rows, q_rows, tau_rows, pair_rows = [], [], [], []
    heat = {}
    for split, ents in (("train", tr_ids), ("val", val_ids)):
        for batch in loader_for(st, ents, batch_size=64):
            comp = stage1_components(st.handle, batch)
            per_sample_keys(comp, batch, key_rows, split)
            its = {}
            for name, (mm, tt) in models.items():
                its[name] = stage2_internals(mm, tt, comp, batch, dev)
                per_query_rows(name, its[name], comp, batch, q_rows, split)
            for a, bname in (("tau0.69_single", "tau0.69_nine"), ("tau0.1_single", "tau0.1_nine")):
                for i, vid in enumerate(batch["var_id"]):
                    pair_rows.append({"pair": f"{a}|{bname}", "split": split, "var_id": vid,
                                      "type": VARIANT_TYPES[int(batch["type_id"][i])],
                                      "z_diff": float((its[a]["z"][i] - its[bname]["z"][i]).norm()),
                                      "z_norm": float(its[a]["z"][i].norm()),
                                      "delta_a": float(its[a]["delta"][i]), "delta_b": float(its[bname]["delta"][i]),
                                      "pred_diff": float(its[a]["delta"][i] - its[bname]["delta"][i])})
            # tau override (fixed Q/K/V, eval)
            if split == "val":
                for name in ("tau0.1_single", "tau0.1_nine", "tau0.69_nine"):
                    mm, tt = models[name]
                    for tv in (0.69, 0.1, 0.03, 0.01):
                        it = stage2_internals(mm, tt, comp, batch, dev, tau_override=tv)
                        # explicit logit check: softmax(cos/tau) over valid tokens == returned weights
                        manual = torch.softmax(it["cos"].masked_fill(~comp["valid"].bool().unsqueeze(1), -1e30) / tv, -1)
                        err = float((manual * comp["valid"].unsqueeze(1) - it["w"]).abs().max())
                        for i in range(len(batch["var_id"])):
                            idx = comp["valid"][i].bool()
                            for j in range(it["w"].shape[1]):
                                H, Hn, pm, kl = entropy_rows(it["w"][i, j, idx].cpu().numpy(), int(idx.sum()))
                                tau_rows.append({"run": name, "tau": tv, "type": VARIANT_TYPES[int(batch["type_id"][i])],
                                                 "H": H, "H_norm": Hn, "max_p": pm, "delta": float(it["delta"][i]),
                                                 "manual_vs_impl_maxerr": err})
            if split == "val" and len(heat) < 3:
                for i, vid in enumerate(batch["var_id"]):
                    t = VARIANT_TYPES[int(batch["type_id"][i])]
                    # representative = mid-protein sample with a full window (not a sequence-edge case)
                    if t not in heat and 50 <= int(batch["wt_pos"][i].max()) and int(batch["wt_pos"][i][batch["wt_pos"][i] > 0].min()) > 20:
                        idx = comp["valid"][i].bool()
                        Kn = F.normalize(comp["K"][i, idx], dim=-1)
                        heat[t] = {"var_id": vid, "C": (Kn @ Kn.T).cpu().numpy()}
    kdf, qdf, tdf, pdf_ = map(pd.DataFrame, (key_rows, q_rows, tau_rows, pair_rows))
    kdf.to_csv(out / "keys_per_sample.csv", index=False)
    qdf.to_csv(out / "queries_per_sample_query.csv", index=False)
    pdf_.to_csv(out / "single_vs_nine_per_sample.csv", index=False)
    key_cols = [c for c in kdf.columns if c not in ("split", "var_id", "type")]
    summarise(kdf, key_cols, ["split", "type"]).to_csv(out / "keys_summary.csv", index=False)
    summarise(kdf, key_cols, ["split"]).to_csv(out / "keys_summary_all.csv", index=False)
    qcols = ["q_norm", "U_mean_norm", "e_type_norm", "S_cos_offdiag_mean", "U_cos_offdiag_mean", "query_cos_offdiag_mean",
             "cos_mean", "cos_std", "cos_range", "logit_std", "logit_range", "H", "H_norm", "max_p", "kl_uniform",
             "z_attn_minus_unif_abs", "z_attn_minus_unif_rel", "delta", "delta_attn_minus_unif", "nine_pairwise_L1_mean"]
    summarise(qdf, qcols, ["run", "split"]).to_csv(out / "queries_summary.csv", index=False)
    summarise(qdf[qdf.split == "val"], ["H", "H_norm", "max_p", "cos_range"], ["run", "type"]).to_csv(
        out / "queries_summary_by_type_val.csv", index=False)
    summarise(tdf, ["H", "H_norm", "max_p", "delta", "manual_vs_impl_maxerr"], ["run", "tau"]).to_csv(
        out / "tau_override_summary.csv", index=False)
    summarise(pdf_, ["z_diff", "z_norm", "pred_diff"], ["pair", "split"]).to_csv(out / "single_vs_nine_summary.csv", index=False)
    report["heatmap_ids"] = {k: v["var_id"] for k, v in heat.items()}

    # ---------------- residual relation + Stage 1 in-sample check (full train/val, best checkpoints)
    huber = torch.nn.HuberLoss(delta=1.0, reduction="none")
    res_rows = []
    full = {s: loader_for(st, st.by_split[s], batch_size=128) for s in ("train", "val")}
    cache_batches = {s: [] for s in full}
    for s, ld in full.items():
        for batch in ld:
            comp = stage1_components(st.handle, batch)
            y = batch["label"].to(dev)
            rec = {"y": y, "y1": comp["y1"], "type": batch["type_id"], "var_id": list(batch["var_id"])}
            for name in ("tau0.1_single", "tau0.1_nine"):
                mm, tt = models[name]
                rec[name] = stage2_internals(mm, tt, comp, batch, dev)["delta"]
            cache_batches[s].append(rec)
    resid_summary = {}
    for s, recs in cache_batches.items():
        y = torch.cat([r["y"] for r in recs]).cpu().numpy()
        y1 = torch.cat([r["y1"] for r in recs]).cpu().numpy()
        ty = torch.cat([r["type"] for r in recs]).numpy()
        groups = np.array([VARIANT_TYPES[t] for t in ty])
        r = y - y1
        d = {"n": int(len(y)), "stage1_huber": float(huber(torch.tensor(y1), torch.tensor(y)).mean()),
             "stage1_resid_rms": float(np.sqrt((r ** 2).mean())), "stage1_resid_mean": float(r.mean()),
             "stage1_metrics": {k: v for k, v in _summarise(y, y1, groups).items() if k in ("spearman", "subset", "rmse", "by_group")}}
        for name in ("tau0.1_single", "tau0.1_nine"):
            dl = torch.cat([rr[name] for rr in recs]).cpu().numpy()
            d[name] = {"delta": q(dl), "delta_rms": float(np.sqrt((dl ** 2).mean())),
                       "corr_delta_resid_pearson": float(np.corrcoef(dl, r)[0, 1]),
                       "stage2_huber": float(huber(torch.tensor(y1 + dl), torch.tensor(y)).mean())}
            for g in VARIANT_TYPES:
                mk = groups == g
                d[name][f"corr_delta_resid_{g}"] = float(np.corrcoef(dl[mk], r[mk])[0, 1])
        resid_summary[s] = d
        res_rows.append(pd.DataFrame({"split": s, "var_id": [v for rr in recs for v in rr["var_id"]], "type": groups, "y": y, "y1": y1, "resid": r,
                                      **{name: torch.cat([rr[name] for rr in recs]).cpu().numpy()
                                         for name in ("tau0.1_single", "tau0.1_nine")}}))
    pd.concat(res_rows).to_csv(out / "residual_per_sample.csv", index=False)
    report["residual"] = resid_summary

    # ---------------- position-level structure shuffle (full validation split)
    val_entries = st.by_split["val"]
    pos = np.array([e["edit"].u + 1 for e in val_entries])
    vids = [e["var_id"] for e in val_entries]
    real_raw = st.store.raw(vids)
    rng = np.random.default_rng(0)
    shuf = []
    for name in ("tau0.1_single", "tau0.1_nine"):
        mm, tt = models[name]

        def run_val(raw_by_vid):
            preds, ys, gs, deltas = [], [], [], []
            for batch in loader_for(st, val_entries, batch_size=128):
                if raw_by_vid is not None:
                    ii = [raw_by_vid[v] for v in batch["var_id"]]
                    batch["struct_raw"] = {k: real_raw[k][ii] for k in real_raw}
                comp = stage1_components(st.handle, batch)
                it = stage2_internals(mm, tt, comp, batch, dev)
                preds.append((comp["y1"] + it["delta"]).cpu().numpy()); deltas.append(it["delta"].cpu().numpy())
                ys.append(batch["label"].numpy()); gs += [VARIANT_TYPES[t] for t in batch["type_id"].tolist()]
            return np.concatenate(preds), np.concatenate(ys), np.array(gs), np.concatenate(deltas)

        p_real, y, g, d_real = run_val(None)
        base = _summarise(y, p_real, g)["subset"]
        uniq = np.unique(pos)
        for rep in range(args.shuffle_reps):
            # permute positions as whole 9-feature blocks; derangement retried until no position maps to itself
            for _ in range(100):
                perm = rng.permutation(uniq)
                if not np.any(perm == uniq):
                    break
            pmap = dict(zip(uniq, perm))
            first_row_at = {}
            for i, p_ in enumerate(pos):
                first_row_at.setdefault(p_, i)
            mapping = {v: first_row_at[pmap[p_]] for v, p_ in zip(vids, pos)}
            p_s, _, _, d_s = run_val(mapping)
            shuf.append({"run": name, "rep": rep, "val_subset_real": base,
                         "val_subset_shuffled": _summarise(y, p_s, g)["subset"],
                         "mean_abs_pred_change": float(np.abs(p_s - p_real).mean()),
                         "delta_rms_real": float(np.sqrt((d_real ** 2).mean())),
                         "delta_rms_shuffled": float(np.sqrt((d_s ** 2).mean()))})
    pd.DataFrame(shuf).to_csv(out / "structure_shuffle_val.csv", index=False)

    # ---------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(heat), figsize=(4.2 * len(heat), 3.6))
    for ax, (t, h) in zip(np.atleast_1d(axes), heat.items()):
        im = ax.imshow(h["C"], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"{t}: {h['var_id']}\nkey cosine (last row/col = meta)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out / "key_cosine_heatmaps.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    v = qdf[qdf.split == "val"]
    for name in ("init_tau0.1_nine", "tau0.1_nine", "tau0.1_single", "tau0.69_single"):
        axes[0].hist(v[v.run == name]["cos_range"], bins=40, histtype="step", label=name)
        axes[1].hist(v[v.run == name]["H_norm"].dropna(), bins=40, histtype="step", label=name)
    axes[0].set_xlabel("per-query range of cos(q, K_i) over valid tokens"); axes[1].set_xlabel("H / log(N_valid)")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "cos_range_and_entropy.png", dpi=130); plt.close(fig)

    tsum = pd.read_csv(out / "tau_override_summary.csv")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name, g_ in tsum[tsum.metric == "H_norm"].groupby("run"):
        ax.plot(g_["tau"], g_["mean"], marker="o", label=name)
    ax.set_xscale("log"); ax.set_xlabel("tau (override, fixed Q/K/V)"); ax.set_ylabel("mean H / log(N_valid)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "tau_override_entropy.png", dpi=130); plt.close(fig)

    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (out / "report_numbers.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()
