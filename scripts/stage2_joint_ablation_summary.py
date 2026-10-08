"""Summarises the R1 joint-fine-tuning ablation (A/B/C/D, runs/stage2_joint_ablation/seed42).
Validation only; test was never evaluated by train_stage2_r1_joint_ablation.py.
Outputs -> analysis/stage2_joint_ablation/
  summary.csv            per-condition best/last metrics, Stage1 weight change, L2-SP penalty
  contrasts.csv           B-A, B-C, A-D, (B-C)-(A-D) on val subset Spearman (best and last)
  contrasts_bootstrap.csv position-grouped paired bootstrap of each contrast's val subset gap
  learning_curves.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage1.checkpoint import default_reference  # noqa: E402
from src.stage1.engine import group_of  # noqa: E402
from src.stage2.diagnostics import DEFAULT_STAGE1, load_setup, loader_for, make_qk_tokenizer  # noqa: E402
from src.stage2.engine import _summarise, evaluate  # noqa: E402
from src.stage2.reverse import ReverseFiLMModel  # noqa: E402
from src.stage2.stage1_adapter import load_stage1_handle  # noqa: E402
from src.stage2.structure import make_fixed_structure_store  # noqa: E402

IN_ROOT = ROOT / "runs/stage2_joint_ablation/seed42"
OUT = ROOT / "analysis/stage2_joint_ablation"
OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["A", "B", "C", "D"]
B = 2000


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def subset_metric(y, p, g):
    vals = [spearman(y[g == t], p[g == t]) for t in ("missense", "indel") if (g == t).sum() >= 2]
    return float(np.mean(vals)) if vals else float("nan")


def load_condition_model(cond: str, which: str, st):
    ck = torch.load(IN_ROOT / cond / f"{which}_stage2.pt", map_location="cuda", weights_only=False)
    tok = make_qk_tokenizer(ck["cfg"])
    tok.fit_preprocessing(st.store.raw([e["var_id"] for e in st.by_split["train"]]))
    tok.load_state_dict(ck["tokenizer"])
    model = ReverseFiLMModel("r1_seq_query")
    model.load_state_dict(ck["stage2"])
    ref = default_reference(st.cfg.get("esm_checkpoint", "esm2_t33_650M_UR50D"), st.cache.wt_hash)
    handle = load_stage1_handle(ck["stage1_ckpt"], ref, "cuda")
    for name, val in ck["stage1_unfrozen"].items():
        mod_name, pname = name.split(".", 1)
        sub = getattr(handle.model, mod_name)
        sub.state_dict()[pname].copy_(val)   # in-place load of just the unfrozen tensors
    return model.to("cuda").eval(), tok.to("cuda").eval(), handle, ck


def main() -> None:
    st = load_setup(DEFAULT_STAGE1, "cuda", splits=("train", "val"))
    train_ids = [e["var_id"] for e in st.by_split["train"]]
    fixed_store = make_fixed_structure_store(st.store, train_ids)
    y = {"val": np.array([float(e["row"]["z_score_D4_D14"]) for e in st.by_split["val"]])}
    g = {"val": np.array([group_of(e["edit"].edit_type) for e in st.by_split["val"]])}
    pos_val = np.array([int(e["row"]["pp"]) for e in st.by_split["val"]])

    rows, curves = [], {}
    pred_cache = {}   # (cond, which) -> (val_subset recomputed, per-sample preds) for bootstrap/contrasts
    for cond in CONDS:
        h = json.loads((IN_ROOT / cond / "history.json").read_text())
        c = json.loads((IN_ROOT / cond / "config.json").read_text())
        hist = h["history"]
        best_ep, last_ep = h["best"]["epoch"], h["last_epoch"]
        curves[cond] = pd.DataFrame([{"epoch": x["epoch"], "val_subset": x["val_stage2_subset"]} for x in hist])
        by_ep = {x["epoch"]: x for x in hist}
        for which, ep in (("best", best_ep), ("last", last_ep)):
            x = by_ep[ep]
            bg = x.get("val_by_group_spearman", {})
            store = fixed_store if c["spec"]["structure"] == "fixed" else st.store
            m, tok, handle, ck = load_condition_model(cond, which, st)
            ld = loader_for(st, st.by_split["val"], 128, shuffle=False, store_override=store, handle_override=handle)
            r = evaluate(m, tok, handle, ld, "cuda")
            p2 = np.array(r["preds_stage2"])
            assert _summarise(y["val"], p2, g["val"])["subset"] - x["val_stage2_subset"] < 1e-3 or which != "best" or ep == 0, \
                "re-inference does not match the logged validation subset at the selected epoch"
            pred_cache[(cond, which)] = p2
            rows.append({
                "condition": cond, "spec_stage1": c["spec"]["stage1"], "spec_structure": c["spec"]["structure"],
                "which": which, "epoch": ep, "epoch0_won": c["epoch0_won"] if which == "best" else None,
                "val_subset": x["val_stage2_subset"], "val_spearman_all": x["val_stage2_spearman"],
                "val_rmse": x["val_stage2_rmse"], "val_mae": x.get("val_stage2_mae"),
                "val_spearman_missense": bg.get("missense"), "val_spearman_synonymous": bg.get("synonymous"),
                "val_spearman_indel": bg.get("indel"), "delta_abs_mean": x.get("delta_abs_mean"),
                "train_task_loss": x.get("train_task_loss"), "train_l2sp_penalty": x.get("train_l2sp_penalty"),
                "stage1_grad_norm_mean": x.get("train_stage1_grad_norm_mean"),
                "n_stage1_unfrozen": c["n_stage1_unfrozen_params"],
                "stage1_weight_change_l2": c.get(f"stage1_weight_change_l2_{which}_vs_init"),
                "stop_reason": c["stop_reason"], "lambda_sp": c["lambda_sp"], "lambda_sp_reduction": c["lambda_sp_reduction"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.round(4).to_string())

    # contrasts on val subset Spearman, best and last, point estimates + bootstrap CI
    upos = np.unique(pos_val)
    rows_of = {pp: np.where(pos_val == pp)[0] for pp in upos}
    rng = np.random.default_rng(0)
    boots = [rng.choice(upos, size=len(upos), replace=True) for _ in range(B)]

    def contrast(which: str, a: str, b: str) -> dict:
        pa, pb = pred_cache[(a, which)], pred_cache[(b, which)]
        point = subset_metric(y["val"], pa, g["val"]) - subset_metric(y["val"], pb, g["val"])
        diffs, n_failed = [], 0
        for samp in boots:
            idx = np.concatenate([rows_of[pp] for pp in samp])
            yb, gb = y["val"][idx], g["val"][idx]
            if (gb == "missense").sum() < 2 or (gb == "indel").sum() < 2:
                n_failed += 1
                continue
            diffs.append(subset_metric(yb, pa[idx], gb) - subset_metric(yb, pb[idx], gb))
        diffs = np.array(diffs)
        return {"which": which, "contrast": f"{a}-{b}", "point": point,
               "ci95_lo": float(np.quantile(diffs, 0.025)), "ci95_hi": float(np.quantile(diffs, 0.975)),
               "p_le_0": float((diffs <= 0).mean()), "n_valid": int(len(diffs)), "n_failed": n_failed}

    crows = []
    for which in ("best", "last"):
        c_ba = contrast(which, "B", "A")
        c_bc = contrast(which, "B", "C")
        c_ad = contrast(which, "A", "D")
        crows += [c_ba, c_bc, c_ad]
        # (B-C) - (A-D): difference of two differences; bootstrap jointly for a valid CI
        pa, pb, pc, pd_ = (pred_cache[(x, which)] for x in ("A", "B", "C", "D"))
        point = (subset_metric(y["val"], pb, g["val"]) - subset_metric(y["val"], pc, g["val"])) - \
                (subset_metric(y["val"], pa, g["val"]) - subset_metric(y["val"], pd_, g["val"]))
        diffs, n_failed = [], 0
        for samp in boots:
            idx = np.concatenate([rows_of[pp] for pp in samp])
            yb, gb = y["val"][idx], g["val"][idx]
            if (gb == "missense").sum() < 2 or (gb == "indel").sum() < 2:
                n_failed += 1
                continue
            bc = subset_metric(yb, pb[idx], gb) - subset_metric(yb, pc[idx], gb)
            ad = subset_metric(yb, pa[idx], gb) - subset_metric(yb, pd_[idx], gb)
            diffs.append(bc - ad)
        diffs = np.array(diffs)
        crows.append({"which": which, "contrast": "(B-C)-(A-D)", "point": point,
                      "ci95_lo": float(np.quantile(diffs, 0.025)), "ci95_hi": float(np.quantile(diffs, 0.975)),
                      "p_le_0": float((diffs <= 0).mean()), "n_valid": int(len(diffs)), "n_failed": n_failed})
    cdf = pd.DataFrame(crows)
    cdf.to_csv(OUT / "contrasts_bootstrap.csv", index=False)
    print(cdf.round(4).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = {"A": "tab:blue", "B": "tab:red", "C": "tab:orange", "D": "tab:green"}
    for cond, c in curves.items():
        ax.plot(c.epoch, c.val_subset, marker=".", color=colors[cond], label=f"{cond} ({CONDS_LABEL[cond]})")
    stage1_line = curves["A"].val_subset.iloc[0]
    ax.axhline(stage1_line, color="k", ls=":", lw=1, label="Stage 1 (epoch 0, all conditions)")
    ax.set_xlabel("epoch"); ax.set_ylabel("val subset Spearman"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=130)
    print(f"[joint-ablation-summary] wrote {OUT}")


CONDS_LABEL = {"A": "frozen+real", "B": "joint+real", "C": "joint+fixed", "D": "frozen+fixed"}

if __name__ == "__main__":
    main()
