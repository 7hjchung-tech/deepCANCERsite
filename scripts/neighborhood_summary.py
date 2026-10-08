"""Collects E0/E1/E2 results (best + last) and position-grouped paired bootstrap contrasts
E1-E0, E2-E1, E2-E0 on val subset Spearman. Validation only; test never touched.
Output -> analysis/stage2_neighborhood/{summary.csv, contrasts_bootstrap.csv, learning_curves.png}
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage1.checkpoint import default_reference  # noqa: E402
from src.stage1.engine import group_of  # noqa: E402
from src.stage2.diagnostics import DEFAULT_STAGE1, load_setup  # noqa: E402
from src.stage2.neighbor_engine import _summarise, forward_batch, make_loader  # noqa: E402
from src.stage2.neighbor_model import NeighborhoodFiLMModel  # noqa: E402
from src.stage2.neighbor_structure import ResidueStructureTokenizer, WTNeighborStore  # noqa: E402
from src.stage2.stage1_adapter import load_stage1_handle, stage1_outputs  # noqa: E402

OUT = ROOT / "analysis/stage2_neighborhood"
RUN = ROOT / "runs/stage2_neighborhood"
CACHE = ROOT / "data/structure/results/wt_neighbor_cache.npz"
B = 2000


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def subset_metric(y, p, g):
    vals = [spearman(y[g == t], p[g == t]) for t in ("missense", "indel") if (g == t).sum() >= 2]
    return float(np.mean(vals)) if vals else float("nan")


def preds_for(cond: str, which: str, st, handle) -> dict:
    if cond == "E0":
        store = WTNeighborStore(CACHE)
        loader = make_loader(st.by_split["val"], st.cache, st.window, st.layers, handle, store, "e1", 128, False)
        y, p, g, ids = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                s1 = stage1_outputs(handle, batch, grad=False)
                y.append(batch["label"].numpy()); p.append(s1["y1"].cpu().numpy())
                g += [group_of(t) for t in batch["edit_type"]]; ids += batch["var_id"]
        return {"y": np.concatenate(y), "p": np.concatenate(p), "g": np.array(g), "ids": ids}
    ck = torch.load(RUN / cond / "seed42" / f"{which}.pt", map_location="cuda", weights_only=False)
    tok = ResidueStructureTokenizer(d_s=32, n_bins=4)
    store = WTNeighborStore(CACHE)
    train_anchors = [int(e["row"]["pp"]) for e in st.by_split["train"]]
    tok.fit_preprocessing(store.raw_for_positions(store.fit_positions_for_train_anchors(train_anchors)))
    tok.load_state_dict(ck["tokenizer"])
    model = NeighborhoodFiLMModel(d=ck["d_model"])
    model.load_state_dict(ck["model"])
    model.to("cuda").eval(); tok.to("cuda").eval()
    mode = "e1" if cond == "E1" else "e2"
    loader = make_loader(st.by_split["val"], st.cache, st.window, st.layers, handle, store, mode, 128, False)
    y, p, g, ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out, s1 = forward_batch(model, tok, handle, batch, "cuda")
            y.append(batch["label"].numpy()); p.append(out["pred"].cpu().numpy())
            g += [group_of(t) for t in batch["edit_type"]]; ids += batch["var_id"]
    return {"y": np.concatenate(y), "p": np.concatenate(p), "g": np.array(g), "ids": ids}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_setup(DEFAULT_STAGE1, "cuda", splits=("train", "val"))
    ref = default_reference(st.cfg.get("esm_checkpoint", "esm2_t33_650M_UR50D"), st.cache.wt_hash)
    handle = load_stage1_handle(DEFAULT_STAGE1, ref, "cuda")
    pos = np.array([int(e["row"]["pp"]) for e in st.by_split["val"]])

    rows, cache = [], {}
    for cond in ("E0", "E1", "E2"):
        whichs = ["best"] if cond == "E0" else ["best", "last"]
        for which in whichs:
            d = preds_for(cond, which, st, handle)
            cache[(cond, which)] = d
            m = _summarise(d["y"], d["p"], d["g"])
            rows.append({"condition": cond, "which": which, "val_subset": m["subset"],
                        "val_spearman_all": m["spearman"], "val_rmse": m["rmse"], "val_mae": m["mae"],
                        "val_missense": m["by_group"].get("missense"), "val_synonymous": m["by_group"].get("synonymous"),
                        "val_indel": m["by_group"].get("indel")})
    cache[("E0", "last")] = cache[("E0", "best")]   # E0 has no training; "last" == "best" == epoch 0
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.round(4).to_string())

    upos = np.unique(pos)
    rows_of = {p: np.where(pos == p)[0] for p in upos}
    rng = np.random.default_rng(0)
    boots = [rng.choice(upos, size=len(upos), replace=True) for _ in range(B)]

    def contrast(which, a, b):
        da, db = cache[(a, which)], cache[(b, which)]
        assert da["ids"] == db["ids"]
        point = subset_metric(da["y"], da["p"], da["g"]) - subset_metric(db["y"], db["p"], db["g"])
        diffs, n_failed = [], 0
        for samp in boots:
            idx = np.concatenate([rows_of[p] for p in samp])
            yb, gb = da["y"][idx], da["g"][idx]
            if (gb == "missense").sum() < 2 or (gb == "indel").sum() < 2:
                n_failed += 1; continue
            diffs.append(subset_metric(yb, da["p"][idx], gb) - subset_metric(yb, db["p"][idx], gb))
        diffs = np.array(diffs)
        return {"which": which, "contrast": f"{a}-{b}", "point": point, "ci95_lo": float(np.quantile(diffs, .025)),
               "ci95_hi": float(np.quantile(diffs, .975)), "p_le_0": float((diffs <= 0).mean()),
               "n_valid": int(len(diffs)), "n_failed": n_failed}

    crows = []
    for which in ("best", "last"):
        crows.append(contrast(which, "E1", "E0"))
        crows.append(contrast(which, "E2", "E1"))
        crows.append(contrast(which, "E2", "E0"))
    cdf = pd.DataFrame(crows)
    cdf.to_csv(OUT / "contrasts_bootstrap.csv", index=False)
    print(cdf.round(4).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"E1": "tab:blue", "E2": "tab:red"}
    for cond in ("E1", "E2"):
        h = json.loads((RUN / cond / "seed42" / "history.json").read_text())
        ep = [x["epoch"] for x in h["history"]]
        val = [x["val_stage2_subset"] for x in h["history"]]
        ax.plot(ep, val, marker=".", color=colors[cond], label=f"{cond} (best@{h['best']['epoch']})")
    ax.axhline(cache[("E0", "best")]["y"].size and subset_metric(cache[("E0","best")]["y"], cache[("E0","best")]["p"], cache[("E0","best")]["g"]),
               color="k", ls=":", lw=1, label="E0 (Stage1 alone)")
    ax.set_xlabel("epoch"); ax.set_ylabel("val subset Spearman"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=130)
    print(f"[neighborhood-summary] wrote {OUT}")


if __name__ == "__main__":
    main()
