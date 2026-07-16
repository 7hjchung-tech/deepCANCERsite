"""
analysis.py — 구조 feature 진단 + 구조-only 예측 baseline (torch 없이 numpy)

출력: analysis_results.json  (PDF 리포트가 읽어감)
  1) 33개 feature 각각의 z_score 상관 (전체 / sav / del)
  2) junction_orient_cos 진단
  3) Ridge 회귀 baseline (구조 33-dim -> z_score) : train/val/test Pearson r, Spearman
  4) depleted vs unchanged 분류 AUC (구조 예측 점수 기반)
"""

import json
import numpy as np
import pandas as pd
from data.structure.code.block_b import FULL_COLS

X = np.load("rad51c_X.npy")                       # [N,33]
meta = pd.read_csv("rad51c_meta.csv")
y = meta["z_score_D4_D14"].to_numpy(dtype=float)
split = meta["split"].to_numpy()
vtype = meta["var_type"].to_numpy()


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())


def auc(score, label):
    """label 0/1, score 높을수록 label=1 예측. Mann-Whitney AUC."""
    label = np.asarray(label)
    r = pd.Series(score).rank().to_numpy()
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---- 1) feature별 상관 ------------------------------------------------------
def corr_table(mask):
    z = y[mask]
    out = []
    for i, name in enumerate(FULL_COLS):
        col = X[mask, i]
        rho = spearman(col, z) if col.std() > 0 else 0.0
        out.append((name, round(rho, 3)))
    return out

corr_all = corr_table(np.ones(len(X), bool))
corr_del = corr_table(vtype == "del")
corr_sav = corr_table(vtype == "sav")

# ---- 2) junction_orient_cos 진단 -------------------------------------------
j = X[:, FULL_COLS.index("B_junction_orient_cos")]
jmask = vtype == "del"
orient_diag = {
    "nonzero_frac": round(float((j != 0).mean()), 3),
    "del_mean": round(float(j[jmask].mean()), 3),
    "del_std": round(float(j[jmask].std()), 3),
    "del_spearman_z": round(spearman(j[jmask], y[jmask]), 3),
}

# ---- 3) Ridge 회귀 baseline (구조 33-dim -> z_score) ------------------------
tr, va, te = split == "train", split == "val", split == "test"
mu, sd = X[tr].mean(0), X[tr].std(0)
sd[sd == 0] = 1.0
Xs = (X - mu) / sd
ymu = y[tr].mean()
yc = y - ymu

def ridge_fit(lam):
    A = Xs[tr]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yc[tr])
    return w

# val 로 lambda 튜닝
best = None
for lam in [0.1, 1, 3, 10, 30, 100, 300]:
    w = ridge_fit(lam)
    pv = Xs[va] @ w
    r = pearson(pv, yc[va])
    if best is None or r > best[1]:
        best = (lam, r, w)
lam, _, w = best
pred = Xs @ w + ymu

reg = {"lambda": lam}
for tag, mask in [("train", tr), ("val", va), ("test", te)]:
    reg[tag] = {"n": int(mask.sum()),
                "pearson": round(pearson(pred[mask], y[mask]), 3),
                "spearman": round(spearman(pred[mask], y[mask]), 3)}
# test 를 type 별로도
reg["test_by_type"] = {}
for t in ["sav", "del"]:
    mm = te & (vtype == t)
    if mm.sum() > 5:
        reg["test_by_type"][t] = {"n": int(mm.sum()),
                                  "pearson": round(pearson(pred[mm], y[mm]), 3),
                                  "spearman": round(spearman(pred[mm], y[mm]), 3)}

# ---- 4) depleted vs unchanged 분류 AUC (구조 예측 기반) ---------------------
fc = meta["functional_classification"].to_numpy()
is_dep = np.isin(fc, ["slow depleted", "fast depleted"]).astype(int)
keep = np.isin(fc, ["slow depleted", "fast depleted", "unchanged"])
dep_score = -pred                       # 예측 z 가 낮을수록 depleted
cls = {}
for tag, mask in [("all", keep), ("test", keep & te)]:
    m = mask
    cls[tag] = {"n": int(m.sum()), "n_depleted": int(is_dep[m].sum()),
                "auc": round(auc(dep_score[m], is_dep[m]), 3)}

results = {
    "n_total": int(len(X)),
    "counts": {t: int((vtype == t).sum()) for t in ["sav", "del", "ins"]},
    "split_counts": {s: int((split == s).sum()) for s in ["train", "val", "test"]},
    "corr_all": corr_all, "corr_del": corr_del, "corr_sav": corr_sav,
    "orient_diag": orient_diag,
    "regression": reg,
    "classification": cls,
}
json.dump(results, open("analysis_results.json", "w"), ensure_ascii=False, indent=2)

# 콘솔 요약
print(f"Ridge(구조 {X.shape[1]}-dim -> z_score)  lambda={lam}")
for t in ["train", "val", "test"]:
    print(f"  {t:5s} n={reg[t]['n']:4d}  Pearson r={reg[t]['pearson']:+.3f}  Spearman={reg[t]['spearman']:+.3f}")
print("  test by type:", reg["test_by_type"])
print(f"분류 depleted vs unchanged  test AUC={cls['test']['auc']}  (n={cls['test']['n']})")
print(f"junction_orient_cos: del ρ(z)={orient_diag['del_spearman_z']}  std={orient_diag['del_std']}")
print("saved: analysis_results.json")
