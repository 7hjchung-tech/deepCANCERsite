"""Linear probe: do Block A structure features explain the Stage 1 residual (y - y1)?

Reads analysis/stage2_diag/residual_per_sample.csv (train/val only; test never used) and the
structure table. Ridge on [8 continuous A_* features, ss one-hot, type one-hot], standardised
with train statistics.
  (a) fit on TRAIN residuals (Stage 1 in-sample) -> score on VAL residuals
  (b) 5-fold cross-fit inside VAL (diagnostic only; no model is selected with it)
Output: analysis/stage2_diag/residual_probe.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage2.schema import CONTINUOUS_COLUMNS, SS_COLUMNS  # noqa: E402

res = pd.read_csv(ROOT / "analysis/stage2_diag/residual_per_sample.csv")
st = pd.read_csv(ROOT / "data/structure/results/rad51c_struct_features.csv")
df = res.merge(
    st[["var_id", "split", *CONTINUOUS_COLUMNS, *SS_COLUMNS]], on="var_id", suffixes=("", "_tab"))
assert (df["split"] == df["split_tab"]).all()
X_cols = list(CONTINUOUS_COLUMNS) + list(SS_COLUMNS)
for t in ("missense", "synonymous", "indel"):
    df[f"type_{t}"] = (df["type"] == t).astype(float)
X_cols_t = X_cols + ["type_missense", "type_synonymous", "type_indel"]
tr, va = df[df.split == "train"], df[df.split == "val"]
mu, sd = tr[X_cols_t].mean(), tr[X_cols_t].std().replace(0, 1)
Z = lambda d, cols: ((d[cols] - mu[cols]) / sd[cols]).to_numpy()  # noqa: E731
alphas = np.logspace(-2, 4, 25)
out = {}
for name, cols in (("type_only", ["type_missense", "type_synonymous", "type_indel"]), ("structure+type", X_cols_t)):
    m = RidgeCV(alphas=alphas).fit(Z(tr, cols), tr["resid"])
    p = m.predict(Z(va, cols))
    within = {t: float(spearmanr(p[va.type == t], va["resid"][va.type == t])[0]) for t in ("missense", "indel")}
    out[f"a_train_to_val::{name}"] = {"pearson": float(pearsonr(p, va["resid"])[0]),
                                      "spearman": float(spearmanr(p, va["resid"])[0]),
                                      "within_type_spearman": within, "alpha": float(m.alpha_),
                                      "mse_val": float(((p - va["resid"]) ** 2).mean()),
                                      "mse_val_zero": float((va["resid"] ** 2).mean())}
    pv = np.zeros(len(va))
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(va)):
        mm = RidgeCV(alphas=alphas).fit(Z(va.iloc[a], cols), va["resid"].iloc[a])
        pv[b] = mm.predict(Z(va.iloc[b], cols))
    out[f"b_val_crossfit::{name}"] = {"pearson": float(pearsonr(pv, va["resid"])[0]),
                                      "spearman": float(spearmanr(pv, va["resid"])[0]),
                                      "within_type_spearman": {t: float(spearmanr(pv[va.type == t], va["resid"][va.type == t])[0])
                                                               for t in ("missense", "indel")}}
(ROOT / "analysis/stage2_diag/residual_probe.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
