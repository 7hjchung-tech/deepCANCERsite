"""Paired, position-grouped bootstrap of the validation gain (Stage 2 - Stage 1 subset Spearman)
for the OOF-target Stage 2 runs. Validation only; test is never loaded.
Output: analysis/stage2_oof/val_gain_bootstrap.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage1.engine import group_of  # noqa: E402
from src.stage2.diagnostics import load_setup, loader_for  # noqa: E402
from src.stage2.engine import _summarise, evaluate  # noqa: E402
from src.stage2.model import Stage2Model  # noqa: E402
from src.stage2.reverse import REVERSE_MODES, ReverseFiLMModel  # noqa: E402
from src.stage2.structure import make_qk_tokenizer  # noqa: E402

RUNS = {f"{m}_oof": f"runs/stage2_oof/{m}/seed42" for m in ("single_query", "nine_query", "r0_struct_mean", "r1_seq_query")}
B = 2000

st = load_setup(device="cuda", splits=("train", "val"))
val = st.by_split["val"]
pos = np.array([int(e["row"]["pp"]) for e in val])
ld = loader_for(st, val, batch_size=128)
rng = np.random.default_rng(0)
upos = np.unique(pos)
boots = [rng.choice(upos, size=len(upos), replace=True) for _ in range(B)]
rows_of = {p: np.where(pos == p)[0] for p in upos}
out = []
for name, d in RUNS.items():
    ck = torch.load(ROOT / d / "best_stage2.pt", map_location="cuda", weights_only=False)
    tok = make_qk_tokenizer(st.cfg)
    tok.fit_preprocessing(st.store.raw([e["var_id"] for e in st.by_split["train"]]))
    tok.load_state_dict(ck["tokenizer"])
    mode = ck["mode"]
    m = ReverseFiLMModel(mode) if mode in REVERSE_MODES else Stage2Model(mode, tau_init=ck["cfg"].get("tau_init"))
    m.load_state_dict(ck["stage2"])
    m.to("cuda"); tok.to("cuda")
    r = evaluate(m, tok, st.handle, ld, "cuda")
    y, p2, g = np.array(r["labels"]), np.array(r["preds_stage2"]), np.array(r["groups"])
    # Stage 1 predictions on the same rows = p2 - delta; recompute from a second pass for clarity
    p1 = []
    with torch.no_grad():
        from src.stage2.stage1_adapter import stage1_outputs
        for batch in ld:
            p1.append(stage1_outputs(st.handle, batch, grad=False)["y1"].cpu().numpy())
    p1 = np.concatenate(p1)
    gain = _summarise(y, p2, g)["subset"] - _summarise(y, p1, g)["subset"]
    bs = []
    for samp in boots:
        idx = np.concatenate([rows_of[p] for p in samp])
        bs.append(_summarise(y[idx], p2[idx], g[idx])["subset"] - _summarise(y[idx], p1[idx], g[idx])["subset"])
    bs = np.array(bs)
    resid, delta = y - p1, p2 - p1
    out.append({"run": name, "val_gain": gain, "ci95_lo": np.quantile(bs, 0.025), "ci95_hi": np.quantile(bs, 0.975),
                "p_gain_le_0": float((bs <= 0).mean()), "delta_rms": float(np.sqrt((delta ** 2).mean())),
                "corr_delta_resid_val": float(np.corrcoef(delta, resid)[0, 1]),
                "corr_delta_resid_val_missense": float(np.corrcoef(delta[g == "missense"], resid[g == "missense"])[0, 1])})
    print(out[-1], flush=True)
df = pd.DataFrame(out)
(ROOT / "analysis/stage2_oof").mkdir(parents=True, exist_ok=True)
df.to_csv(ROOT / "analysis/stage2_oof/val_gain_bootstrap.csv", index=False)
print(df.round(4).to_string())
