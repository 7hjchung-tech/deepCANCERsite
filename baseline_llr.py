"""
baseline_llr.py  —  P4: the "must-beat" ESM-2 zero-shot baseline.

THE IDEA
--------
Before building a fancy model, we measure how well a SIMPLE, training-free
signal already predicts the SGE functional score. That signal is the ESM-2
LLR (log-likelihood ratio):

    LLR = logP(mutant amino acid | context) - logP(wild-type amino acid | context)

A very negative LLR means "ESM-2 thinks the mutant is unnatural here" -> likely
damaging. If our trained model can't beat this number, it isn't earning its keep.

We report, on the held-out val/test splits:
  * Spearman / Pearson correlation between LLR and the SGE z-score (regression)
  * ROC-AUC for separating depleted (loss-of-function) vs unchanged (classification)

EFFICIENCY
----------
LLR needs, for each position, the model's predicted log-prob of all 20 amino
acids when that position is masked. Many variants share a position, so we mask
each UNIQUE position ONCE (~375 forward passes) and look up every variant from
that — instead of one forward pass per variant (~4,500).

HOW TO RUN
----------
    .venv/bin/python baseline_llr.py
First run downloads the ESM-2 model (~600 MB) and then runs on CPU (a few
minutes). Results are saved to data/baseline_llr.csv.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"   # laptop-friendly; 650M is stronger
MANIFEST = "data/split_manifest.csv"
WT_PATH = "data/wt_sequence.txt"
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


# --------------------------------------------------------------------------
# 1. Load model + tokenizer once.
# --------------------------------------------------------------------------
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[model] loading {MODEL_NAME} on {device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    return tok, model, torch.device(device)


# --------------------------------------------------------------------------
# 2. For ONE masked position, return {amino_acid: log_prob} for the 20 AAs.
# --------------------------------------------------------------------------
@torch.no_grad()
def logprobs_at_position(tok, model, device, seq, pos_1based):
    residues = list(seq)
    residues[pos_1based - 1] = tok.mask_token       # hide the residue
    masked = "".join(residues)
    inputs = tok(masked, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0]              # [tokens, vocab]
    # find the masked token's position in the tokenized input
    mask_idx = (inputs["input_ids"][0] == tok.mask_token_id).nonzero()[0].item()
    logp = torch.log_softmax(logits[mask_idx], dim=-1).float().cpu()
    return {aa: float(logp[tok.convert_tokens_to_ids(aa)]) for aa in AA20}


# --------------------------------------------------------------------------
# 3. Compute LLR for every missense variant (reusing per-position log-probs).
# --------------------------------------------------------------------------
def compute_llr(df, wt_seq, tok, model, device):
    # Only missense has a single-residue WT->mut substitution that LLR is
    # defined for. (Inframe indels would need a different score.)
    mis = df[df["slim_consequence"] == "missense"].copy()
    positions = sorted(mis["pp"].astype(int).unique())
    print(f"[llr] {len(mis)} missense variants across {len(positions)} positions")

    cache = {}
    for i, p in enumerate(positions, 1):
        cache[p] = logprobs_at_position(tok, model, device, wt_seq, p)
        if i % 50 == 0 or i == len(positions):
            print(f"  masked {i}/{len(positions)} positions")

    def llr(row):
        p = int(row["pp"])
        lp = cache[p]
        wt = wt_seq[p - 1]                 # actual WT residue at this position
        mut = str(row["alt_aa"])
        if mut not in lp or wt not in lp:  # e.g. WT='*'? shouldn't happen here
            return np.nan
        return lp[mut] - lp[wt]

    mis["llr"] = mis.apply(llr, axis=1)
    return mis


# --------------------------------------------------------------------------
# 4. Metrics per split.
# --------------------------------------------------------------------------
def report(mis):
    print("\n========== BASELINE RESULTS (ESM-2 LLR) ==========")
    for split in ("val", "test"):
        s = mis[(mis["split"] == split) & mis["llr"].notna() &
                mis["z_score_D4_D14"].notna()]
        x = s["llr"].to_numpy()
        y = s["z_score_D4_D14"].to_numpy()

        rho = spearmanr(x, y).statistic
        r = pearsonr(x, y).statistic

        # Classification: depleted (fast/slow) = 1, unchanged = 0. Drop enriched.
        cls = s[s["functional_classification"].isin(
            ["fast depleted", "slow depleted", "unchanged"])].copy()
        cls["is_depleted"] = cls["functional_classification"].isin(
            ["fast depleted", "slow depleted"]).astype(int)
        # depleted variants have LOWER z and (expected) more-negative LLR, so we
        # use -LLR as the "damaging" score for AUC.
        auc = roc_auc_score(cls["is_depleted"], -cls["llr"])

        print(f"\n[{split}]  n={len(s)}")
        print(f"  Spearman rho (LLR vs z) = {rho:+.3f}")
        print(f"  Pearson  r   (LLR vs z) = {r:+.3f}")
        print(f"  ROC-AUC depleted-vs-unchanged = {auc:.3f}  (n={len(cls)})")
    print("\n(These are the numbers the trained model must beat.)")


if __name__ == "__main__":
    wt_seq = open(WT_PATH).read().strip()
    df = pd.read_csv(MANIFEST)
    tok, model, device = load_model()
    mis = compute_llr(df, wt_seq, tok, model, device)
    mis[["var_id", "split", "pp", "ref_aa", "alt_aa", "llr",
         "z_score_D4_D14", "functional_classification"]].to_csv(
        "data/baseline_llr.csv", index=False)
    print("[save] wrote data/baseline_llr.csv")
    report(mis)
