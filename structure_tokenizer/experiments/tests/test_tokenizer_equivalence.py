"""실험용 StructModel 과 배포용 ../tokenizer.py 가 같은 토큰을 만드는지 (Qk, 양 끝 clip)."""
import os
import sys

import numpy as np
import pytest
import torch

import config as C
from data import load_data
from encoders import fit_bins, pad_bins
from model import StructModel

sys.path.append(C.TOK_DIR)
import tokenizer as T  # noqa: E402


@pytest.mark.parametrize("n_bins", [2, 4, 7])
def test_same_bins_and_tokens(n_bins):
    d = load_data()
    tr = np.flatnonzero(d.split == "train")
    old = fit_bins("quantile_plddt", d.cont[tr], None, {"n_bins": n_bins})
    new = T.fit_qk_bins(d.cont[tr], n_bins)
    assert all(np.allclose(a, b) for a, b in zip(old, new))
    torch.manual_seed(0)
    sm = StructModel(1, "ple", 32, bins=pad_bins([old]), extrapolate=False)
    tok = T.StructureTokenizer(new, 32)
    with torch.no_grad():
        sm.ln_weight.normal_()
        sm.ln_bias.normal_()
        tok.value_w.copy_(sm.value_w[0]); tok.ss_emb.copy_(sm.ss_emb[0]); tok.field_emb.copy_(sm.field_emb[0])
        tok.norm.weight.copy_(sm.ln_weight[0]); tok.norm.bias.copy_(sm.ln_bias[0])
        a = sm.tokens(torch.as_tensor(d.cont, dtype=torch.float32)[None], torch.as_tensor(d.ss)[None])[0]
        b = tok(torch.as_tensor(d.cont, dtype=torch.float32), torch.as_tensor(d.ss))
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
