"""배치형 모델(M 개 동시)이 모델을 따로 만든 것과 같은지, 토큰 배치·cross-attention 성질이 맞는지."""
import numpy as np
import pytest
import torch

import config as C
from data import load_data
from encoders import prepare
from model import build_model
from train import fit, predict

P = {"d_s": 16, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "n_bins": 4,
     "extrapolate": True, "min_samples_leaf": 8, "min_impurity_decrease": 1e-6,
     "n_frequencies": 6, "sigma": 0.1}


def _setup(variant, M=3, seed=0):
    d = load_data()
    rng = np.random.default_rng(seed)
    rows = [rng.choice(d.n, 2000, replace=False) for _ in range(M)]
    prep = prepare(variant, P, d.cont, d.y, rows)
    torch.manual_seed(seed)
    m = build_model(M, variant, P, prep, 32, 16)
    X = torch.as_tensor(prep.X)
    ss = torch.as_tensor(d.ss); ty = torch.as_tensor(d.typ)
    return d, rows, prep, m, X, ss, ty


def _slice_model(m, prep, variant, k):
    """M 개짜리 모델에서 k 번째만 떼어낸 M=1 모델."""
    from encoders import Prepared
    bins = None if prep.bins is None else {key: v[k:k + 1] for key, v in prep.bins.items()}
    one = build_model(1, variant, P, Prepared(prep.X[k:k + 1], prep.ynorm[k:k + 1],
                      prep.y_mu[k:k + 1], prep.y_sd[k:k + 1], bins), 32, 16)
    with torch.no_grad():
        for (n1, p1), (n, p) in zip(one.named_parameters(), m.named_parameters()):
            assert n1 == n
            p1.copy_(p[k:k + 1])
    return one


@pytest.mark.parametrize("variant", ["L", "Q", "Qk", "T", "PLR"])
def test_batched_equals_individual(variant):
    d, rows, prep, m, X, ss, ty = _setup(variant)
    m.eval()
    idx = torch.as_tensor(np.stack([r[:50] for r in rows]))
    ar = torch.arange(m.M)[:, None]
    out = m(X[ar, idx], ss[idx], ty[idx])
    for k in range(m.M):
        one = _slice_model(m, prep, variant, k).eval()
        o1 = one(X[k:k + 1, idx[k]][None][0], ss[idx[k]][None], ty[idx[k]][None])
        torch.testing.assert_close(out[k:k + 1], o1, rtol=1e-5, atol=1e-5)


def test_models_are_independent_in_backward():
    d, rows, prep, m, X, ss, ty = _setup("Qk")
    idx = torch.as_tensor(np.stack([r[:64] for r in rows]))
    ar = torch.arange(m.M)[:, None]
    out = m(X[ar, idx], ss[idx], ty[idx])
    out[0].sum().backward()                               # 모델 0 의 손실만
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.all(p.grad[1:] == 0), f"{n}: 다른 모델로 gradient 가 샌다"


def test_token_order_and_ss_position():
    d, rows, prep, m, X, ss, ty = _setup("Qk")
    idx = torch.as_tensor(np.stack([r[:20] for r in rows]))
    ar = torch.arange(m.M)[:, None]
    t0 = m.tokens(X[ar, idx], ss[idx])
    t1 = m.tokens(X[ar, idx], (ss[idx] + 1) % 3)
    changed = (t0 - t1).abs().amax(dim=(0, 1, 3)) > 0
    assert changed.tolist() == [i == C.SS_TOKEN for i in range(C.N_TOKENS)]
    # pLDDT(연속 0번)만 바꾸면 토큰 0 만 바뀐다
    X2 = X.clone(); X2[..., C.PLDDT_COL] += 7.0
    t2 = m.tokens(X2[ar, idx], ss[idx])
    changed = (t0 - t2).abs().amax(dim=(0, 1, 3)) > 1e-7
    assert changed.tolist() == [i == 0 for i in range(C.N_TOKENS)]


def test_cross_attention_probe():
    """2026-09-30 설계: 변이유형별 query → 토큰 9개. 가중치 합 1, 유형이 바뀌면 가중치가 바뀜,
    구조가 같으면(같은 position) 같은 유형끼리 같은 가중치."""
    d, rows, prep, m, X, ss, ty = _setup("Qk", M=2)
    m.eval()
    idx = torch.as_tensor(np.stack([r[:100] for r in rows]))
    ar = torch.arange(m.M)[:, None]
    o, w = m.cross_attend(X[ar, idx], ss[idx], ty[idx], return_weights=True)
    assert w.shape == (2, 100, m.n_heads, 9)
    torch.testing.assert_close(w.sum(-1), torch.ones_like(w.sum(-1)))
    o2, w2 = m.cross_attend(X[ar, idx], ss[idx], (ty[idx] + 1) % 3, return_weights=True)
    assert (w - w2).abs().max() > 1e-4, "변이유형 query 가 가중치에 영향을 줘야 한다"
    # 같은 position 의 같은 유형 변이 → 같은 토큰·같은 query → 같은 가중치
    same = [(i, j) for i in range(100) for j in range(i + 1, 100)
            if d.pos[rows[0][i]] == d.pos[rows[0][j]] and d.typ[rows[0][i]] == d.typ[rows[0][j]]]
    for i, j in same[:10]:
        torch.testing.assert_close(w[0, i], w[0, j])


def test_early_stopping_restores_best_state():
    d, rows, prep, m, X, ss, ty = _setup("Qk", M=2)
    va = [np.setdiff1d(np.arange(d.n), r)[:500] for r in rows]
    yn = torch.as_tensor(prep.ynorm)
    res = fit(m, X, ss, ty, yn, rows, va, lr=3e-3, weight_decay=1e-4, batch_size=128,
              max_epochs=15, patience=3, fixed_epochs=None, seed=0, device="cpu")
    pv = predict(m, X, ss, ty, va, "cpu")
    for k in range(2):
        mse = float(np.mean((pv[k] - prep.ynorm[k, va[k]]) ** 2))
        assert abs(mse - res.best_val_mse[k]) < 1e-4, (mse, res.best_val_mse[k])
