"""E2 — PLE 구현 검증 (실험용 model.py 기준). 원저자 공식 구현(rtdl_num_embeddings)과 수치로 대조한다."""
import numpy as np
import pytest
import torch
from rtdl_num_embeddings import PiecewiseLinearEncoding, compute_bins

import config as C
from data import load_data
from encoders import fit_bins, pad_bins, quantile_bins, tree_bins
from model import StructModel


def _real_like(seed=0, n=3000):
    """우리 데이터처럼 동점(같은 position 반복)이 많은 값."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 8.0, size=300)
    return base[rng.integers(0, 300, n)]


@pytest.mark.parametrize("n_bins", [2, 3, 4, 16, 64])
def test_quantile_bins_match_official(n_bins):
    x = _real_like()
    ours = quantile_bins(x, n_bins)
    ref = compute_bins(torch.as_tensor(x[:, None], dtype=torch.float64), n_bins=n_bins)[0]
    np.testing.assert_allclose(ours, ref.numpy(), rtol=0, atol=1e-9)


def test_tree_bins_match_official():
    rng = np.random.default_rng(1)
    x = _real_like(2)
    y = np.where(x > 20, -1.0, 0.5) + rng.normal(0, 0.3, len(x))
    ours = tree_bins(x, y, n_bins=8, min_samples_leaf=16, min_impurity_decrease=1e-4)
    ref = compute_bins(torch.as_tensor(x[:, None], dtype=torch.float64), n_bins=8,
                       tree_kwargs={"min_samples_leaf": 16, "min_impurity_decrease": 1e-4,
                                    "random_state": 0},
                       y=torch.as_tensor(y), regression=True)[0]
    # 공식 구현은 트리 임계값을 torch.as_tensor(list) 로 float32 로 바꿔 저장한다 → float32 정밀도로 대조
    assert len(ours) == len(ref)
    np.testing.assert_allclose(ours, ref.numpy(), rtol=1e-6, atol=0)


def _model_with_bins(bins_list, extrapolate):
    pb = pad_bins([bins_list])
    return StructModel(1, "ple", 8, "linear", bins=pb, extrapolate=extrapolate), pb


def test_ple_extrapolate_matches_official():
    """원논문 식(1)·공식 구현: 양 끝 구간은 선형 외삽 (e1 ≤ 0, eT ≥ 1)."""
    rng = np.random.default_rng(3)
    bins_list = [quantile_bins(_real_like(s), nb) for s, nb in
                 zip(range(8), [2, 3, 4, 5, 8, 16, 1, 64])]
    m, pb = _model_with_bins(bins_list, extrapolate=True)
    lo = np.array([b[0] for b in bins_list]); hi = np.array([b[-1] for b in bins_list])
    x = rng.uniform(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo), size=(500, 8))  # 범위 밖 포함
    ours = m.ple(torch.as_tensor(x, dtype=torch.float32)[None])[0]             # [B,8,T]
    ours_flat = torch.cat([ours[:, f, :pb["n_bins"][0, f]] for f in range(8)], dim=1)
    ref = PiecewiseLinearEncoding([torch.as_tensor(b, dtype=torch.float32) for b in bins_list])
    np.testing.assert_allclose(ours_flat.numpy(), ref(torch.as_tensor(x, dtype=torch.float32)).numpy(),
                               rtol=1e-5, atol=1e-5)
    assert (ours_flat.numpy() < 0).any() and (ours_flat.numpy() > 1).any(), "외삽 값이 실제로 나와야 함"


def test_ple_clip_mode_is_old_spec():
    """extrapolate=False 는 명세 §4.1 의 clip((x-b_{t-1})/(b_t-b_{t-1}), 0, 1)."""
    rng = np.random.default_rng(4)
    bins_list = [quantile_bins(_real_like(s), 4) for s in range(8)]
    m, pb = _model_with_bins(bins_list, extrapolate=False)
    x = rng.uniform(-10, 80, size=(300, 8))
    ours = m.ple(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
    for f, b in enumerate(bins_list):
        ref = np.clip((x[:, f, None] - b[:-1]) / np.diff(b), 0, 1)
        np.testing.assert_allclose(ours[:, f, :len(b) - 1], ref, rtol=1e-5, atol=1e-5)
        assert (ours[:, f, len(b) - 1:] == 0).all(), "패딩 칸은 0"


def test_bins_use_train_rows_only():
    d = load_data()
    rng = np.random.default_rng(5)
    tr = rng.choice(d.n, 3000, replace=False)
    other = np.setdiff1d(np.arange(d.n), tr)
    yn = (d.y - d.y[tr].mean()) / d.y[tr].std()
    for mode, p in (("quantile", {"n_bins": 8}),
                    ("tree", {"n_bins": 8, "min_samples_leaf": 8, "min_impurity_decrease": 1e-6})):
        a = fit_bins(mode, d.cont[tr], yn[tr], p)
        cont2, y2 = d.cont.copy(), yn.copy()
        cont2[other] += 1000.0; y2[other] = -y2[other]          # train 밖을 망가뜨려도
        b = fit_bins(mode, cont2[tr], y2[tr], p)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)


def test_plddt_domain_knots():
    d = load_data()
    bl = fit_bins("quantile_plddt", d.cont, d.y, {"n_bins": 16})
    np.testing.assert_array_equal(bl[C.PLDDT_COL], C.PLDDT_DOMAIN_BINS)
    assert len(bl[1]) > 5, "나머지 feature 는 분위수 경계"


def test_synthetic_kink_recovery():
    """정답을 아는 합성 데이터: pLDDT 70 에서 꺾이는 반응을 PLE 는 복원, 스칼라 선형은 못 함."""
    rng = np.random.default_rng(6)
    x = rng.uniform(25, 99, 4000)
    y_true = np.where(x < 70, 0.0, -0.15 * (x - 70))
    y = y_true + rng.normal(0, 0.2, len(x))
    b = C.PLDDT_DOMAIN_BINS
    E = np.clip((x[:, None] - b[:-1]) / np.diff(b), 0, 1)
    A = np.c_[E, np.ones(len(x))]
    fit_ple = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    A2 = np.c_[x, np.ones(len(x))]
    fit_lin = A2 @ np.linalg.lstsq(A2, y, rcond=None)[0]
    r2 = lambda f: 1 - np.mean((y_true - f) ** 2) / np.var(y_true)
    assert r2(fit_ple) > 0.99, r2(fit_ple)
    assert r2(fit_lin) < 0.90, r2(fit_lin)
