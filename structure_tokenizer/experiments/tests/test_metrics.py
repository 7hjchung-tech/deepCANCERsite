import numpy as np
from scipy.stats import pearsonr, spearmanr

from metrics import INDEL, MISSENSE, pearson, spearman, subset_score


def test_against_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500); b = a + rng.normal(size=500)
    a[:50] = a[0]                                                  # 동점 포함
    assert abs(spearman(a, b) - spearmanr(a, b)[0]) < 1e-12
    assert abs(pearson(a, b) - pearsonr(a, b)[0]) < 1e-12


def test_foldwise_scoring_removes_offset_artifact():
    """type-only 처럼 fold 마다 상수만 다른 예측: fold 를 합치면 가짜 상관, fold 별이면 정확히 0."""
    from stats import _scores
    rng = np.random.default_rng(2)
    n = 1000
    typ = np.where(rng.random(n) < 0.8, MISSENSE, INDEL)
    fold = rng.integers(0, 5, n)
    offset = np.array([0.3, -0.2, 0.1, 0.5, -0.4])
    y = rng.normal(size=n) - 2 * offset[fold]                    # fold 난이도가 예측 절편과 역상관
    pred = offset[fold] + np.where(typ == INDEL, -10, 0)         # fold 마다 절편만 다른 상수 예측
    pooled = subset_score(pred, y, typ)
    fw = _scores(pred[None, None], y, typ, np.arange(n), fold[None])[0]
    assert abs(pooled) > 0.2 and fw == 0.0, (pooled, fw)


def test_subset_is_mean_of_within_group_not_pooled():
    rng = np.random.default_rng(1)
    typ = np.r_[np.full(400, MISSENSE), np.full(60, INDEL), np.full(100, 1)]
    y = rng.normal(size=len(typ)) + np.where(typ == INDEL, -10, 0)
    pred = np.where(typ == INDEL, -10, 0) + rng.normal(size=len(typ))   # 그룹 평균만 맞춘 예측
    s = subset_score(pred, y, typ)
    manual = np.mean([spearman(pred[typ == g], y[typ == g]) for g in (MISSENSE, INDEL)])
    assert abs(s - manual) < 1e-12
    m = (typ == MISSENSE) | (typ == INDEL)
    assert spearman(pred[m], y[m]) > s + 0.2, "합친 상관은 그룹 평균 차이로 부풀려진다"
