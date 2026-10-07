"""
metrics.py — 지표. 주 지표는 Stage 1 의 select_on="subset" 과 같은 정의.

    subset = mean( Spearman(missense 안), Spearman(indel 안) )

두 그룹을 **합친** 상관이 아니라 그룹별 Spearman 의 평균이다. 합치면 그룹 평균 차이
(missense −3.19 vs indel −9.92)만으로 상관이 부풀려진다. synonymous 는 position 신호가
없어(순열검정 p=0.216) 보고만 하고 해석하지 않는다 (명세 §7.4).
Spearman 을 주 지표로 쓰는 것은 ProteinGym(Notin+ 2023)의 관례와도 같다.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from config import TYPE3

MISSENSE, SYNONYMOUS, INDEL = (TYPE3.index(t) for t in ("missense", "synonymous", "indel"))


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    """평균 순위(ties=average) 의 Pearson = scipy.stats.spearmanr 과 같은 정의."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return np.nan
    return pearson(rankdata(a), rankdata(b))


def _spearman_rank_info(p, y) -> float:
    """그룹 안에서 예측이 상수면(예: type-only 모델) 순위 정보가 없으므로 0 으로 둔다.
    그냥 두면 nan 이 되어 바닥 기준선의 주 지표가 정의되지 않는다."""
    if p.size >= 3 and np.ptp(p) == 0 and np.ptp(y) > 0:
        return 0.0
    return spearman(p, y)


def subset_score(pred, y, typ) -> float:
    """주 지표. missense·indel 각각의 Spearman 평균."""
    pred, y, typ = np.asarray(pred, float), np.asarray(y, float), np.asarray(typ)
    vals = [_spearman_rank_info(pred[typ == g], y[typ == g]) for g in (MISSENSE, INDEL)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if len(vals) == 2 else np.nan


def all_metrics(pred, y, typ) -> dict:
    pred, y, typ = np.asarray(pred, float), np.asarray(y, float), np.asarray(typ)
    out = {"subset": subset_score(pred, y, typ),
           "overall_pearson": pearson(pred, y), "overall_spearman": spearman(pred, y),
           "mse": float(np.mean((pred - y) ** 2))}
    for g, name in enumerate(TYPE3):
        m = typ == g
        out[f"{name}_spearman"] = spearman(pred[m], y[m])
        out[f"{name}_pearson"] = pearson(pred[m], y[m])
    return out
