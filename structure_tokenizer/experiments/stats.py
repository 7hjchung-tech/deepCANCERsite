"""
stats.py — 판정용 통계. 계획서 §6 의 규칙을 구현한다.

  · paired difference 의 CI: **position 단위 cluster bootstrap** (Field & Welsh 2007).
    같은 position 의 변이는 Block A 가 같아 독립이 아니므로 position 을 재표집한다.
    한 재표집 안에서 두 모델의 지표를 **같은 행으로** 계산해 차를 낸다(paired).
    지표는 repeat×seed 마다 따로 계산해 평균한다 = "학습 1회의 기대 성능" 기준.
  · 보조 검정: fold 수준 **corrected resampled t-test** (Nadeau & Bengio 2003).
    CV fold 끼리 train 이 겹쳐 차이들이 양의 상관을 가지므로 분산에 (1/J + n_test/n_train) 보정.
  · 다중비교: Holm (1979).
  · 예산 곡선: Dodge+ 2019 의 "n 번 탐색했을 때 기대 최고 val 성능".
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from metrics import INDEL, MISSENSE, _spearman_rank_info


def group_spearmans(pred, y, typ):
    """(missense ρ, indel ρ) — subset 은 둘의 평균."""
    return (_spearman_rank_info(pred[typ == MISSENSE], y[typ == MISSENSE]),
            _spearman_rank_info(pred[typ == INDEL], y[typ == INDEL]))


def _scores(oof, y, typ, rows, fold_of):
    """oof [R,S,N] → (subset, missense, indel). **바깥 fold 마다 따로 계산해 평균**한다.

    fold 를 합쳐서 순위 지표를 계산하면, fold 마다 모델의 예측 척도·절편이 달라 생기는
    차이가 가짜 상관을 만든다 (Forman & Scholz 2010, SIGKDD Explorations).
    실제로 type-only 모델이 합친 계산에서 0 이 아닌 −0.07 이 나와 이 문제를 확인했다.
    fold_of [R,N]: repeat 별 행의 바깥 fold 번호.
    """
    out = []
    for r in range(oof.shape[0]):
        fr = fold_of[r, rows]
        for k in np.unique(fr):
            rk = rows[fr == k]
            for s in range(oof.shape[1]):
                m, i = group_spearmans(oof[r, s, rk], y[rk], typ[rk])
                out.append((0.5 * (m + i), m, i))
    return np.nanmean(np.array(out), axis=0)


def _cluster_rows(pos, rng, idx_by_pos, uniq):
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([idx_by_pos[p] for p in picked])


def bootstrap(oofs: dict, y, typ, pos, fold_of, pairs: list, n_boot=2000, seed=0) -> dict:
    """여러 모델의 단독 CI 와 여러 쌍의 paired 차이 CI 를 **같은 재표집**으로 한 번에.

    position 을 복원추출한다. 한 position 은 repeat 마다 정확히 한 fold 에 속하므로
    재표집된 행을 fold 별로 다시 나눠 fold 별 지표를 계산할 수 있다.
    반환: {"single": {name: {metric: (point, lo, hi)}}, "paired": {(A,B): {metric: dict}}}
    """
    names = sorted(set(oofs) | {a for p in pairs for a in p})
    all_rows = np.arange(len(y))
    point = {n: _scores(oofs[n], y, typ, all_rows, fold_of) for n in names}
    uniq = np.unique(pos)
    idx_by_pos = {p: np.flatnonzero(pos == p) for p in uniq}
    rng = np.random.default_rng(seed)
    boots = {n: np.empty((n_boot, 3)) for n in names}
    for b in range(n_boot):
        rows = _cluster_rows(pos, rng, idx_by_pos, uniq)
        for n in names:
            boots[n][b] = _scores(oofs[n], y, typ, rows, fold_of)
    keys = ("subset", "missense", "indel")
    single = {n: {k: (point[n][j], *np.nanpercentile(boots[n][:, j], [2.5, 97.5]))
                  for j, k in enumerate(keys)} for n in names}
    paired = {}
    for A, B in pairs:
        paired[(A, B)] = {}
        for j, k in enumerate(keys):
            d = boots[A][:, j] - boots[B][:, j]
            d = d[np.isfinite(d)]
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            lo, hi = np.percentile(d, [2.5, 97.5])
            paired[(A, B)][k] = {"diff": point[A][j] - point[B][j], "lo": lo, "hi": hi,
                                 "se": d.std(), "p_boot": p}
    return {"single": single, "paired": paired}


def fold_diffs(oofA, oofB, y, typ, test_rows_by_rk: dict) -> np.ndarray:
    """fold (r,k) 마다 subset 차이 (seed 평균)."""
    out = []
    for (r, k), rows in sorted(test_rows_by_rk.items()):
        a = np.nanmean([np.mean(group_spearmans(oofA[r, s, rows], y[rows], typ[rows]))
                        for s in range(oofA.shape[1])])
        b = np.nanmean([np.mean(group_spearmans(oofB[r, s, rows], y[rows], typ[rows]))
                        for s in range(oofB.shape[1])])
        out.append(a - b)
    return np.array(out)


def corrected_ttest(diffs: np.ndarray, k: int) -> dict:
    """Nadeau & Bengio (2003). J = fold×repeat 개의 차이, n_test/n_train = 1/(k-1)."""
    J = len(diffs)
    m, v = diffs.mean(), diffs.var(ddof=1)
    se = np.sqrt((1.0 / J + 1.0 / (k - 1)) * v)
    t = m / se if se > 0 else np.nan
    p = 2 * sps.t.sf(abs(t), df=J - 1) if np.isfinite(t) else np.nan
    return {"mean": m, "se": se, "t": t, "p": p, "J": J}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (n - i) * p))
        out[k] = running
    return out


def expected_max(values, ns) -> dict:
    """Dodge+ 2019: 관측된 N 개 trial 점수의 경험분포에서 n 개를 뽑을 때 최댓값의 기댓값."""
    v = np.sort(np.asarray([x for x in values if np.isfinite(x)]))
    N = len(v)
    i = np.arange(1, N + 1)
    return {n: float(np.sum(v * ((i / N) ** n - ((i - 1) / N) ** n))) for n in ns if N}
