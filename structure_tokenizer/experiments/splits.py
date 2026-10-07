"""
splits.py — position 기반 nested CV.

바깥(outer)  : position 을 K 등분. fold 하나가 평가용 test. 선택에는 절대 안 쓴다.
안쪽(inner) : 바깥 train 의 position 을 다시 K_in 등분. 하이퍼파라미터 선택 전용.
(Cawley & Talbot, JMLR 2010; Varma & Simon, BMC Bioinformatics 2006)

같은 position 의 변이는 Block A 가 동일해 독립이 아니므로, 행이 아니라 position 을 나눈다
(ProteinGym modulo/contiguous 분할과 같은 원칙; Grimm+ 2015 의 circularity 방지).
"""
from __future__ import annotations

import hashlib

import numpy as np

from config import PROTOCOL


def _chunks(items: np.ndarray, k: int, seed: int) -> list:
    perm = np.random.default_rng(seed).permutation(np.sort(items))
    return [np.sort(c) for c in np.array_split(perm, k)]


def outer_fold_positions(positions: np.ndarray, repeat: int, k: int | None = None) -> list:
    k = k or PROTOCOL["outer_k"]
    return _chunks(np.unique(positions), k, PROTOCOL["fold_seed0"] + repeat)


def inner_fold_positions(train_positions: np.ndarray, repeat: int, fold: int,
                         k_in: int | None = None) -> list:
    k_in = k_in or PROTOCOL["inner_k"]
    return _chunks(np.unique(train_positions), k_in,
                   PROTOCOL["fold_seed0"] + 10_000 + 100 * repeat + fold)


def rows_of(pos: np.ndarray, positions) -> np.ndarray:
    return np.flatnonzero(np.isin(pos, positions))


def nested_split(pos: np.ndarray, repeat: int, fold: int, k: int | None = None,
                 k_in: int | None = None) -> dict:
    """한 바깥 fold 의 행 인덱스 묶음.

    반환: outer_train, outer_test, inner = [(inner_train, inner_val), ...]
    """
    outer = outer_fold_positions(pos, repeat, k)
    test_p = outer[fold]
    train_p = np.setdiff1d(np.unique(pos), test_p)
    inner = []
    for j, val_p in enumerate(inner_fold_positions(train_p, repeat, fold, k_in)):
        tr_p = np.setdiff1d(train_p, val_p)
        inner.append((rows_of(pos, tr_p), rows_of(pos, val_p)))
    out = {"outer_train": rows_of(pos, train_p), "outer_test": rows_of(pos, test_p),
           "inner": inner, "test_positions": test_p, "train_positions": train_p}
    _check(pos, out)
    return out


def _check(pos: np.ndarray, s: dict) -> None:
    tr, te = s["outer_train"], s["outer_test"]
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(pos[tr], pos[te])) == 0, "바깥 train/test 가 position 을 공유"
    seen = []
    for itr, iva in s["inner"]:
        assert len(np.intersect1d(pos[itr], pos[iva])) == 0, "안쪽 train/val 이 position 공유"
        assert len(np.intersect1d(iva, te)) == 0 and len(np.intersect1d(itr, te)) == 0, \
            "안쪽 분할에 바깥 test 가 섞였다"
        assert set(itr) | set(iva) == set(tr)
        seen.append(iva)
    allv = np.concatenate(seen)
    assert len(allv) == len(tr) and set(allv) == set(tr), "안쪽 val 이 바깥 train 을 정확히 한 번씩 덮지 않는다"


def fingerprint(pos: np.ndarray, repeat: int) -> str:
    """분할 식별자. 다른 실험과 같은 분할을 썼는지 대조할 때 쓴다."""
    folds = outer_fold_positions(pos, repeat)
    s = "|".join(",".join(map(str, f)) for f in folds)
    return hashlib.sha256(s.encode()).hexdigest()[:16]
