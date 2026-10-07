"""데이터 로드·분할·셔플 대조군의 가정 검사."""
import numpy as np

import config as C
from data import load_data, shuffle_structure
from splits import nested_split, outer_fold_positions


def test_load_counts():
    d = load_data()          # 내부 assert: 개수표, position 당 Block A 동일, shipped split 무결성
    assert d.n == C.N_ROWS and len(d.positions()) == C.N_POSITIONS


def test_outer_folds_partition_positions():
    d = load_data()
    for r in range(C.PROTOCOL["repeats"]):
        folds = outer_fold_positions(d.pos, r)
        allp = np.concatenate(folds)
        assert len(allp) == len(set(allp)) == C.N_POSITIONS
        covered = np.zeros(d.n, int)
        for k in range(C.PROTOCOL["outer_k"]):
            sp = nested_split(d.pos, r, k)            # 내부 assert: 누출·덮개 검사
            covered[sp["outer_test"]] += 1
        assert (covered == 1).all(), "각 행은 repeat 당 정확히 한 번 test 여야 한다"


def test_repeats_differ():
    d = load_data()
    a, b = outer_fold_positions(d.pos, 0), outer_fold_positions(d.pos, 1)
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


def test_shuffle_control():
    d = load_data()
    s = shuffle_structure(d, seed=1000)
    assert s.shuffled and (s.y == d.y).all() and (s.pos == d.pos).all() and (s.typ == d.typ).all()
    # 같은 position → 같은 값 유지
    for p in np.unique(d.pos):
        rows = np.flatnonzero(s.pos == p)
        assert (s.cont[rows] == s.cont[rows[0]]).all() and (s.ss[rows] == s.ss[rows[0]]).all()
    # 대부분의 position 이 실제로 다른 값을 받았다
    first = {p: np.flatnonzero(d.pos == p)[0] for p in np.unique(d.pos)}
    changed = np.mean([not np.array_equal(d.cont[i], s.cont[i]) for i in first.values()])
    assert changed > 0.95
