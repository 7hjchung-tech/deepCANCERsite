"""
data.py — 5,887 변이 + Block A 9개 값. 로드할 때 가정을 전부 assert 로 검사한다.

입력 파일은 기존 structtok/build_dataset_v2.py 가 만든 v2_dataset.csv 를 그대로 복사한 것
(AlphaFold 구조에서 Block A 를 뽑는 부분은 검증이 끝났으므로 다시 만들지 않는다).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from config import (CONT_FIELDS, DATA_PATH, DATA_SHA256, EXPECTED_COUNTS, N_POSITIONS,
                    N_ROWS, POS, SS_CLASSES, SS_FIELD, TARGET, TYPE3)


@dataclass(frozen=True)
class Data:
    cont: np.ndarray      # [N, 8] float64, 열 순서 = CONT_FIELDS (원래 단위: pLDDT, 비율, Å)
    ss: np.ndarray        # [N] int64, SS_CLASSES 인덱스
    typ: np.ndarray       # [N] int64, TYPE3 인덱스
    y: np.ndarray         # [N] float64, z_score_D4_D14
    pos: np.ndarray       # [N] int64, 앵커 residue 번호 (1..376)
    split: np.ndarray     # [N] str, 배포된(shipped) 분할 train/val/test
    var_id: np.ndarray    # [N] str
    shuffled: bool = False

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def type3(self) -> np.ndarray:
        return np.asarray(TYPE3)[self.typ]

    def positions(self) -> np.ndarray:
        return np.unique(self.pos)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(path: str = DATA_PATH, check_hash: bool = True) -> Data:
    if check_hash:
        got = _sha256(path)
        assert got == DATA_SHA256, f"데이터 파일이 바뀌었다: {got}"
    df = pd.read_csv(path)

    # --- 가정 검사 -----------------------------------------------------------
    assert len(df) == N_ROWS, f"행 수 {len(df)} != {N_ROWS}"
    assert df[CONT_FIELDS + [SS_FIELD, TARGET, POS, "type3", "split"]].isna().sum().sum() == 0
    assert df.var_id.is_unique, "var_id 중복"
    assert set(df.type3) == set(TYPE3), f"type3 값: {set(df.type3)}"
    assert set(df[SS_FIELD]) <= set(SS_CLASSES), f"ss 값: {set(df[SS_FIELD])}"
    assert df[POS].nunique() == N_POSITIONS, f"position {df[POS].nunique()} != {N_POSITIONS}"
    counts = df.groupby(["split", "type3"]).size().to_dict()
    assert counts == EXPECTED_COUNTS, f"명세 §2.1 개수와 다름: {counts}"
    # 명세 §1 포인트 1: 같은 position 이면 Block A 9개 값이 완전히 같다
    nun = df.groupby(POS)[CONT_FIELDS + [SS_FIELD]].nunique().to_numpy().max()
    assert nun == 1, "같은 position 에서 Block A 가 다르다"
    # shipped split 은 position 기준이어야 한다 (한 position 이 두 split 에 걸치면 안 됨)
    assert df.groupby(POS)["split"].nunique().max() == 1, "shipped split 이 position 을 가른다"

    return Data(
        cont=df[CONT_FIELDS].to_numpy(np.float64),
        ss=df[SS_FIELD].map({s: i for i, s in enumerate(SS_CLASSES)}).to_numpy(np.int64),
        typ=df["type3"].map({t: i for i, t in enumerate(TYPE3)}).to_numpy(np.int64),
        y=df[TARGET].to_numpy(np.float64),
        pos=df[POS].to_numpy(np.int64),
        split=df["split"].to_numpy(str),
        var_id=df["var_id"].to_numpy(str),
    )


def shuffle_structure(d: Data, seed: int) -> Data:
    """음성 대조군 (Hewitt & Liang 2019 의 control task 에 해당).

    행을 섞지 않는다. position p 의 Block A 9개 값을 **다른 position 하나의 값으로 통째로**
    바꾼다(전단사). 그래서
      · 같은 position → 같은 값 (유지)
      · 각 feature 의 position 수준 주변분포, feature 간 상관 (유지)
      · "이 변이가 구조상 어디인가" (파괴)
    """
    uniq = np.unique(d.pos)
    rng = np.random.default_rng(seed)
    src = dict(zip(uniq, rng.permutation(uniq)))        # p 의 값을 src[p] 에서 가져온다
    first_row = {p: np.flatnonzero(d.pos == p)[0] for p in uniq}
    take = np.array([first_row[src[p]] for p in d.pos])
    out = replace(d, cont=d.cont[take].copy(), ss=d.ss[take].copy(), shuffled=True)
    # 검사: 같은 position → 같은 값 유지, 값의 position 수준 다중집합 보존
    for p in uniq[:5]:
        rows = np.flatnonzero(out.pos == p)
        assert (out.cont[rows] == out.cont[rows[0]]).all()
    a = np.sort(np.array([d.cont[first_row[p]] for p in uniq]), axis=0)
    b = np.sort(np.array([out.cont[first_row[p]] for p in uniq]), axis=0)
    assert np.array_equal(a, b), "셔플이 position 수준 주변분포를 바꿨다"
    return out
