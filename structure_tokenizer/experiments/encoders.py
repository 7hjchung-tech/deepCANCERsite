"""
encoders.py — 연속 feature 전처리 (numpy). **항상 그 모델의 train 행에서만 fit 한다.**

PLE 경계 규칙은 원저자 공식 구현 rtdl_num_embeddings.compute_bins 와 같게 맞췄다
(tests/test_encoders.py 가 수치로 대조한다):
  · 분위수 경계  : quantile(linspace(0,1,T+1)) 를 구한 뒤 **중복값 제거(unique)**
  · 트리 경계    : [min, max] + 단일 feature 결정트리의 분할 임계값들, unique
  · pLDDT 도메인 : [0, 50, 70, 90, 100] 고정 (명세 §4.3)
분위수는 행 단위로 계산한다(같은 position 이 여러 행이면 그만큼 가중). 원논문도 train
"객체(행)" 단위이고, 이전 구현과도 같다.

L·PLR 후보는 원논문처럼 scikit-learn QuantileTransformer(정규분포 출력)로 전처리한다
(Gorishniy+ 2022 부록 E "Data preprocessing").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeRegressor

from config import N_CONT, PLDDT_COL, PLDDT_DOMAIN_BINS


# ------------------------------------------------------------------ 경계
def quantile_bins(x_train: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x_train, float)
    if n_bins < 1:
        raise ValueError("n_bins >= 1")
    b = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if b.size < 2:
        raise ValueError("상수 feature 에는 경계를 만들 수 없다")
    return b


def tree_bins(x_train: np.ndarray, y_train: np.ndarray, n_bins: int,
              min_samples_leaf: int, min_impurity_decrease: float) -> np.ndarray:
    """T-PLE (Gorishniy+ 2022 §3.2.2). **타깃을 쓰므로 train 행만** 넘겨야 한다."""
    x = np.asarray(x_train, float)
    edges = [float(x.min()), float(x.max())]
    tree = DecisionTreeRegressor(max_leaf_nodes=n_bins, min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 random_state=0).fit(x.reshape(-1, 1), y_train).tree_
    for node in range(tree.node_count):
        if tree.children_left[node] != tree.children_right[node]:      # 분할 노드
            edges.append(float(tree.threshold[node]))
    b = np.unique(np.asarray(edges))
    if b.size < 2:
        raise ValueError("상수 feature")
    return b


def fit_bins(mode: str, cont_train: np.ndarray, y_train_norm: np.ndarray, p: dict) -> list:
    """8개 연속 feature 의 경계 목록.  mode: quantile | quantile_plddt | tree"""
    out = []
    for j in range(N_CONT):
        x = cont_train[:, j]
        if mode == "quantile_plddt" and j == PLDDT_COL:
            out.append(PLDDT_DOMAIN_BINS.copy())
        elif mode in ("quantile", "quantile_plddt"):
            out.append(quantile_bins(x, p["n_bins"]))
        elif mode == "tree":
            out.append(tree_bins(x, y_train_norm, p["n_bins"], p["min_samples_leaf"],
                                 p["min_impurity_decrease"]))
        else:
            raise ValueError(mode)
    return out


def pad_bins(bins_per_model: list) -> dict:
    """모델마다 feature 마다 구간 수가 다르므로 [M, 8, T_max] 로 패딩한다.

    padded 칸은 valid=False 이고 model.py 가 0 으로 만든다. first/last 는 각 feature 의
    실제 첫·마지막 구간 표시(외삽 처리용).
    """
    M = len(bins_per_model)
    T = max(len(b) - 1 for bl in bins_per_model for b in bl)
    lo = np.zeros((M, N_CONT, T)); width = np.ones((M, N_CONT, T))
    valid = np.zeros((M, N_CONT, T), bool)
    first = np.zeros((M, N_CONT, T), bool); last = np.zeros((M, N_CONT, T), bool)
    n_bins = np.zeros((M, N_CONT), int)
    for m, bl in enumerate(bins_per_model):
        assert len(bl) == N_CONT
        for f, b in enumerate(bl):
            b = np.asarray(b, float)
            assert b.ndim == 1 and b.size >= 2 and np.all(np.diff(b) > 0), f"경계 이상: {b}"
            t = b.size - 1
            lo[m, f, :t] = b[:-1]; width[m, f, :t] = np.diff(b)
            valid[m, f, :t] = True; first[m, f, 0] = True; last[m, f, t - 1] = True
            n_bins[m, f] = t
    return {"lo": lo, "width": width, "valid": valid, "first": first, "last": last,
            "n_bins": n_bins}


def quantile_normal(cont_train: np.ndarray, cont_all: np.ndarray) -> np.ndarray:
    n = len(cont_train)
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=max(min(n // 30, 1000), 10),
                             subsample=10 ** 9, random_state=0)
    return qt.fit(cont_train).transform(cont_all)


# ------------------------------------------------------------------ 모델 묶음 입력
@dataclass
class Prepared:
    """M 개 모델(각자 train 행이 다름)에 들어갈 입력 묶음."""
    X: np.ndarray              # [M, N, 8] 모델별 연속 입력 (PLE 는 원값, L/PLR 은 변환값)
    ynorm: np.ndarray          # [M, N] 모델별 train 통계로 표준화한 타깃
    y_mu: np.ndarray           # [M]
    y_sd: np.ndarray           # [M]
    bins: dict | None = None   # pad_bins 결과 (PLE 계열만)
    raw_bins: list = field(default_factory=list)   # 기록용 원래 경계


ENCODER_OF = {"L": "lin", "Q": "ple", "Qk": "ple", "T": "ple", "PLR": "plr"}
BINMODE_OF = {"Q": "quantile", "Qk": "quantile_plddt", "T": "tree"}


def prepare(variant: str, p: dict, cont: np.ndarray, y: np.ndarray,
            train_rows: list) -> Prepared:
    M, N = len(train_rows), len(y)
    X = np.empty((M, N, N_CONT)); ynorm = np.empty((M, N))
    mu = np.empty(M); sd = np.empty(M); raw = []
    for m, rows in enumerate(train_rows):
        mu[m], sd[m] = y[rows].mean(), y[rows].std()       # train 통계로만 표준화
        ynorm[m] = (y - mu[m]) / sd[m]
        enc = ENCODER_OF[variant]
        if enc == "ple":
            raw.append(fit_bins(BINMODE_OF[variant], cont[rows], ynorm[m, rows], p))
            X[m] = cont
        else:
            X[m] = quantile_normal(cont[rows], cont)
    return Prepared(X=X.astype(np.float32), ynorm=ynorm.astype(np.float32), y_mu=mu, y_sd=sd,
                    bins=pad_bins(raw) if raw else None, raw_bins=raw)
