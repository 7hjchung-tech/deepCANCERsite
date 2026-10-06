"""
tokenizer.py — RAD51C StructureTokenizer (Qk). 변이 위치의 구조 값 9개 → 토큰 9개.

입력 (변이마다, 변이 시작 위치 = 앵커 잔기의 값. 같은 위치의 변이는 값이 같다)
  연속 8개  A_plddt, A_rsasa, A_dist_walker_a, A_dist_walker_b, A_dist_atp_contact,
            A_dist_ssdna_binding, A_dist_bcdx2_interface, A_dist_cx3_interface
  범주 1개  A_ss_class ∈ {helix, sheet, loop}

토큰 (FT-Transformer 방식, Gorishniy+ 2021)
  token_f = LayerNorm( value_f + field_emb_f )            f = 9개 항목, 출력 [B, 9, d_s]
  value_f  (연속) = PLE(x_f) @ W_f      W_f: [구간 수, d_s], bias 없음
           (범주) = ss 임베딩
  토큰 순서 = FIELD_ORDER (pLDDT, ss, rSASA, 거리 6개)

Qk 인코딩 = PLE (Piecewise Linear Encoding, Gorishniy+ 2022)
  · pLDDT 의 구간 경계는 AlphaFold 공식 신뢰구간 [0, 50, 70, 90, 100] 로 고정
  · 나머지 7개는 train 행의 분위수 경계 (기본 4구간, 중복 경계는 합침)
  · PLE_t(x) = clip((x − lo_t) / width_t, 0, 1)  — 모든 구간을 0~1 로 자름 (양 끝 외삽 없음)
  · 경계는 반드시 train 행만으로 정한다 (fit_qk_bins 에 train 행만 넘길 것)

이 모듈은 토큰 9개를 만드는 데까지만 한다. 2026-09-30 설계대로 토큰을 하나로 합치지(pooling) 않고,
Stage 2 가 변이 쪽 query 로 cross-attention 해서 읽는다. 변이유형은 Stage 2 의 조건으로 넣는다.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

FIELD_ORDER = [
    "A_plddt",
    "A_ss_class",
    "A_rsasa",
    "A_dist_walker_a",
    "A_dist_walker_b",
    "A_dist_atp_contact",
    "A_dist_ssdna_binding",
    "A_dist_bcdx2_interface",
    "A_dist_cx3_interface",
]
SS_FIELD = "A_ss_class"
SS_CLASSES = ["helix", "sheet", "loop"]
CONT_FIELDS = [f for f in FIELD_ORDER if f != SS_FIELD]          # 연속 8개, 순서 유지
PLDDT_DOMAIN_BINS = np.array([0.0, 50.0, 70.0, 90.0, 100.0])     # AlphaFold pLDDT 신뢰구간
DEFAULT_N_BINS = 4        # 잠정값 (최종 하이퍼파라미터 확정 전)
DEFAULT_D_S = 32          # 잠정값

N_TOKENS, N_CONT = len(FIELD_ORDER), len(CONT_FIELDS)
assert N_TOKENS == 9 and N_CONT == 8
assert FIELD_ORDER.index(SS_FIELD) == 1 and CONT_FIELDS[0] == "A_plddt"


# ---------------------------------------------------------------- 입력 꺼내기
def features_from_frame(df) -> tuple[np.ndarray, np.ndarray]:
    """DataFrame(build_dataset.py 의 v2_dataset.csv 형식) → (cont [N,8] float32, ss [N] int64)."""
    missing = [c for c in FIELD_ORDER if c not in df.columns]
    if missing:
        raise KeyError(f"구조 컬럼이 없음: {missing}")
    cont = df[CONT_FIELDS].to_numpy(np.float32)
    ss_map = {s: i for i, s in enumerate(SS_CLASSES)}
    bad = set(df[SS_FIELD]) - set(ss_map)
    if bad:
        raise ValueError(f"알 수 없는 ss 값: {bad}")
    ss = df[SS_FIELD].map(ss_map).to_numpy(np.int64)
    if not np.isfinite(cont).all():
        raise ValueError("구조 값에 NaN/inf 가 있음")
    return cont, ss


# ---------------------------------------------------------------- Qk 구간 경계
def quantile_bins(x_train: np.ndarray, n_bins: int) -> np.ndarray:
    """분위수 경계 (rtdl_num_embeddings.compute_bins 와 같은 규칙: quantile 후 중복 제거)."""
    x = np.asarray(x_train, float)
    if n_bins < 1:
        raise ValueError("n_bins >= 1")
    b = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if b.size < 2:
        raise ValueError("값이 하나뿐인 feature 에는 경계를 만들 수 없다")
    return b


def fit_qk_bins(cont_train: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> list[np.ndarray]:
    """Qk 경계 8개. **train 행만** 넘길 것. pLDDT 는 공식 경계 고정, 나머지는 분위수."""
    cont_train = np.asarray(cont_train, float)
    assert cont_train.ndim == 2 and cont_train.shape[1] == N_CONT
    return [PLDDT_DOMAIN_BINS.copy() if j == 0 else quantile_bins(cont_train[:, j], n_bins)
            for j in range(N_CONT)]


# ---------------------------------------------------------------- 토크나이저
class StructureTokenizer(nn.Module):
    """구조 값 9개 → 토큰 [B, 9, d_s].

    tok = StructureTokenizer(fit_qk_bins(cont[train_rows]), d_s=32)
    tokens = tok(torch.as_tensor(cont[rows]), torch.as_tensor(ss[rows]))     # [B, 9, 32]
    """

    def __init__(self, bins: Sequence[np.ndarray], d_s: int = DEFAULT_D_S):
        super().__init__()
        bins = [np.asarray(b, float) for b in bins]
        assert len(bins) == N_CONT, "경계가 8개(연속 feature 수)여야 한다"
        for b in bins:
            assert b.ndim == 1 and b.size >= 2 and np.all(np.diff(b) > 0), f"경계 이상: {b}"
        self.d_s = int(d_s)
        n = np.array([b.size - 1 for b in bins])                      # feature 별 구간 수
        T = int(n.max())
        lo = np.zeros((N_CONT, T)); width = np.ones((N_CONT, T)); valid = np.zeros((N_CONT, T), bool)
        for f, b in enumerate(bins):
            lo[f, :n[f]], width[f, :n[f]], valid[f, :n[f]] = b[:-1], np.diff(b), True
        # 경계는 학습하지 않는 버퍼 (state_dict 에 함께 저장된다)
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("width", torch.as_tensor(width, dtype=torch.float32))
        self.register_buffer("valid", torch.as_tensor(valid))
        self.register_buffer("n_bins", torch.as_tensor(n))

        d = self.d_s
        bound = torch.as_tensor(1.0 / np.sqrt(n), dtype=torch.float32)[:, None, None]
        self.value_w = nn.Parameter(torch.empty(N_CONT, T, d).uniform_(-1, 1) * bound)   # fan_in = 구간 수
        self.ss_emb = nn.Parameter(torch.empty(len(SS_CLASSES), d).uniform_(-1, 1) / math.sqrt(d))
        self.field_emb = nn.Parameter(torch.empty(N_TOKENS, d).uniform_(-1, 1) / math.sqrt(d))
        self.norm = nn.LayerNorm(d)                                    # 토큰 9개가 공유

    @property
    def bins(self) -> list[np.ndarray]:
        """저장·재구성용 원래 경계."""
        out = []
        for f in range(N_CONT):
            t = int(self.n_bins[f])
            lo, w = self.lo[f, :t].cpu().numpy(), self.width[f, :t].cpu().numpy()
            out.append(np.append(lo, lo[-1] + w[-1]).astype(float))
        return out

    def ple(self, cont: torch.Tensor) -> torch.Tensor:
        """[B, 8] → [B, 8, T]. 구간마다 0~1 로 자름, 없는 구간(패딩)은 0."""
        e = (cont[..., None] - self.lo) / self.width
        return e.clamp(0.0, 1.0) * self.valid

    def forward(self, cont: torch.Tensor, ss: torch.Tensor) -> torch.Tensor:
        """cont [B, 8] (원래 단위: pLDDT, 비율, Å), ss [B] (SS_CLASSES 인덱스) → [B, 9, d_s]"""
        cont = cont.to(self.lo.dtype)
        val = torch.einsum("bft,ftd->bfd", self.ple(cont), self.value_w)        # [B, 8, d]
        ss_tok = self.ss_emb[ss.long()]                                          # [B, d]
        tok = torch.cat([val[:, :1], ss_tok[:, None], val[:, 1:]], dim=1)        # FIELD_ORDER 순서
        return self.norm(tok + self.field_emb)

    @classmethod
    def from_train(cls, cont_train: np.ndarray, n_bins: int = DEFAULT_N_BINS,
                   d_s: int = DEFAULT_D_S) -> "StructureTokenizer":
        return cls(fit_qk_bins(cont_train, n_bins), d_s)

    def save(self, path: str) -> None:
        torch.save({"bins": [b.tolist() for b in self.bins], "d_s": self.d_s,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "StructureTokenizer":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        tok = cls([np.asarray(b) for b in ck["bins"]], ck["d_s"])
        tok.load_state_dict(ck["state_dict"])
        return tok
