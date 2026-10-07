"""
model_e7.py — E7: 학습된 Stage 1(호준 C·W10, 동결) 위에 구조 토큰 9개를 읽는 Stage 2 (M 개 동시).

노션 Stage 2 계획(1단계)을 가장 작게 구현한 것:
  · Stage 1 은 동결. 출력은 미리 뽑아 둔 값(e7_stage1_reconstruct.py)을 버퍼로 조회한다.
  · 잔차 학습: ŷ = Stage 1 예측 + g(·). g 의 마지막 층을 0 으로 초기화 → 학습 시작점 = Stage 1 그대로.
  · 변이유형은 Stage 2 의 조건(e_type) — 2026-09-30 결정.

Stage 1 쪽 입력을 읽는 방식 (query)
  "tok" (주 분석, 호준 제안: README §9 의 K/V/attention_valid 를 Stage 2 입력으로)
      Stage 2 자체 pooling:  a_t = softmax_t(W_k K_t · q2),  u = Σ a_t W_v V_t
      구조 읽기(arm c):      토큰마다 query W_q K_t 로 구조 9개를 cross-attention → o_t,  s = Σ a_t o_t
  "z"   (보조 분석) Stage 1 이 pooling 해 둔 요약 벡터 z_seq 하나
      구조 읽기(arm c):      query W_q z 로 구조 9개를 cross-attention → s

  r = W z + e_type (+ u) (+ s) → ReLU MLP → 잔차 (마지막 Linear 0 초기화)

팔: b = 구조 없음 / c = 구조 / d = c 와 같은 모델에 셔플 구조(데이터 쪽에서 섞음)
Stage 1 의 z, K, V 는 척도가 정해져 있지 않아 파라미터 없는 LayerNorm 으로 맞춘 뒤 사영한다(우리 결정).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TYPE3
from model import BLinear, StructModel, _uniform


def _ln(x):
    return F.layer_norm(x, (x.shape[-1],))


class Stage2Model(nn.Module):
    needs_rows = True                      # train.py 가 행 번호를 넘겨준다 (Stage 1 출력 조회용)

    def __init__(self, M: int, query: str, arm: str, S1: dict, struct_kwargs: dict | None,
                 fusion_dim: int = 128, head_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        assert query in ("z", "tok") and arm in ("b", "c")
        self.M, self.query, self.arm, self.dropout = M, query, arm, float(dropout)
        Fd = fusion_dim
        # Stage 1 출력 = 학습 대상이 아닌 조회용 버퍼 (모든 모델 공유)
        self.register_buffer("p1n", S1["p1n"], persistent=False)     # [N] 표준화된 Stage 1 예측
        self.register_buffer("z", S1["z"], persistent=False)         # [N,128]
        D1 = S1["z"].shape[-1]
        self.z_proj = BLinear(M, D1, Fd)
        self.type_emb = _uniform((M, len(TYPE3), Fd), 1 / math.sqrt(Fd))
        if query == "tok":
            self.register_buffer("K", S1["K"], persistent=False)     # [N,T,128] fp16
            self.register_buffer("V", S1["V"], persistent=False)
            self.register_buffer("tv", S1["tok_valid"], persistent=False)   # [N,T] bool
            self.pool_q = _uniform((M, Fd), 1 / math.sqrt(Fd))
            self.k2 = BLinear(M, D1, Fd)
            self.v2 = BLinear(M, D1, Fd)
        if arm == "c":
            self.struct = StructModel(M, head="linear", fusion_dim=Fd, **struct_kwargs)
            self.q_proj = BLinear(M, D1, Fd)
        self.hidden = BLinear(M, Fd, head_hidden)
        self.out = BLinear(M, head_hidden, 1)
        with torch.no_grad():                                         # 잔차 경로 0 초기화
            self.out.weight.zero_()
            self.out.bias.zero_()
        for n, p in self.named_parameters():
            assert p.shape[0] == M, f"{n} 의 첫 차원이 M 이 아니다"

    # ------------------------------------------------------------ Stage 1 토큰 pooling
    def pool(self, rows):
        """rows [M,B] → Stage 2 pooling 가중치 a [M,B,T] 와 정규화된 K, V"""
        K, V = _ln(self.K[rows].float()), _ln(self.V[rows].float())   # [M,B,T,128]
        s = torch.einsum("mbtd,md->mbt", self.k2(K), self.pool_q) / math.sqrt(self.pool_q.shape[-1])
        s = s.masked_fill(~self.tv[rows], float("-inf"))
        return torch.softmax(s, dim=-1), K, V

    def struct_read(self, x, ss, rows, return_weights=False):
        """구조 읽기 → s [M,B,Fd] (return_weights 면 토큰 9개 가중치 [M,B,9], 헤드 평균)"""
        if self.query == "z":
            o, w = self.struct.attend_with_query(x, ss, self.q_proj(_ln(self.z[rows])), True)
            w9 = w.mean(dim=2)                                           # [M,B,H,9] → [M,B,9]
        else:
            a, K, _ = self.pool(rows)
            ot, w = self.struct.attend_with_queries(x, ss, self.q_proj(K), True)   # [M,B,T,Fd]
            o = torch.einsum("mbt,mbtd->mbd", a, ot)
            w9 = torch.einsum("mbt,mbth->mbh", a, w.mean(dim=3))       # Stage 2 pooling 가중 평균
        return (o, w9) if return_weights else o

    def forward(self, x, ss, typ, rows):
        ar = torch.arange(self.M, device=x.device)[:, None]
        r = self.z_proj(_ln(self.z[rows])) + self.type_emb[ar, typ]
        if self.query == "tok":
            a, _, V = self.pool(rows)
            r = r + torch.einsum("mbt,mbtd->mbd", a, self.v2(V))
        if self.arm == "c":
            r = r + self.struct_read(x, ss, rows)
        h = F.dropout(F.relu(self.hidden(r)), self.dropout, self.training)
        return self.p1n[rows] + self.out(h).squeeze(-1)

    # ------------------------------------------------------------ 기타
    def param_groups(self, weight_decay: float):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            (decay if (n.endswith(".weight") or n.endswith("value_w")) else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    def n_params_per_model(self) -> int:
        unused = {"struct.type_query", "struct.type_emb", "struct.out.weight", "struct.out.bias"}
        return sum(p.numel() for n, p in self.named_parameters() if n not in unused) // self.M
