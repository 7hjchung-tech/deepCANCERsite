"""
model.py — 실험용 StructureTokenizer + cross-attention 읽기 + probe head. **M 개 독립 모델을 한 번에.**

    연속 f:  e_f = Enc_f(x_f) ∈ R^{d_s}          Enc = PLE→Linear_f | PLR | Linear  (인코딩 후보 5가지)
    범주 ss: e_ss = Embedding[ss]
             t_f = LayerNorm(e_f + field_f)       f = 1..9   → 토큰 [B, 9, d_s]
    읽기   : query → 토큰 9개 multi-head cross-attention (head 4개) → [B, fusion_dim]
             probe(E1·E3)에서는 query = 변이유형별 학습 query 3개 (구조만 보는 실험이라 ESM 이 없음)
             E7·E8 에서는 query 를 Stage 1 쪽에서 넘겨받는다 (attend_with_query / attend_with_queries)
    probe  : 읽은 값 + 변이유형 임베딩 → MLP(또는 선형) head

배포용 토크나이저(../tokenizer.py)와 토큰 계산이 같다 (tests/test_tokenizer_equivalence.py).
여기서는 하이퍼파라미터 탐색을 빠르게 하려고 모든 가중치에 맨 앞 차원 M 을 두고 einsum 으로
M 개 모델을 한 번에 계산한다. 손실을 모델별 평균의 **합**으로 두고, AdamW 는 원소별 연산이며
gradient clipping 을 쓰지 않으므로 M 개 모델은 서로 완전히 독립으로 학습된다 (tests/test_model.py).

초기화: Linear 는 nn.Linear 기본(±1/√fan_in). field/ss/type embedding 은 FT-Transformer
관례(±1/√d). PLR 주파수는 공식 구현처럼 N(0,σ) 를 ±3σ 에서 절단.
(파라미터를 만드는 순서는 실험을 돌렸던 코드와 같다 → 같은 seed 면 같은 초기값)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_CONT, N_TOKENS, SS_CLASSES, TYPE3


def _uniform(shape, bound):
    return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))


class BLinear(nn.Module):
    """M 개의 독립 Linear.  x [M, ..., fin] → [M, ..., fout]"""

    def __init__(self, M: int, fin: int, fout: int, bias: bool = True):
        super().__init__()
        bound = 1.0 / math.sqrt(fin)
        self.weight = _uniform((M, fin, fout), bound)
        self.bias = _uniform((M, fout), bound) if bias else None

    def forward(self, x):
        y = torch.einsum("m...i,mio->m...o", x, self.weight)
        if self.bias is not None:
            y = y + self.bias.view(self.bias.shape[0], *([1] * (x.dim() - 2)), -1)
        return y


class StructModel(nn.Module):
    N_HEADS = 4                                   # cross-attention head 수 (고정, 튜닝 안 함)

    def __init__(self, M: int, encoder: str, d_s: int, head: str = "mlp",
                 bins: dict | None = None, extrapolate: bool = False,
                 n_frequencies: int | None = None, sigma: float | None = None,
                 fusion_dim: int = 128, head_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.M, self.encoder, self.d_s = M, encoder, d_s
        self.head_kind, self.dropout = head, float(dropout)
        self.extrapolate = bool(extrapolate)
        d, Fd = d_s, fusion_dim

        # ---------------- 연속 feature 값 인코더 (후보마다 다름)
        if encoder == "ple":
            assert bins is not None
            for k in ("lo", "width"):
                self.register_buffer(k, torch.as_tensor(bins[k], dtype=torch.float32))
            for k in ("valid", "first", "last"):
                self.register_buffer(k, torch.as_tensor(bins[k], dtype=torch.bool))
            T = self.lo.shape[-1]
            w = torch.empty(M, N_CONT, T, d)
            nb = torch.as_tensor(bins["n_bins"])                      # [M, 8] 실제 구간 수
            bound = (1.0 / nb.float().sqrt())[..., None, None]         # feature 별 fan_in
            self.value_w = nn.Parameter(w.uniform_(-1, 1) * bound)    # bias 없음
        elif encoder == "plr":
            k = int(n_frequencies)
            c = torch.empty(M, N_CONT, k)
            nn.init.trunc_normal_(c, 0.0, sigma, a=-3 * sigma, b=3 * sigma)
            self.freq = nn.Parameter(c)
            self.value_w = _uniform((M, N_CONT, 2 * k, d), 1 / math.sqrt(2 * k))
            self.value_bias = _uniform((M, N_CONT, d), 1 / math.sqrt(2 * k))
        elif encoder == "lin":
            self.value_w = _uniform((M, N_CONT, d), 1 / math.sqrt(d))  # FT-Transformer 관례
        else:
            raise ValueError(encoder)

        self.ss_emb = _uniform((M, len(SS_CLASSES), d), 1 / math.sqrt(d))
        self.field_emb = _uniform((M, N_TOKENS, d), 1 / math.sqrt(d))
        self.ln_weight = nn.Parameter(torch.ones(M, d))            # 9 토큰 공유 LayerNorm
        self.ln_bias = nn.Parameter(torch.zeros(M, d))

        # ---------------- cross-attention 읽기 (K·V 사영이 d_s → Fd 변환을 맡는다)
        assert Fd % self.N_HEADS == 0
        self.n_heads = self.N_HEADS
        self.k_proj = BLinear(M, d, Fd)
        self.v_proj = BLinear(M, d, Fd)
        self.o_proj = BLinear(M, Fd, Fd)
        self.type_query = _uniform((M, len(TYPE3), Fd), 1 / math.sqrt(Fd))

        # ---------------- probe head (E7·E8 처럼 query 를 밖에서 받을 때는 쓰지 않는다)
        self.type_emb = _uniform((M, len(TYPE3), Fd), 1 / math.sqrt(Fd))
        if head == "linear":
            self.out = BLinear(M, Fd, 1)
        elif head == "mlp":
            self.hidden = BLinear(M, Fd, head_hidden)
            self.out = BLinear(M, head_hidden, 1)
        else:
            raise ValueError(head)

        # 모든 파라미터가 맨 앞에 모델 차원 M 을 가져야 train.py 의 모델별 best-state 복원이 맞다
        for n, p in self.named_parameters():
            assert p.shape[0] == M, f"{n} 의 첫 차원이 M 이 아니다: {tuple(p.shape)}"

    # ------------------------------------------------------------ 인코더
    def ple(self, x):
        """x [M,B,8] → [M,B,8,T]. 원논문 식(1)/공식 구현과 같은 PLE.

        extrapolate=False: 모든 구간을 [0,1] 로 자름 (배포 토크나이저와 같음)
        extrapolate=True : 첫 구간은 위만(≤1), 마지막 구간은 아래만(≥0) 자름 → 양 끝 선형 외삽
                           (원논문 p.4 "e1 ≤ 0, eT ≥ 1", rtdl_num_embeddings 와 동일)
        """
        e = (x[..., None] - self.lo[:, None]) / self.width[:, None]
        inf = torch.tensor(float("inf"), device=x.device)
        zero, one = torch.zeros((), device=x.device), torch.ones((), device=x.device)
        if self.extrapolate:
            lower = torch.where(self.first, -inf, zero)
            upper = torch.where(self.last, inf, one)
        else:
            lower = torch.zeros_like(self.lo)
            upper = torch.ones_like(self.lo)
        e = torch.maximum(torch.minimum(e, upper[:, None]), lower[:, None])
        return e * self.valid[:, None]                       # 패딩 칸은 항상 0

    def values(self, x):
        """연속 8개 → [M,B,8,d]"""
        if self.encoder == "ple":
            return torch.einsum("mbft,mftd->mbfd", self.ple(x), self.value_w)
        if self.encoder == "plr":
            v = 2 * math.pi * self.freq[:, None] * x[..., None]            # [M,B,8,k]
            h = torch.cat([torch.cos(v), torch.sin(v)], dim=-1)
            return F.relu(torch.einsum("mbfk,mfkd->mbfd", h, self.value_w)
                          + self.value_bias[:, None])
        return x[..., None] * self.value_w[:, None]                         # lin

    # ------------------------------------------------------------ 토큰
    def tokens(self, x, ss):
        """x [M,B,8], ss [M,B] → [M,B,9,d]  (토큰 순서 = config.FIELD_ORDER)"""
        M = self.M
        ar = torch.arange(M, device=x.device)[:, None]
        val = self.values(x)
        ss_tok = self.ss_emb[ar, ss]                                        # [M,B,d]
        t = torch.cat([val[:, :, :1], ss_tok[:, :, None], val[:, :, 1:]], dim=2)
        t = t + self.field_emb[:, None]
        t = F.layer_norm(t, (self.d_s,))
        return t * self.ln_weight[:, None, None] + self.ln_bias[:, None, None]

    # ------------------------------------------------------------ cross-attention 읽기
    def attend_with_queries(self, x, ss, queries, return_weights=False):
        """query 여러 개 [M,B,T,Fd] 로 구조 토큰 9개를 읽는다 → [M,B,T,Fd] (가중치 [M,B,T,H,9])"""
        M, B, T = queries.shape[:3]
        t = self.tokens(x, ss)                                          # [M,B,9,d]
        H = self.n_heads
        K, V = self.k_proj(t), self.v_proj(t)                           # [M,B,9,Fd]
        Fd = K.shape[-1]
        dh = Fd // H
        q = queries.reshape(M, B, T, H, dh)
        Kh, Vh = K.view(M, B, N_TOKENS, H, dh), V.view(M, B, N_TOKENS, H, dh)
        s = torch.einsum("mbqhd,mbthd->mbqht", q, Kh) / math.sqrt(dh)
        w = torch.softmax(s, dim=-1)
        o = torch.einsum("mbqht,mbthd->mbqhd", w, Vh).reshape(M, B, T, Fd)
        o = self.o_proj(o)
        return (o, w) if return_weights else o

    def attend_with_query(self, x, ss, query, return_weights=False):
        """query 하나 [M,B,Fd] → [M,B,Fd] (가중치 [M,B,H,9])"""
        o, w = self.attend_with_queries(x, ss, query[:, :, None], True)
        return (o[:, :, 0], w[:, :, 0]) if return_weights else o[:, :, 0]

    def cross_attend(self, x, ss, typ, return_weights=False):
        """probe: 변이유형별 학습 query → 토큰 9개"""
        ar = torch.arange(x.shape[0], device=x.device)[:, None]
        return self.attend_with_query(x, ss, self.type_query[ar, typ], return_weights)

    def forward(self, x, ss, typ):
        ar = torch.arange(self.M, device=x.device)[:, None]
        q = self.cross_attend(x, ss, typ) + self.type_emb[ar, typ]     # 변이유형은 조건으로
        if self.head_kind == "mlp":
            q = F.dropout(F.relu(self.hidden(q)), self.dropout, self.training)
        return self.out(q).squeeze(-1)                                      # [M,B]

    # ------------------------------------------------------------ 기타
    def param_groups(self, weight_decay: float):
        """weight decay 는 Linear 가중치에만 (FT-Transformer/rtdl 코드 관례)."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            is_linear_w = n == "value_w" or n.endswith(".weight")   # ln_weight 는 제외됨
            (decay if is_linear_w else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    def n_params_per_model(self) -> int:
        return sum(p.numel() for p in self.parameters()) // self.M


def build_model(M: int, variant: str, p: dict, prep, fusion_dim: int, head_hidden: int,
                head: str = "mlp") -> StructModel:
    from encoders import ENCODER_OF
    return StructModel(M, ENCODER_OF[variant], int(p["d_s"]), head, bins=prep.bins,
                       extrapolate=bool(p.get("extrapolate", False)),
                       n_frequencies=p.get("n_frequencies"), sigma=p.get("sigma"),
                       fusion_dim=fusion_dim, head_hidden=head_hidden,
                       dropout=float(p.get("dropout", 0.0)))
