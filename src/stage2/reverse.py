"""Reverse-direction Stage 2 models: sequence tokens are modulated by structure (token-wise FiLM).

    h_i  : Stage 1 unified_reference_delta content token (= Stage 1 V for residue slots,
           before position/layer embeddings). The Stage 1 edit-metadata token (last K/V
           slot) is NOT a residue token and is excluded from FiLM and pooling in both models.
    v_j  = W_V s_j                          shared value projection, 32 -> d
    R0 : c_i = mean_j v_j                   (same context for every token)
    R1 : q_i = W_Q h_i, k_j = W_K s_j       (d_a = 32), a_ij = softmax_j(q_i.k_j / sqrt(d_a)),
         c_i = sum_j a_ij v_j               softmax over the 9 structure tokens
    [gamma_i, beta_i] = g([c_i ; e_type])   d+d -> 32 -> 2d, last layer zero-initialised
    h_mod_i = (1 + gamma_i) * h_i + beta_i
    z = masked mean_i h_mod_i               over valid residue slots
    delta = Head(z)                         d -> 32 -> 1, last layer zero-initialised
    y_final = y1 + delta

R0 and R1 share every module except W_Q / W_K (R1 only). No dummy parameters are added.
The forward signature matches Stage2Model so src.stage2.engine.train_stage2 is reused as is.

R2 (`r2_multihead_query`, ReverseMultiHeadFiLMModel) strengthens R1 along the two axes the
Stage 2 diagnosis (analysis/stage2_diag/REPORT.md) flagged for the forward model's attention:
capacity and sharpness, applied here to the reverse (sequence-queries-structure) direction.
  * n_heads independent (Q, K, V) triples (standard multi-head cross-attention), each over the
    SAME 9 structure tokens, concatenated and mixed by an output projection W_O -- more than a
    single d_a=32 projection can represent.
  * a single learnable temperature tau = softplus(log_tau) + 1e-4 divides every head's scaled
    dot product (same parameterisation as Stage 1/Stage 2 forward), so sharpness is learned
    rather than fixed at the standard 1/sqrt(d_head) scaling R1 uses.
    logits_h = (q_h . k_h) / sqrt(d_head) / tau
R0/R1 are left untouched; R2 is a separate class so existing results and tests stay valid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .schema import N_STRUCT_TOKENS, STRUCT_TOKEN_DIM, VARIANT_TYPES

REVERSE_MODES = ("r0_struct_mean", "r1_seq_query", "r2_multihead_query")


class ReverseFiLMModel(nn.Module):
    def __init__(self, mode: str, d: int = 128, struct_dim: int = STRUCT_TOKEN_DIM, d_a: int = 32,
                 film_hidden: int = 32, head_hidden: int = 32) -> None:
        super().__init__()
        if mode not in ("r0_struct_mean", "r1_seq_query"):
            raise ValueError(f"ReverseFiLMModel mode must be r0_struct_mean or r1_seq_query, got {mode!r}")
        self.mode = self.query_mode = mode
        self.d, self.d_a = d, d_a
        self.w_v = nn.Linear(struct_dim, d)                         # shared value projection
        self.type_emb = nn.Embedding(len(VARIANT_TYPES), d)
        if mode == "r1_seq_query":
            self.w_q = nn.Linear(d, d_a)                            # sequence token -> query
            self.w_k = nn.Linear(struct_dim, d_a)                   # structure token -> key
        self.film = nn.Sequential(nn.Linear(2 * d, film_hidden), nn.GELU(), nn.Linear(film_hidden, 2 * d))
        self.head = nn.Sequential(nn.Linear(d, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
        for last in (self.film[-1], self.head[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, S: Tensor, K: Tensor, V: Tensor, valid: Tensor, type_id: Tensor, y1: Tensor) -> dict:
        """S [B,9,32]; V/valid from Stage 1 with the metadata token last (K is unused)."""
        if S.shape[1] != N_STRUCT_TOKENS:
            raise ValueError(f"expected {N_STRUCT_TOKENS} structure tokens, got {S.shape[1]}")
        h = V[:, :-1]                                               # [B, N_res, d] residue content tokens
        m = valid[:, :-1].to(h.dtype)                               # [B, N_res]
        v = self.w_v(S)                                             # [B, 9, d]
        if self.mode == "r0_struct_mean":
            c = v.mean(1, keepdim=True).expand(-1, h.shape[1], -1)  # [B, N_res, d]
            attn = None
        else:
            qy = self.w_q(h)                                        # [B, N_res, d_a]
            ky = self.w_k(S)                                        # [B, 9, d_a]
            attn = torch.softmax(torch.einsum("bia,bja->bij", qy, ky) / math.sqrt(self.d_a), dim=-1)
            c = torch.einsum("bij,bjd->bid", attn, v)               # [B, N_res, d]
        e = self.type_emb(type_id).unsqueeze(1).expand(-1, h.shape[1], -1)
        gamma, beta = self.film(torch.cat([c, e], -1)).chunk(2, -1)
        h_mod = (1.0 + gamma) * h + beta
        z = (h_mod * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)
        delta = self.head(z).squeeze(-1)
        if attn is not None:
            ent_tok = -(attn * attn.clamp_min(1e-12).log()).sum(-1)    # [B, N_res], reference log(9)
            entropy = ((ent_tok * m).sum(1) / m.sum(1).clamp_min(1.0)).unsqueeze(1)
        else:
            entropy = torch.full((h.shape[0], 1), float("nan"), device=h.device)
        # gamma/beta are reported over valid residue tokens only
        sel = m.bool()
        return {"pred": y1 + delta, "delta": delta, "z": z, "c": c, "attn": attn,
                "gamma": gamma[sel], "beta": beta[sel], "entropy": entropy}

    def tau(self) -> Tensor:          # interface parity with Stage2Model (no temperature here)
        return torch.tensor(float("nan"))

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ReverseMultiHeadFiLMModel(nn.Module):
    """r2_multihead_query: multi-head version of R1, with a learnable temperature.

    q_h,i = W_Q^h h_i, k_h,j = W_K^h s_j, v_h,j = W_V^h s_j        (d_head per head)
    a_h,ij = softmax_j( q_h,i . k_h,j / sqrt(d_head) / tau )       tau = softplus(log_tau) + 1e-4
    c_h,i = sum_j a_h,ij v_h,j
    c_i = W_O( concat_h c_h,i )                                   n_heads*d_head -> d
    [gamma_i, beta_i] = g([c_i ; e_type]);  h_mod_i = (1+gamma_i) h_i + beta_i   (same as R0/R1)
    """

    mode = query_mode = "r2_multihead_query"

    def __init__(self, d: int = 128, struct_dim: int = STRUCT_TOKEN_DIM, n_heads: int = 4, d_head: int = 16,
                 tau_init: float | None = None, film_hidden: int = 32, head_hidden: int = 32) -> None:
        super().__init__()
        self.d, self.n_heads, self.d_head = d, n_heads, d_head
        dh = n_heads * d_head
        self.type_emb = nn.Embedding(len(VARIANT_TYPES), d)
        self.w_q = nn.Linear(d, dh)
        self.w_k = nn.Linear(struct_dim, dh)
        self.w_v = nn.Linear(struct_dim, dh)
        self.w_o = nn.Linear(dh, d)
        init_log_tau = 0.0 if tau_init is None else math.log(math.expm1(tau_init - 1e-4))
        self.log_tau = nn.Parameter(torch.tensor(init_log_tau))
        self.film = nn.Sequential(nn.Linear(2 * d, film_hidden), nn.GELU(), nn.Linear(film_hidden, 2 * d))
        self.head = nn.Sequential(nn.Linear(d, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
        for last in (self.film[-1], self.head[-1]):               # zero-init only the true last layers,
            nn.init.zeros_(last.weight)                            # exactly as R0/R1 -- w_o keeps normal
            nn.init.zeros_(last.bias)                              # init so c carries a real signal from step 1

    def tau(self) -> Tensor:
        return F.softplus(self.log_tau) + 1e-4

    def forward(self, S: Tensor, K: Tensor, V: Tensor, valid: Tensor, type_id: Tensor, y1: Tensor) -> dict:
        if S.shape[1] != N_STRUCT_TOKENS:
            raise ValueError(f"expected {N_STRUCT_TOKENS} structure tokens, got {S.shape[1]}")
        h = V[:, :-1]                                                # [B, N_res, d]
        m = valid[:, :-1].to(h.dtype)                                # [B, N_res]
        B, N_res = h.shape[0], h.shape[1]
        H, D = self.n_heads, self.d_head
        q = self.w_q(h).view(B, N_res, H, D)
        k = self.w_k(S).view(B, N_STRUCT_TOKENS, H, D)
        v = self.w_v(S).view(B, N_STRUCT_TOKENS, H, D)
        logits = torch.einsum("bihd,bjhd->bihj", q, k) / math.sqrt(D) / self.tau()
        attn = torch.softmax(logits, dim=-1)                         # [B, N_res, H, 9], softmax over structure tokens
        c_h = torch.einsum("bihj,bjhd->bihd", attn, v).reshape(B, N_res, H * D)
        c = self.w_o(c_h)                                            # [B, N_res, d]
        e = self.type_emb(type_id).unsqueeze(1).expand(-1, N_res, -1)
        gamma, beta = self.film(torch.cat([c, e], -1)).chunk(2, -1)
        h_mod = (1.0 + gamma) * h + beta
        z = (h_mod * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)
        delta = self.head(z).squeeze(-1)
        ent = -(attn * attn.clamp_min(1e-12).log()).sum(-1)          # [B, N_res, H], reference log(9)
        entropy = (ent * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)   # [B, H]
        sel = m.bool()
        return {"pred": y1 + delta, "delta": delta, "z": z, "c": c, "attn": attn,
                "gamma": gamma[sel], "beta": beta[sel], "entropy": entropy}

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_reverse_model(mode: str, tau_init: float | None = None) -> nn.Module:
    if mode == "r2_multihead_query":
        return ReverseMultiHeadFiLMModel(tau_init=tau_init)
    return ReverseFiLMModel(mode)
