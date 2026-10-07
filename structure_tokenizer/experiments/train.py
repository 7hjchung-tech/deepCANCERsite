"""
train.py — M 개 모델 동시 학습. 모델 m 은 자기 train 행으로만 학습하고 자기 val 행으로만
early stopping 한다.

학습 규칙 (Gorishniy+ 2022 부록 E / TabM 과 같은 계열)
  · AdamW, lr 스케줄 없음, gradient clipping 없음(M 개 모델 독립성 유지를 위해서도 필요)
  · 손실 MSE(기본) 또는 Huber δ=1 (E7: Stage 1 과 같은 손실), 타깃은 모델별 train 통계로 표준화
  · early stopping: val MSE 가 patience 에폭 연속 개선되지 않으면 멈추고 **최고 시점 가중치로 복원**
    (선택 지표(subset Spearman)는 val 이 작을 때 요동이 커서, 멈춤 기준은 손실로 둔다 — 우리 결정)
  · fixed_epochs 를 주면 val 없이 정확히 그만큼 학습 (바깥 train 전체 재학습용)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FitResult:
    best_epoch: np.ndarray      # [M] 0부터 센 에폭 번호 (fixed 모드면 마지막 에폭)
    best_val_mse: np.ndarray    # [M] (fixed 모드면 nan)
    epochs_run: int


def _rows_kw(model, idx):
    """행 번호가 필요한 모델(ESM 특징을 내부 버퍼로 들고 있는 model_esm)에만 행 번호를 넘긴다."""
    return {"rows": idx} if getattr(model, "needs_rows", False) else {}


def _to(device, *arrs):
    return [torch.as_tensor(a, device=device) for a in arrs]


def _pad_rows(rows_list, rng=None, length=None):
    """모델별 행 목록 → [M, L]. 짧은 쪽은 자기 행에서 무작위로 채운다(학습) / 첫 행 반복(평가)."""
    L = length or max(len(r) for r in rows_list)
    out, mask = np.empty((len(rows_list), L), np.int64), np.zeros((len(rows_list), L), bool)
    for m, r in enumerate(rows_list):
        r = np.asarray(r)
        if rng is not None:
            p = rng.permutation(r)
            if len(p) < L:
                p = np.concatenate([p, rng.choice(r, L - len(p))])
            out[m] = p
            mask[m] = True
        else:
            out[m, :len(r)] = r
            out[m, len(r):] = r[0]
            mask[m, :len(r)] = True
    return out, mask


@torch.no_grad()
def predict(model, X, ss, typ, rows_list, device) -> list:
    """모델 m 을 rows_list[m] 에 적용. **표준화된 척도**로 반환(역변환은 호출자)."""
    model.eval()
    idx, mask = _pad_rows(rows_list)
    idx_t = torch.as_tensor(idx, device=device)
    ar = torch.arange(model.M, device=device)[:, None]
    out = model(X[ar, idx_t], ss[idx_t], typ[idx_t], **_rows_kw(model, idx_t)).float().cpu().numpy()
    return [out[m, mask[m]] for m in range(model.M)]


def fit(model, X, ss, typ, ynorm, train_rows: list, val_rows: list | None, *, lr: float,
        weight_decay: float, batch_size: int, max_epochs: int, patience: int,
        fixed_epochs: int | None, seed: int, device, loss: str = "mse") -> FitResult:
    """X [M,N,8], ss [N], typ [N], ynorm [M,N] — 전부 device 위 텐서."""
    M = model.M
    assert len(train_rows) == M and (val_rows is None or len(val_rows) == M)
    assert (fixed_epochs is None) != (val_rows is None), "early stopping 이면 val, fixed 면 val 없음"
    assert loss in ("mse", "huber")
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.param_groups(weight_decay), lr=lr)
    ar = torch.arange(M, device=device)[:, None]

    if val_rows is not None:
        vidx, vmask = _pad_rows(val_rows)
        vidx_t, vmask_t = _to(device, vidx, vmask)
        yv = ynorm[ar, vidx_t]
    best_mse = np.full(M, np.inf)
    best_ep = np.zeros(M, int)
    best_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    n_ep = fixed_epochs if fixed_epochs is not None else max_epochs

    ep = -1
    for ep in range(n_ep):
        model.train()
        I, _ = _pad_rows(train_rows, rng=rng)
        I_t = torch.as_tensor(I, device=device)
        for b in range(0, I.shape[1], batch_size):
            idx = I_t[:, b:b + batch_size]
            pred = model(X[ar, idx], ss[idx], typ[idx], **_rows_kw(model, idx))
            err = pred - ynorm[ar, idx]
            per = err ** 2 if loss == "mse" else torch.nn.functional.huber_loss(
                pred, ynorm[ar, idx], reduction="none", delta=1.0)
            obj = per.mean(dim=1).sum()                                   # 모델별 평균의 합
            opt.zero_grad(set_to_none=True)
            obj.backward()
            opt.step()

        if val_rows is None:
            continue
        model.eval()
        with torch.no_grad():
            pv = model(X[ar, vidx_t], ss[vidx_t], typ[vidx_t], **_rows_kw(model, vidx_t))
            mse = (((pv - yv) ** 2) * vmask_t).sum(1) / vmask_t.sum(1)
        mse = mse.cpu().numpy()
        imp = mse < best_mse
        if imp.any():
            imp_t = torch.as_tensor(imp, device=device)
            for n, p in model.named_parameters():
                best_state[n][imp_t] = p.detach()[imp_t]
            best_mse[imp] = mse[imp]
            best_ep[imp] = ep
        if np.all(ep - best_ep >= patience):
            break

    if val_rows is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(best_state[n])
        return FitResult(best_epoch=best_ep, best_val_mse=best_mse, epochs_run=ep + 1)
    return FitResult(best_epoch=np.full(M, n_ep - 1), best_val_mse=np.full(M, np.nan),
                     epochs_run=n_ep)
