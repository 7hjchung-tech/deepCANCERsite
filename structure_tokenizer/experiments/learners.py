"""
learners.py — nested CV 에 끼워 넣는 학습기들. 모두 같은 인터페이스를 가진다.

  suggest(trial)                       → 하이퍼파라미터 dict (Optuna)
  inner(p, data, sp)                   → (안쪽 fold 예측을 바깥 train 전체에 모은 벡터, 기록)
  refit(p, attrs, data, sp, seeds)     → 바깥 test 예측 [S, n_test]

학습기
  NN   : 인코딩 후보 (L / Q / Qk / T / PLR) + 변이유형 query cross-attention probe + MLP head  ← 비교 대상
  GBDT : 원래 9개 값 + 튜닝된 gradient boosting                    ← 천장 추정 (Grinsztajn+ 2022)
  TYPE : 변이유형 평균만                                            ← 바닥
"""
from __future__ import annotations

import numpy as np
import optuna
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C
from encoders import prepare
from model import build_model
from train import fit, predict


def suggest_from(trial, space: dict) -> dict:
    p = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "log":
            p[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            p[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "intlog":
            p[name] = trial.suggest_int(name, spec[1], spec[2], log=True)
        elif kind == "cat":
            p[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(kind)
    return p


def _onehot(idx, k):
    return np.eye(k)[idx]


# ====================================================================== NN
class NNLearner:
    VARIANTS = ("L", "Q", "Qk", "T", "PLR")

    def __init__(self, variant: str, device, proto: dict):
        assert variant in self.VARIANTS
        self.variant, self.device, self.proto = variant, device, proto
        space = dict(C.SPACE_COMMON)
        space.update(C.SPACE_MLP_HEAD)
        space.update({"L": {}, "Q": C.SPACE_PLE, "Qk": C.SPACE_PLE, "T": C.SPACE_TREE,
                      "PLR": C.SPACE_PLR}[variant])
        self.space = space

    def sampler(self, seed):
        return optuna.samplers.TPESampler(seed=seed)

    def suggest(self, trial):
        return suggest_from(trial, self.space)

    def _run(self, p, data, train_rows, val_rows, fixed_epochs, seed, pred_rows):
        dev, P = self.device, self.proto
        prep = prepare(self.variant, p, data.cont, data.y, train_rows)
        torch.manual_seed(seed)
        model = build_model(len(train_rows), self.variant, p, prep,
                            P["fusion_dim"], P["head_hidden"]).to(dev)
        X = torch.as_tensor(prep.X, device=dev)
        yn = torch.as_tensor(prep.ynorm, device=dev)
        ss = torch.as_tensor(data.ss, device=dev)
        ty = torch.as_tensor(data.typ, device=dev)
        res = fit(model, X, ss, ty, yn, train_rows, val_rows, lr=p["lr"],
                  weight_decay=p["weight_decay"], batch_size=P["batch_size"],
                  max_epochs=P["max_epochs"], patience=P["patience"],
                  fixed_epochs=fixed_epochs, seed=seed, device=dev)
        preds = predict(model, X, ss, ty, pred_rows, dev)
        preds = [pr * prep.y_sd[m] + prep.y_mu[m] for m, pr in enumerate(preds)]   # 원래 척도로
        return preds, res, prep, model

    def inner(self, p, data, sp):
        tr = [a for a, _ in sp["inner"]]
        va = [b for _, b in sp["inner"]]
        # 모든 trial 에 같은 seed(0) — 설정 간 차이가 초기화 운이 아니라 설정에서 오도록
        preds, res, prep, model = self._run(p, data, tr, va, None, 0, va)
        oof = np.full(data.n, np.nan)
        for rows, pr in zip(va, preds):
            oof[rows] = pr
        attrs = {"best_epochs": res.best_epoch.tolist(), "epochs_run": res.epochs_run,
                 "val_mse_norm": res.best_val_mse.tolist(),
                 "n_params": model.n_params_per_model()}
        if prep.bins is not None:
            attrs["ple_bins_mean"] = prep.bins["n_bins"].mean(0).round(2).tolist()
        return oof, attrs

    def refit(self, p, attrs, data, sp, seeds):
        # 선택된 설정의 안쪽 best epoch 중앙값만큼, 바깥 train 전체로 seed S 개 동시 학습
        n_ep = int(np.median(attrs["best_epochs"])) + 1
        S = len(seeds)
        tr = [sp["outer_train"]] * S
        te = [sp["outer_test"]] * S
        preds, res, prep, model = self._run(p, data, tr, None, n_ep, int(seeds[0]), te)
        attrs = {"refit_epochs": n_ep, "n_params": model.n_params_per_model()}
        attrs["attn_by_type"] = self._attn_by_type(model, prep, data, sp["outer_test"], S)
        return np.stack(preds), attrs

    def _attn_by_type(self, model, prep, data, te, S):
        """바깥 test 행에서 변이유형별로 토큰 9개(FIELD_ORDER)에 준 attention 가중치 평균
        (head·seed 평균). 해석용 기록 — 판정에는 쓰지 않는다."""
        dev = self.device
        model.eval()
        with torch.no_grad():
            X = torch.as_tensor(prep.X[:, te], device=dev)
            ss = torch.as_tensor(np.tile(data.ss[te], (S, 1)), device=dev)
            ty = torch.as_tensor(np.tile(data.typ[te], (S, 1)), device=dev)
            _, w = model.cross_attend(X, ss, ty, return_weights=True)     # [S,n,H,9]
            w = w.mean(dim=2).cpu().numpy()
        return {C.TYPE3[g]: w[:, data.typ[te] == g].mean(axis=(0, 1)).round(4).tolist()
                for g in range(len(C.TYPE3)) if (data.typ[te] == g).any()}


# ====================================================================== GBDT
class GBDTLearner:
    """천장 추정용. 원래 9개 값(연속 8 + ss one-hot) + 변이유형 one-hot."""
    SPACE = {"learning_rate": ("log", 0.005, 0.5), "max_iter": ("intlog", 50, 1000),
             "max_leaf_nodes": ("intlog", 2, 64), "min_samples_leaf": ("intlog", 1, 128),
             "l2_regularization": ("log", 1e-6, 10.0), "max_features": ("float", 0.3, 1.0)}

    def __init__(self, *a, **k):
        pass

    def sampler(self, seed):
        return optuna.samplers.TPESampler(seed=seed)

    def suggest(self, trial):
        return suggest_from(trial, self.SPACE)

    @staticmethod
    def feats(data):
        return np.c_[data.cont, _onehot(data.ss, 3), _onehot(data.typ, 3)]

    def _fit_pred(self, p, X, y, tr, te, seed):
        m = HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **p)
        return m.fit(X[tr], y[tr]).predict(X[te])

    def inner(self, p, data, sp):
        X = self.feats(data)
        oof = np.full(data.n, np.nan)
        for tr, va in sp["inner"]:
            oof[va] = self._fit_pred(p, X, data.y, tr, va, 0)
        return oof, {}

    def refit(self, p, attrs, data, sp, seeds):
        X = self.feats(data)
        return np.stack([self._fit_pred(p, X, data.y, sp["outer_train"], sp["outer_test"], s)
                         for s in seeds]), {}


# ====================================================================== TYPE (바닥)
class TypeLearner:
    def __init__(self, *a, **k):
        pass

    def sampler(self, seed):
        return optuna.samplers.RandomSampler(seed=seed)

    def suggest(self, trial):
        return {}

    @staticmethod
    def _fit_pred(data, tr, te):
        means = np.array([data.y[tr][data.typ[tr] == g].mean() for g in range(3)])
        return means[data.typ[te]]

    def inner(self, p, data, sp):
        oof = np.full(data.n, np.nan)
        for tr, va in sp["inner"]:
            oof[va] = self._fit_pred(data, tr, va)
        return oof, {}

    def refit(self, p, attrs, data, sp, seeds):
        return np.stack([self._fit_pred(data, sp["outer_train"], sp["outer_test"])] * len(seeds)), {}


FIXED_TRIALS = {"TYPE": 1}      # 하이퍼파라미터 없음


def make_learner(name: str, device, proto):
    if name in NNLearner.VARIANTS:
        return NNLearner(name, device, proto)
    return {"GBDT": GBDTLearner, "TYPE": TypeLearner}[name]()
