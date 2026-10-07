"""
e7_stage2.py — E7: 학습된 Stage 1(호준 C·W10, 동결) 위에서 구조 토큰 9개가 보탬이 되는가.

설계 (Stage 2 1단계의 가장 작은 버전, model_e7.py 참고)
  Stage 1     : HF DeepCANCERsite/stage1-checkpoints 의 unified_reference_delta W10 best.pt.
                주 분석 = seed 44 (호준 제안: val 1등, 다른 seed 와 0.003 이내로 사실상 동률).
                나머지 seed 42/43/45/46 은 "결론이 Stage 1 seed 에 좌우되는가" 확인용.
  Stage 1 입력: 주 분석 = K/V 토큰 (호준 제안, README §9) / 보조 = 요약 벡터 z_seq
  팔          : b = 구조 없음 (노션 대조군 "구조 없이 Stage 1 만 추가 학습")
                c = 구조 토큰 9개 cross-attention / d = c 와 같은 모델 + 셔플 구조
  분할        : 배포(shipped) 분할 그대로. Stage 1 이 train 라벨로 학습됐으므로 위치 CV 를 쓰면 누출.
                train 으로 학습, val 로 선택(TPE 20회)과 early stopping, test 는 마지막에 한 번.
                5885 행 (민선 캐시에 없는 2개 제외 — test 의 Val351dup 1개 포함).
  학습        : 잔차, 마지막 층 0 초기화, Huber δ=1 (Stage 1 과 같은 손실), 고른 설정으로 seed 5개.
                탐색: lr LogU[1e-5, 3e-3], wd, dropout (+ c/d 는 d_s, n_bins). 세 팔 모두 20회.
  구조 인코딩 : 현재 명세 Qk, clip 고정 (E6/E6b 와 같음). d_s·n_bins 는 c/d 에서 함께 탐색.

미리 정한 판정 (결과 보기 전, 2026-10-05)
  주 가설: seed 44·tok 에서 c − d > 0 (구조 정보 효과) 그리고 c − b > 0 (실제 이득).
  CI 는 test 위치 cluster bootstrap 2000회. 다른 seed·z_seq 결과는 보조(탐색)로만 보고.

실행:  python e7_stage2.py --s1seed 44 --query tok     (작업 하나: 팔 b, c, d)
       python e7_stage2.py --analyze                     (→ results/REPORT_E7.md)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import optuna
import torch

import config as C
from data import load_data, shuffle_structure
from encoders import prepare
from learners import suggest_from
from metrics import all_metrics
from model_e7 import Stage2Model
from train import fit, predict

S1_DIR = os.path.join(C.RESULTS_DIR, "E7")
OUT = os.path.join(C.RESULTS_DIR, "E7", "stage2")
ARMS = {"b": ("b", False), "c": ("c", False), "d": ("c", True)}
S1_SEEDS, QUERIES = [44, 42, 43, 45, 46], ["tok", "z"]
PRIMARY = (44, "tok")
N_TRIALS, SEEDS = 20, 5
SPACE_E7, SPACE_STRUCT = C.SPACE_STAGE2, C.SPACE_STAGE2_STRUCT   # config.py 참고


def load_s1(s1seed: int, d):
    """Stage 1 출력(e7_stage1_reconstruct.py)을 우리 행 순서(load_data, 5887)에 맞춰 놓는다."""
    z = np.load(os.path.join(S1_DIR, f"stage1_unified_reference_delta_W10_seed{s1seed}.npz"))
    idx_of = {v: i for i, v in enumerate(d.var_id)}
    rows = np.array([idx_of[v] for v in z["var_id"]])
    assert (d.split[rows] == z["split"]).all(), "Stage 1 분할과 우리 분할이 다르다"
    assert np.allclose(d.y[rows], z["y"]), "라벨이 다르다"
    N, T = d.n, z["K"].shape[1]
    out = {"p1": np.zeros(N), "z": np.zeros((N, 128), np.float32),
           "K": np.zeros((N, T, 128), np.float16), "V": np.zeros((N, T, 128), np.float16),
           "tok_valid": np.zeros((N, T), bool), "avail": np.zeros(N, bool)}
    out["p1"][rows], out["z"][rows] = z["pred"], z["z_seq"]
    out["K"][rows], out["V"][rows], out["tok_valid"][rows] = z["K"], z["V"], z["tok_valid"]
    out["avail"][rows] = True
    return out


def splits_of(d, avail):
    sp = {k: np.flatnonzero((d.split == k) & avail) for k in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(d.pos[sp[a]]) & set(d.pos[sp[b]]), f"{a}/{b} 가 position 공유"
    return sp


def run(s1seed: int, query: str, device="cpu"):
    os.makedirs(OUT, exist_ok=True)
    d0 = load_data()
    s1 = load_s1(s1seed, d0)
    split = splits_of(d0, s1["avail"])
    P = C.PROTOCOL
    dev = device
    Zt, Kt, Vt, TVt = (torch.as_tensor(s1[k], device=dev) for k in ("z", "K", "V", "tok_valid"))
    for arm, (kind, shuf) in ARMS.items():
        tag = f"s{s1seed}_{query}_{arm}"
        if os.path.exists(os.path.join(OUT, tag + ".json")):
            continue
        t0 = time.time()
        d = shuffle_structure(d0, seed=P["shuffle_seed0"] + s1seed) if shuf else d0
        space = dict(SPACE_E7)
        if kind == "c":
            space.update(SPACE_STRUCT)
        ss, ty = torch.as_tensor(d.ss, device=dev), torch.as_tensor(d.typ, device=dev)

        def train_eval(p, M, seed, eval_sets):
            pp = {**p, "n_bins": p.get("n_bins", 4)}
            prep = prepare("Qk", pp, d.cont, d.y, [split["train"]] * M)
            assert np.allclose(prep.y_mu, prep.y_mu[0]) and np.allclose(prep.y_sd, prep.y_sd[0])
            p1n = torch.as_tensor((s1["p1"] - prep.y_mu[0]) / prep.y_sd[0], dtype=torch.float32, device=dev)
            S1 = {"p1n": p1n, "z": Zt, "K": Kt, "V": Vt, "tok_valid": TVt}
            sk = ({"encoder": "ple", "d_s": int(pp.get("d_s", 32)), "bins": prep.bins,
                   "extrapolate": False} if kind == "c" else None)
            torch.manual_seed(seed)
            model = Stage2Model(M, query, kind, S1, sk, fusion_dim=P["fusion_dim"],
                                head_hidden=P["head_hidden"], dropout=float(pp["dropout"])).to(dev)
            X, yn = torch.as_tensor(prep.X, device=dev), torch.as_tensor(prep.ynorm, device=dev)
            fr = fit(model, X, ss, ty, yn, [split["train"]] * M, [split["val"]] * M, lr=pp["lr"],
                     weight_decay=pp["weight_decay"], batch_size=P["batch_size"],
                     max_epochs=P["max_epochs"], patience=P["patience"], fixed_epochs=None,
                     seed=seed, device=dev, loss="huber")
            outs = {}
            for name, rows in eval_sets.items():
                pr = predict(model, X, ss, ty, [rows] * M, dev)
                outs[name] = np.stack([x * prep.y_sd[m] + prep.y_mu[m] for m, x in enumerate(pr)])
            return outs, model, X, fr

        def objective(trial):
            p = suggest_from(trial, space)
            outs, *_ = train_eval(p, 1, 0, {"val": split["val"]})
            return all_metrics(outs["val"][0], d.y[split["val"]], d.typ[split["val"]])["subset"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=200 + s1seed))
        st.optimize(objective, n_trials=N_TRIALS)
        best = st.best_params
        outs, model, X, fr = train_eval(best, SEEDS, 0, {"val": split["val"], "test": split["test"]})
        extra = {}
        if kind == "c":
            # 구조 토큰 9개 attention (헤드·Stage 2 pooling 가중 평균, seed 평균) — 쓸 수 있는 모든 행
            model.eval()
            allr = np.flatnonzero(s1["avail"])
            A = np.zeros((d.n, 9), np.float32)
            with torch.no_grad():
                for s in range(0, len(allr), 512):
                    r = torch.as_tensor(np.tile(allr[s:s + 512], (SEEDS, 1)), device=dev)
                    ar = torch.arange(SEEDS, device=dev)[:, None]
                    _, w9 = model.struct_read(X[ar, r], ss[r], r, return_weights=True)
                    A[allr[s:s + 512]] = w9.mean(0).cpu().numpy()
            extra["attn"] = A
        np.savez_compressed(os.path.join(OUT, tag + ".npz"), val_rows=split["val"],
                            test_rows=split["test"], preds_val=outs["val"], preds_test=outs["test"], **extra)
        te = split["test"]
        m = [all_metrics(p_, d.y[te], d.typ[te]) for p_ in outs["test"]]
        json.dump({"s1seed": s1seed, "query": query, "arm": arm, "best_params": best,
                   "best_val_subset": st.best_value, "trial_values": [t.value for t in st.trials],
                   "test_subset_per_seed": [x["subset"] for x in m],
                   "best_epoch": fr.best_epoch.tolist(), "n_params": model.n_params_per_model(),
                   "seconds": round(time.time() - t0, 1)},
                  open(os.path.join(OUT, tag + ".json"), "w"), indent=2, default=float)
        print(f"[{tag}] val {st.best_value:.4f}  test {np.mean([x['subset'] for x in m]):.4f}  "
              f"epochs {fr.best_epoch.tolist()}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1seed", type=int, choices=S1_SEEDS)
    ap.add_argument("--query", choices=QUERIES)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--out", default=OUT, help="스모크 테스트용 다른 폴더")
    a = ap.parse_args()
    torch.set_num_threads(1)
    OUT, N_TRIALS = a.out, a.n_trials
    if a.analyze:
        from e7_analyze import analyze
        analyze(OUT)
    else:
        run(a.s1seed, a.query)
