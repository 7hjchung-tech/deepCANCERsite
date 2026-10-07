"""
cv.py — nested CV 의 한 칸 = (학습기, repeat r, 바깥 fold k).

  1. 바깥 train 안에서 안쪽 K_in-fold 로 Optuna 탐색 (목적함수 = 안쪽 예측을 모은 subset Spearman)
  2. 최고 trial 의 설정으로 바깥 train 전체에 seed S 개 재학습
  3. 바깥 test 예측 저장  → merge 후 compare.py 가 paired 비교

바깥 test 는 1 에서 한 번도 쓰이지 않는다. 결과 폴더에 done.json 이 있으면 건너뛴다(재시작 가능).
탐색 기록은 SQLite 에 남아, 중간에 끊겨도 남은 trial 만 이어서 돈다.

실행 예:
  python cv.py --learner Qk --repeat 0 --fold 0 --device cuda
  python cv.py --learner Qk --shuffle --repeat 0 --fold 0     # 셔플 구조 대조군
  python cv.py --learner GBDT --repeat 0 --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import optuna
import torch

import config as C
from data import load_data, shuffle_structure
from learners import FIXED_TRIALS, make_learner
from metrics import all_metrics, subset_score
from splits import fingerprint, nested_split


def job_name(learner: str, shuffle: bool) -> str:
    return learner + ("-shuf" if shuffle else "")


def run_job(learner: str, shuffle: bool, repeat: int, fold: int,
            n_trials: int, out_root: str, device: str, proto: dict) -> str:
    name = job_name(learner, shuffle)
    jobdir = os.path.join(out_root, name, f"r{repeat}_k{fold}")
    if os.path.exists(os.path.join(jobdir, "done.json")):
        return jobdir
    os.makedirs(jobdir, exist_ok=True)
    t0 = time.time()

    data = load_data()
    if shuffle:
        data = shuffle_structure(data, seed=proto["shuffle_seed0"] + repeat)
    sp = nested_split(data.pos, repeat, fold, proto["outer_k"], proto["inner_k"])
    L = make_learner(learner, device, proto)
    n_trials = FIXED_TRIALS.get(learner, n_trials)
    otr = sp["outer_train"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="hpo", direction="maximize", load_if_exists=True,
        storage=f"sqlite:///{os.path.join(jobdir, 'study.db')}",
        sampler=L.sampler(proto["hpo_seed0"] + 1000 * repeat + fold))
    done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)
               for t in study.trials)

    def objective(trial):
        p = L.suggest(trial)
        oof, attrs = L.inner(p, data, sp)
        assert np.isfinite(oof[otr]).all(), "안쪽 예측이 바깥 train 을 다 덮지 못했다"
        assert np.isnan(oof[sp["outer_test"]]).all(), "안쪽 단계에서 바깥 test 를 예측했다"
        # 안쪽 fold 마다 계산해 평균 (fold 를 합치면 fold 간 예측 척도 차이가 가짜 상관을 만든다)
        per = [all_metrics(oof[va], data.y[va], data.typ[va]) for _, va in sp["inner"]]
        m = {k: float(np.nanmean([f[k] for f in per])) for k in per[0]}
        for k, v in {**attrs, **{f"inner_{k}": v for k, v in m.items()}}.items():
            trial.set_user_attr(k, v)
        return m["subset"]

    def progress(study_, trial):
        n = len(study_.trials)
        if n % 10 == 0 or n == n_trials:
            try:
                b = study_.best_value
            except ValueError:
                b = float("nan")
            print(f"[{name} r{repeat}k{fold}] trial {n}/{n_trials}  best inner subset={b:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if n_trials - done > 0:
        study.optimize(objective, n_trials=n_trials - done, callbacks=[progress],
                       catch=(RuntimeError, ValueError, FloatingPointError))

    best = study.best_trial
    seeds = list(range(proto["refit_seeds"]))
    preds, rattrs = L.refit(best.params, best.user_attrs, data, sp, seeds)
    te = sp["outer_test"]
    assert preds.shape == (len(seeds), len(te)) and np.isfinite(preds).all()

    np.savez_compressed(os.path.join(jobdir, "pred.npz"), test_rows=te, preds=preds,
                        seeds=np.array(seeds))
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state",
                                  "duration")).to_csv(os.path.join(jobdir, "trials.csv"),
                                                      index=False)
    test_m = [all_metrics(p, data.y[te], data.typ[te]) for p in preds]
    sel = {"job": name, "learner": learner, "shuffle": shuffle, "repeat": repeat, "fold": fold,
           "split_fingerprint": fingerprint(data.pos, repeat), "data_sha256": C.DATA_SHA256,
           "n_trials_complete": sum(t.state == optuna.trial.TrialState.COMPLETE
                                    for t in study.trials),
           "best_trial": best.number, "best_inner_subset": best.value,
           "best_params": best.params, "best_attrs": best.user_attrs, "refit": rattrs,
           # 참고용. 선택에는 쓰지 않았다. 판정은 compare.py 에서 전체 OOF 로.
           "outer_test_subset_per_seed": [m["subset"] for m in test_m],
           "n_outer_train": int(len(otr)), "n_outer_test": int(len(te)),
           "seconds": round(time.time() - t0, 1), "device": str(device),
           "torch": torch.__version__, "optuna": optuna.__version__,
           "python": platform.python_version(), "protocol": proto}
    with open(os.path.join(jobdir, "selected.json"), "w") as fh:
        json.dump(sel, fh, indent=2, default=float, ensure_ascii=False)
    with open(os.path.join(jobdir, "done.json"), "w") as fh:
        json.dump({"seconds": sel["seconds"]}, fh)
    print(f"[{name} r{repeat}k{fold}] 완료 {sel['seconds']}s  best inner={best.value:.4f}  "
          f"outer test(참고)={np.mean(sel['outer_test_subset_per_seed']):.4f}", flush=True)
    return jobdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=C.PROTOCOL["n_trials"])
    ap.add_argument("--out", default=C.RESULTS_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--proto", default="{}", help="PROTOCOL 덮어쓰기 JSON (스모크 테스트용)")
    a = ap.parse_args()
    proto = {**C.PROTOCOL, **json.loads(a.proto)}
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_THREADS", "1"))))
    run_job(a.learner, a.shuffle, a.repeat, a.fold, a.n_trials, a.out, a.device, proto)


if __name__ == "__main__":
    main()
