"""
launch.py — 실험 그룹의 작업(학습기 × repeat × fold)을 병렬로 돌린다. 끝난 작업은 건너뛴다.

그룹
  E1   천장·바닥    : GBDT (원래 9개 값), TYPE (변이유형만)
  E3   인코딩 비교  : L, Q, Qk, T, PLR  (+ 셔플 구조 대조군 Qk-shuf)
                     probe = 변이유형 query → 토큰 9개 cross-attention → MLP head

실행 예:
  python launch.py --groups E1 E3 --workers 8 --device cpu --n_trials 50
  python launch.py --groups E3 --smoke --workers 4 --device cpu     # 배관만 확인
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as C
from cv import job_name

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = {"repeats": 1, "max_epochs": 4, "patience": 2, "refit_seeds": 2}


def jobs_for(group: str, repeats: int, k: int) -> list:
    if group == "E1":
        specs = [("GBDT", False), ("TYPE", False)]
    elif group == "E3":
        specs = [(v, False) for v in ("L", "Q", "Qk", "T", "PLR")] + [("Qk", True)]
    else:
        raise ValueError(group)
    return [(s, r, f) for s in specs for r in range(repeats) for f in range(k)]


def running_jobs() -> set:
    """이 기계에서 지금 돌고 있는 cv.py 작업들 (다른 launcher 가 띄운 것 포함).

    launcher 를 중간에 바꿔 띄울 때, 아직 돌고 있는 작업을 또 시작하지 않기 위해 쓴다
    (같은 study.db 에 두 프로세스가 trial 을 쓰면 trial 수가 예산을 넘는다).
    """
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    found = set()
    for line in out.splitlines():
        tok = line.split()
        # 실제로 python 이 cv.py 를 실행 중인 줄만 (셸 명령 문자열에 'cv.py' 글자가 든 경우 제외)
        if len(tok) < 2 or not tok[1].endswith("cv.py") or "python" not in tok[0].lower():
            continue
        get = lambda k, d=None: tok[tok.index(k) + 1] if k in tok and tok.index(k) + 1 < len(tok) else d
        try:
            found.add((get("--learner"), "--shuffle" in tok, int(get("--repeat")), int(get("--fold"))))
        except (TypeError, ValueError):
            continue
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only_repeat", type=int, default=None,
                    help="이 repeat 의 작업만 (두 기계가 겹치지 않게 나눌 때)")
    ap.add_argument("--max_procs", type=int, default=0,
                    help="이 기계의 cv.py 프로세스 총수 상한 (이전 launcher 가 남긴 것 포함). 0=workers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_trials", type=int, default=C.PROTOCOL["n_trials"])
    ap.add_argument("--out", default=C.RESULTS_DIR)
    ap.add_argument("--smoke", action="store_true", help="아주 작은 예산으로 배관만 확인")
    a = ap.parse_args()

    proto = dict(C.PROTOCOL)
    n_trials = a.n_trials
    if a.smoke:
        proto.update(SMOKE)
        n_trials = min(n_trials, 3)
    jobs = [j for g in a.groups for j in jobs_for(g, proto["repeats"], proto["outer_k"])]
    if a.only_repeat is not None:
        jobs = [j for j in jobs if j[1] == a.only_repeat]
    os.makedirs(os.path.join(a.out, "_logs"), exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(os.path.join(
        a.out, job_name(*j[0]), f"r{j[1]}_k{j[2]}", "done.json"))]
    busy = running_jobs()
    skipped = [j for j in todo if (*j[0], j[1], j[2]) in busy]
    todo = [j for j in todo if (*j[0], j[1], j[2]) not in busy]
    max_procs = a.max_procs or a.workers
    print(f"작업 {len(jobs)}개 중 남은 것 {len(todo)}개 (이미 돌고 있어 건너뜀 {len(skipped)}개) · "
          f"동시 {a.workers}개 · 기계 전체 상한 {max_procs} · trial {n_trials} · device {a.device}"
          f"{' · SMOKE' if a.smoke else ''}", flush=True)
    import threading
    gate = threading.Lock()

    def run(j):
        (learner, shuf), r, f = j
        tag = f"{job_name(learner, shuf)}_r{r}_k{f}"
        cmd = [sys.executable, os.path.join(HERE, "cv.py"), "--learner", learner,
               "--repeat", str(r), "--fold", str(f),
               "--n_trials", str(n_trials), "--out", a.out, "--device", a.device,
               "--proto", json.dumps(proto)] + (["--shuffle"] if shuf else [])
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "TORCH_THREADS": "1"}
        log = open(os.path.join(a.out, "_logs", tag + ".log"), "w")
        # 잠금을 쥔 채로: 이전 launcher 가 남긴 것까지 합쳐 max_procs 미만이 될 때까지 기다렸다가
        # 띄우고, ps 에 보일 때까지 잠깐 기다린다 (여러 스레드가 동시에 통과하는 경쟁 방지)
        with gate:
            while len(running_jobs()) >= max_procs:
                time.sleep(30)
            if os.path.exists(os.path.join(a.out, job_name(learner, shuf),
                                           f"r{r}_k{f}", "done.json")):
                log.close()
                return tag, 0, 0.0                       # 기다리는 사이 다른 쪽이 끝냄
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            time.sleep(2)
        t0 = time.time()
        rc = proc.wait()
        log.close()
        return tag, rc, time.time() - t0

    t0, fails = time.time(), []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run, j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            tag, rc, sec = fu.result()
            if rc != 0:
                fails.append(tag)
            print(f"  [{i}/{len(todo)}] {'OK ' if rc == 0 else 'ERR'} {tag}  {sec:.0f}s  "
                  f"(누적 {(time.time() - t0) / 60:.1f}분)", flush=True)
    print(f"끝. 실패 {len(fails)}개" + (f": {fails} — _logs/ 확인" if fails else ""))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
