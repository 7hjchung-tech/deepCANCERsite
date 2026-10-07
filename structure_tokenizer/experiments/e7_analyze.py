"""
e7_analyze.py — E7 결과 정리 → results/REPORT_E7.md

  1. 주 분석 (Stage 1 seed 44, K/V 토큰 query): a0(Stage 1 그대로)/b/c/d 의 test 성능과 짝지은 차이
  2. 보조: Stage 1 seed 5개 × query(tok, z) 각각의 차이, 그리고 seed 5개를 합친 추정
     (합친 추정 = stats.bootstrap 에서 Stage 1 seed 를 repeat 축으로 넣어 "seed 평균 성능"의 차이)
  3. attention (c, 해석용): 변이유형별 평균, 같은 position 안 손상 − 무해 missense
  4. 선택 기록: val 최고점, test, 파라미터 수, 멈춘 에폭
CI 는 모두 test 위치 cluster bootstrap 2000회 (같은 재표집으로 모든 팔을 계산 → paired).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config as C
from data import load_data
from stats import bootstrap

TOK = None


def _tok_names():
    return [f.replace("A_", "").replace("dist_", "d_") for f in C.FIELD_ORDER]


def _load_job(out, s1seed, query):
    res = {}
    for arm in ("b", "c", "d"):
        f = os.path.join(out, f"s{s1seed}_{query}_{arm}")
        if os.path.exists(f + ".npz") and os.path.exists(f + ".json"):
            res[arm] = (np.load(f + ".npz"), json.load(open(f + ".json")))
    return res if len(res) == 3 else None


def _fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def _pfmt(p, m):
    return f"{p[m]['diff']:+.3f} [{p[m]['lo']:+.3f}, {p[m]['hi']:+.3f}]"


def analyze(out, report_path=None):
    from e7_stage2 import S1_SEEDS, QUERIES, PRIMARY, SEEDS, load_s1, splits_of
    d = load_data()
    TOKN = _tok_names()
    fc = pd.read_csv(C.DATA_PATH, usecols=["functional_classification"])["functional_classification"].to_numpy()
    s1 = {s: load_s1(s, d) for s in S1_SEEDS}
    split = splits_of(d, s1[S1_SEEDS[0]]["avail"])
    te = split["test"]
    y, typ, pos = d.y[te], d.typ[te], d.pos[te]
    jobs = {(s, q): _load_job(out, s, q) for s in S1_SEEDS for q in QUERIES}
    done = {k: v for k, v in jobs.items() if v is not None}
    L = ["# E7 — 학습된 Stage 1(호준 C·W10, 동결) 위에서 구조 토큰이 보탬이 되는가", "",
         f"- 완료된 작업: {len(done)}/{len(jobs)}. test = 배포 분할 test {len(te)}개 "
         "(민선 캐시에 없는 Val351dup 제외), position {} 곳.".format(len(np.unique(pos))),
         "- 팔: **a0** Stage 1 그대로 / **b** Stage 2, 구조 없음 / **c** Stage 2 + 구조 토큰 9개 / "
         "**d** c 와 같은 모델 + 셔플 구조",
         "- 지표는 seed 별로 계산해 평균. CI 는 test position cluster bootstrap 2000회.", ""]

    # ---- Stage 1 과적합 (Stage 2 가 배울 잔차의 크기)
    L += ["## 0. 동결 Stage 1 의 train / val / test 맞춤 정도", "",
          "| Stage 1 seed | train subset | val subset | test subset | 잔차 SD train / val / test |",
          "|---|---|---|---|---|"]
    from metrics import all_metrics
    for s in S1_SEEDS:
        r = []
        for k in ("train", "val", "test"):
            rows = split[k]
            r.append((all_metrics(s1[s]["p1"][rows], d.y[rows], d.typ[rows])["subset"],
                      np.std(d.y[rows] - s1[s]["p1"][rows])))
        L.append(f"| {s} | {r[0][0]:.3f} | {r[1][0]:.3f} | {r[2][0]:.3f} | "
                 f"{r[0][1]:.2f} / {r[1][1]:.2f} / {r[2][1]:.2f} |")
    L += ["", "train 잔차가 val/test 보다 작다 = Stage 1 이 train 을 일부 외웠다. Stage 2 는 이 작아진 "
          "train 잔차로 학습하므로, 보정 크기를 과소 추정하거나 금방 val 이 나빠질 수 있다 "
          "(stacking 의 out-of-fold 문제, Wolpert 1992). 실제 Stage 2 계획도 같은 조건이다.", ""]

    def oof_block(keys):
        """keys: [(s1seed, query)] → {arm: [R,S,n_test]} (R = Stage 1 seed)"""
        o = {a: [] for a in ("a0", "b", "c", "d")}
        for k in keys:
            job = done[k]
            o["a0"].append(np.tile(s1[k[0]]["p1"][te], (SEEDS, 1)))
            for a in ("b", "c", "d"):
                z = job[a][0]
                assert np.array_equal(z["test_rows"], te)
                o[a].append(z["preds_test"])
        return {a: np.stack(v) for a, v in o.items()}

    pairs = [("c", "b"), ("c", "d"), ("d", "b"), ("b", "a0"), ("c", "a0")]
    lab = {"a0": "Stage 1 그대로", "b": "Stage 2 구조 없음", "c": "Stage 2 + 구조",
           "d": "Stage 2 + 셔플 구조"}

    # ---- 1. 주 분석
    if PRIMARY in done:
        oo = oof_block([PRIMARY])
        B = bootstrap(oo, y, typ, pos, np.zeros((1, len(te)), int), pairs, n_boot=2000)
        L += [f"## 1. 주 분석 — Stage 1 seed {PRIMARY[0]}, Stage 1 K/V 토큰으로 읽기", "",
              "| 팔 | subset | missense ρ | indel ρ |", "|---|---|---|---|"]
        for a in ("a0", "b", "c", "d"):
            s = B["single"][a]
            L.append(f"| {a}: {lab[a]} | {_fmt(s['subset'])} | {_fmt(s['missense'])} | {_fmt(s['indel'])} |")
        L += ["", "| 비교 | Δsubset [95% CI] | p(boot) | Δmissense | Δindel |", "|---|---|---|---|---|"]
        for A_, B_ in pairs:
            p = B["paired"][(A_, B_)]
            L.append(f"| {A_} − {B_} | {_pfmt(p, 'subset')} | {p['subset']['p_boot']:.3f} | "
                     f"{_pfmt(p, 'missense')} | {_pfmt(p, 'indel')} |")
        L += ["", "읽는 법: **c − d** = 구조 '정보'의 효과 (같은 모델, 구조만 섞음). **c − b** = 구조 경로를 "
              "붙였을 때의 실제 이득. **b − a0** = 구조 없이 Stage 2 를 더 학습한 효과.", ""]

    # ---- 2. Stage 1 seed · query 별
    L += ["## 2. Stage 1 seed · query 별 (보조)", "",
          "| Stage 1 seed | query | a0 subset | c − b | c − d | d − b | b − a0 |", "|---|---|---|---|---|---|---|"]
    for q in QUERIES:
        for s in S1_SEEDS:
            if (s, q) not in done:
                continue
            oo = oof_block([(s, q)])
            B = bootstrap(oo, y, typ, pos, np.zeros((1, len(te)), int), pairs[:4], n_boot=1000, seed=1)
            g = lambda a, b_: _pfmt(B["paired"][(a, b_)], "subset")
            L.append(f"| {s} | {q} | {B['single']['a0']['subset'][0]:.3f} | {g('c', 'b')} | {g('c', 'd')} | "
                     f"{g('d', 'b')} | {g('b', 'a0')} |")
        keys = [(s, q) for s in S1_SEEDS if (s, q) in done]
        if len(keys) > 1:
            oo = oof_block(keys)
            B = bootstrap(oo, y, typ, pos, np.zeros((len(keys), len(te)), int), pairs[:4], n_boot=2000, seed=2)
            g = lambda a, b_: _pfmt(B["paired"][(a, b_)], "subset")
            L.append(f"| **{len(keys)}개 합침** | {q} | {B['single']['a0']['subset'][0]:.3f} | **{g('c', 'b')}** | "
                     f"**{g('c', 'd')}** | {g('d', 'b')} | {g('b', 'a0')} |")
    L += ["", "합친 행: Stage 1 seed 를 반복 축으로 넣어 'seed 평균 성능'의 차이를 같은 재표집으로 계산.", ""]

    # ---- 3. attention
    if PRIMARY in done:
        L += [f"## 3. attention — 주 분석 c (해석용, 판정 아님)", ""]
        avail = s1[PRIMARY[0]]["avail"]
        for arm in ("c", "d"):
            A = done[PRIMARY][arm][0]["attn"]
            L += [f"**{arm}: {lab[arm]}** — 변이유형별 평균 (쓸 수 있는 전체 행, 균등이면 0.111)", "",
                  "| | " + " | ".join(TOKN) + " |", "|---|" + "---|" * 9]
            for gi, t in enumerate(C.TYPE3):
                m = avail & (d.typ == gi)
                L.append(f"| {t} | " + " | ".join(f"{v:.3f}" for v in A[m].mean(0)) + " |")
            mis = avail & (d.typ == C.TYPE3.index("missense"))
            per = []
            for p in np.unique(d.pos[mis]):
                dm = mis & (d.pos == p) & (fc == "fast depleted")
                ok = mis & (d.pos == p) & (fc == "unchanged")
                if dm.any() and ok.any():
                    per.append(A[dm].mean(0) - A[ok].mean(0))
            per = np.array(per)
            rng = np.random.default_rng(0)
            bs = np.array([per[rng.integers(0, len(per), len(per))].mean(0) for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            L.append(f"| 손상−무해 (같은 position, {len(per)}곳) | " + " | ".join(
                f"{m_:+.4f}{'*' if (l > 0 or h < 0) else ''}" for m_, l, h in zip(per.mean(0), lo, hi)) + " |")
            L.append("")
        L += ["`*` = 95% CI 가 0 을 포함하지 않음. 구조 토큰은 position 만의 함수라 같은 position 안의 차이는 "
              "Stage 1 쪽 query 에서만 온다. d 는 셔플 구조라 토큰 이름의 뜻이 없다(비교 기준).", ""]

    # ---- 4. 선택 기록
    L += ["## 4. 선택 기록", "",
          "| 작업 | 팔 | val 최고 | test (seed 평균) | val − test | 파라미터 | 멈춘 에폭 (seed별) | lr | 시간 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (s, q), job in sorted(done.items(), key=lambda kv: (kv[0][1] != "tok", kv[0][0] != 44, kv[0][0])):
        for a in ("b", "c", "d"):
            m = job[a][1]
            t = np.mean(m["test_subset_per_seed"])
            L.append(f"| s{s} {q} | {a} | {m['best_val_subset']:.3f} | {t:.3f} | {m['best_val_subset'] - t:+.3f} | "
                     f"{m['n_params']:,} | {m['best_epoch']} | {m['best_params']['lr']:.1e} | {m['seconds']:.0f}s |")
    L += ["", "val 은 Stage 1 의 early stopping 에도 쓰였으므로 val − test 낙관 폭에는 Stage 1 몫도 들어 있다.", ""]
    path = report_path or os.path.join(C.RESULTS_DIR, "REPORT_E7.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved -> {path}")
