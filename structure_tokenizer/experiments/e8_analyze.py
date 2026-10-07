"""
e8_analyze.py — E8 결과 정리 → results/REPORT_E8.md

  1. test 성능 (seed 5개 평균) 과 짝지은 차이 c−b, c−d, d−b  (position cluster bootstrap 2000회)
  2. 참고: 호준이 학습한 원래 Stage 1 (HF, E7 에서 복원) 과 우리가 다시 학습한 b 비교
  3. seed 별 val / test 와 멈춘 에폭
  4. attention (c, 해석용): 변이유형별 평균, 같은 position 안 손상 − 무해 missense
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config as C
from data import load_data
from stats import bootstrap


def analyze(out):
    from e8_joint import ARMS, SEEDS
    d = load_data()
    row_of = {v: i for i, v in enumerate(d.var_id)}
    fc_all = pd.read_csv(C.DATA_PATH, usecols=["functional_classification"])["functional_classification"].to_numpy()
    runs = {(a, s): (np.load(os.path.join(out, f"{a}_seed{s}.npz")), json.load(open(os.path.join(out, f"{a}_seed{s}.json"))))
            for a in ARMS for s in SEEDS if os.path.exists(os.path.join(out, f"{a}_seed{s}.json"))}
    pack_vid = np.load(os.path.join(C.RESULTS_DIR, "E7", "stage1_unified_reference_delta_W10_seed44.npz"))["var_id"]
    drow = np.array([row_of[v] for v in pack_vid])               # 묶음 행 → 우리 행
    te = runs[("b", SEEDS[0])][0]["test_rows"]
    for (a, s), (z, _) in runs.items():
        assert np.array_equal(z["test_rows"], te)
    y, typ, pos = d.y[drow[te]], d.typ[drow[te]], d.pos[drow[te]]
    oofs = {a: np.stack([runs[(a, s)][0]["pred_test"] for s in SEEDS])[None] for a in ARMS}
    # 호준 원래 Stage 1 (E7 복원) — 행 순서가 묶음과 같다 (같은 frozen.pt var_id 순서)
    hj = []
    for s in SEEDS:
        z = np.load(os.path.join(C.RESULTS_DIR, "E7", f"stage1_unified_reference_delta_W10_seed{s}.npz"))
        assert np.array_equal(z["var_id"], pack_vid)
        hj.append(z["pred"][te])
    oofs["hojun"] = np.stack(hj)[None]
    pairs = [("c", "b"), ("c", "d"), ("d", "b"), ("b", "hojun")]
    B = bootstrap(oofs, y, typ, pos, np.zeros((1, len(te)), int), pairs, n_boot=2000)
    lab = {"b": "Stage 1 (구조 없음, 다시 학습)", "c": "Stage 1 + 구조 (같이 학습)", "d": "Stage 1 + 셔플 구조",
           "hojun": "참고: 호준 원래 Stage 1 (HF)"}
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    g = lambda p, m: f"{p[m]['diff']:+.3f} [{p[m]['lo']:+.3f}, {p[m]['hi']:+.3f}]"
    L = ["# E8 — 구조 토큰을 Stage 1 과 처음부터 같이 학습하면 보탬이 되는가", "",
         f"- test {len(te)}개 (배포 분할, Val351dup 제외), position {len(np.unique(pos))}곳. 지표는 seed 별 계산 후 평균.",
         "- 학습은 호준 설정 그대로 (lr 1e-4, batch 32, Huber, val subset 으로 에폭 선택, patience 10). 구조 쪽은 "
         "Qk·n_bins 4·d_s 32 고정, 튜닝 없음.", "",
         "## 1. test 성능", "", "| 팔 | subset | missense ρ | indel ρ |", "|---|---|---|---|"]
    for a in ARMS + ["hojun"]:
        s = B["single"][a]
        L.append(f"| {a}: {lab[a]} | {f(s['subset'])} | {f(s['missense'])} | {f(s['indel'])} |")
    L += ["", "| 비교 | Δsubset [95% CI] | p(boot) | Δmissense | Δindel |", "|---|---|---|---|---|"]
    for A_, B_ in pairs:
        p = B["paired"][(A_, B_)]
        L.append(f"| {A_} − {B_} | {g(p, 'subset')} | {p['subset']['p_boot']:.3f} | {g(p, 'missense')} | {g(p, 'indel')} |")
    L += ["", "**c − d** = 구조 '정보'의 효과, **c − b** = 구조를 같이 학습했을 때의 실제 이득. "
          "b − hojun 은 우리 학습 환경이 호준 결과를 재현하는지 보는 참고값.", ""]

    # seed 별
    L += ["## 2. seed 별", "", "| 팔 | seed | val subset (에폭 선택에 씀) | test subset | test missense | test indel | 최고 에폭 / 돈 에폭 |",
          "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for s in SEEDS:
            m = runs[(a, s)][1]
            L.append(f"| {a} | {s} | {m['val_subset']:.3f} | {m['test_subset']:.3f} | {m['test_missense']:.3f} | "
                     f"{m['test_indel']:.3f} | {m['best_epoch']} / {m['epochs_run']} |")
        vm = np.mean([runs[(a, s)][1]["val_subset"] for s in SEEDS])
        tm = np.mean([runs[(a, s)][1]["test_subset"] for s in SEEDS])
        L.append(f"| **{a} 평균** | | **{vm:.3f}** | **{tm:.3f}** | | | |")
    L += [""]

    # attention
    A_all = np.zeros((d.n, 9))
    have = np.zeros(d.n, bool)
    A_all[drow] = np.mean([runs[("c", s)][0]["attn"] for s in SEEDS], axis=0)
    have[drow] = True
    TOKN = [x.replace("A_", "").replace("dist_", "d_") for x in C.FIELD_ORDER]
    L += ["## 3. attention — c (seed 5개 평균, 쓸 수 있는 전체 행, 해석용)", "",
          "| | " + " | ".join(TOKN) + " |", "|---|" + "---|" * 9]
    for gi, t in enumerate(C.TYPE3):
        m = have & (d.typ == gi)
        L.append(f"| {t} | " + " | ".join(f"{v:.3f}" for v in A_all[m].mean(0)) + " |")
    mis = have & (d.typ == C.TYPE3.index("missense"))
    per = []
    for p_ in np.unique(d.pos[mis]):
        dm, ok = mis & (d.pos == p_) & (fc_all == "fast depleted"), mis & (d.pos == p_) & (fc_all == "unchanged")
        if dm.any() and ok.any():
            per.append(A_all[dm].mean(0) - A_all[ok].mean(0))
    per = np.array(per)
    rng = np.random.default_rng(0)
    bs = np.array([per[rng.integers(0, len(per), len(per))].mean(0) for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    L.append(f"| 손상−무해 (같은 position, {len(per)}곳) | " + " | ".join(
        f"{m_:+.4f}{'*' if (l > 0 or h < 0) else ''}" for m_, l, h in zip(per.mean(0), lo, hi)) + " |")
    L += ["", "`*` = 95% CI 가 0 을 포함하지 않음. attention 은 '어디를 읽었나'이지 중요도가 아니다.", ""]
    path = os.path.join(C.RESULTS_DIR, "REPORT_E8.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved -> {path}")
