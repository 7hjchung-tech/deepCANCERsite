"""
compare.py — E1·E3 결과를 모아 판정표를 만든다 → results/REPORT_E1_E3.md

표
  1. 모든 모델의 out-of-fold 성능 (천장 GBDT, 바닥 TYPE 포함) + position cluster bootstrap 95% CI
  2. 판정: 인코딩 후보 vs 기준 Qk — paired Δ, CI, bootstrap p, Holm, corrected t
  3. 대조군·천장 (해석용): 셔플 구조, GBDT, TYPE
  4. 낙관 폭: 고를 때 점수(inner) − 실제 점수(outer) — 선택 과적합 (Cawley & Talbot 2010)
  5. 변이유형별 attention 가중치 — 해석용
  6. 선택된 하이퍼파라미터·예산 곡선(Dodge+ 2019)·fANOVA

판정 규칙 (실험 전에 정한 것)
  · 주 지표 subset. Δ 의 95% CI 가 0 을 넘으면(Holm 보정 bootstrap p<0.05 와 함께) "구분됨"
  · 현재 설정(Qk)은 대안이 구분되게 이길 때만 바꾼다. 동률이면 단순한 쪽
  · 검출 한계 MDE ≈ 0.04: CI 가 ±MDE 안에 다 들어오면 "실용적으로 동등", 걸치면 "측정 불가"

실행:  python compare.py [--results results] [--n_boot 2000] [--no_fanova]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import optuna
import pandas as pd

import config as C
from data import load_data
from metrics import subset_score
from splits import nested_split
from stats import bootstrap, corrected_ttest, expected_max, fold_diffs, holm

MDE = 0.04
REF = "Qk"
FAMILY = ["L", "Q", "T", "PLR"]
CONTROLS = ["Qk-shuf", "GBDT", "TYPE"]


def load_oofs(root, d, names=None):
    oofs, info, missing = {}, {}, {}
    K = C.PROTOCOL["outer_k"]
    for vdir in sorted(glob.glob(os.path.join(root, "*"))):
        name = os.path.basename(vdir)
        if name.startswith("_") or not os.path.isdir(vdir) or (names and name not in names):
            continue
        jobs = sorted(glob.glob(os.path.join(vdir, "r*_k*", "done.json")))
        if not jobs:
            continue
        sels = [json.load(open(os.path.join(os.path.dirname(j), "selected.json"))) for j in jobs]
        R_here = max(s["repeat"] for s in sels) + 1
        S = len(np.load(os.path.join(os.path.dirname(jobs[0]), "pred.npz"))["seeds"])
        oof = np.full((R_here, S, d.n), np.nan)
        for j, s in zip(jobs, sels):
            z = np.load(os.path.join(os.path.dirname(j), "pred.npz"))
            oof[s["repeat"], :, z["test_rows"]] = z["preds"].T
        if not np.isfinite(oof).all():
            missing[name] = f"{len(jobs)}/{R_here * K} 작업"
            continue
        oofs[name], info[name] = oof, sels
    return oofs, info, missing


def fmt(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def verdict(dd):
    lo, hi = dd["lo"], dd["hi"]
    stat = "대안 우세" if lo > 0 else ("현재 우세" if hi < 0 else "구분 안 됨")
    if -MDE < lo and hi < MDE:
        prac = "실용적 동등"
    elif lo > MDE or hi < -MDE:
        prac = "실용적 차이"
    else:
        prac = "측정 불가 폭"
    return f"{stat} · {prac}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULTS_DIR)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--no_fanova", action="store_true")
    a = ap.parse_args()
    d = load_data()
    wanted = set([REF] + FAMILY + CONTROLS)
    oofs, info, missing = load_oofs(a.results, d, wanted)
    R = min(o.shape[0] for o in oofs.values())
    oofs = {k: v[:R] for k, v in oofs.items()}
    lines = ["# E1 · E3 — 구조만으로 어디까지 / 인코딩 5가지 비교", "",
             f"- 완료된 모델: {', '.join(sorted(oofs))}",
             f"- 미완료/없음(제외): {sorted(set(wanted) - set(oofs)) or '없음'} {missing or ''}",
             f"- repeat {R} × outer {C.PROTOCOL['outer_k']}-fold, 모든 행이 repeat 마다 한 번씩 test",
             f"- 주 지표 subset = mean(Spearman_missense, Spearman_indel). **바깥 fold 마다 계산해 "
             f"평균** (Forman & Scholz 2010). 괄호는 position cluster bootstrap 95% CI "
             f"({a.n_boot}회). synonymous 는 해석하지 않음 (명세 §7.4)", ""]

    cell = pd.Series(d.y).groupby([d.pos, d.typ]).transform("mean").to_numpy()
    oracle_subset = subset_score(cell, d.y, d.typ)

    ref = REF
    pairs = [(v, ref) for v in FAMILY + CONTROLS if v in oofs and ref in oofs]
    rk = {(r, k): nested_split(d.pos, r, k)["outer_test"]
          for r in range(R) for k in range(C.PROTOCOL["outer_k"])}
    fold_of = np.full((R, d.n), -1)
    for (r, k), rows in rk.items():
        fold_of[r, rows] = k
    assert (fold_of >= 0).all()
    print(f"bootstrap {a.n_boot}회 · 모델 {len(oofs)}개 · 비교 {len(pairs)}쌍 …", flush=True)
    B = bootstrap(oofs, d.y, d.typ, d.pos, fold_of, pairs, n_boot=a.n_boot, seed=0)

    # ---------------------------------------------------------------- 1
    lines += ["## 1. 모든 모델의 out-of-fold 성능 (천장·바닥 포함)", "",
              "| 모델 | subset | missense ρ | indel ρ |", "|---|---|---|---|"]
    order = sorted(oofs, key=lambda n: -B["single"][n]["subset"][0])
    for n in order:
        s = B["single"][n]
        lines.append(f"| {n} | {fmt(*s['subset'])} | {fmt(*s['missense'])} | {fmt(*s['indel'])} |")
    lines += [f"| (상한) position×type oracle, in-sample | {oracle_subset:.3f} | | |", "",
              "- GBDT = 원래 9개 값으로 뽑을 수 있는 최대치의 추정(천장), TYPE = 바닥, "
              "`-shuf` = position↔구조 대응을 끊은 대조군", ""]

    def block(title, rows, family=True, note=""):
        rows = [p for p in rows if p in B["paired"]]
        if not rows:
            return []
        ph = holm({p: B["paired"][p]["subset"]["p_boot"] for p in rows}) if family else {}
        out = [f"## {title}", "", note or "Δ = 앞 − 뒤.", "",
               "| 비교 | Δsubset [95% CI] | p(boot) | Holm p | Δmissense | Δindel | corrected-t p | 판정 |",
               "|---|---|---|---|---|---|---|---|"]
        for p in rows:
            pdd = B["paired"][p]
            ct = corrected_ttest(fold_diffs(oofs[p[0]], oofs[p[1]], d.y, d.typ, rk),
                                 C.PROTOCOL["outer_k"])
            s = pdd["subset"]
            out.append(f"| {p[0]} − {p[1]} | {s['diff']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] | "
                       f"{s['p_boot']:.3f} | {ph.get(p, float('nan')):.3f} | "
                       f"{pdd['missense']['diff']:+.3f} | {pdd['indel']['diff']:+.3f} | "
                       f"{ct['p']:.3f} | {verdict(s)} |")
        return out + [""]

    lines += block("2. 판정 — 인코딩 후보 vs 기준", [(v, ref) for v in FAMILY],
                   note=f"기준 = `{ref}`. Δ = 후보 − 기준.")
    lines += block("3. 대조군과 천장 (판정 아님, 해석용)", [(v, ref) for v in CONTROLS],
                   family=False)
    lines += ["- `-shuf` 의 Δ 가 크게 음수여야 구조 정보가 실제로 쓰인다는 뜻 (selectivity)",
              "- `GBDT` 의 Δ 가 양수면 그만큼이 **토큰화로 더 회수할 여유**", ""]

    # ---------------------------------------------------------------- 5 낙관 폭
    lines += ["## 4. 낙관 폭 — 고를 때 점수(inner) − 실제 점수(outer)", "",
              "fold 10개 평균. 클수록 하이퍼파라미터 선택이 안쪽 fold 의 잡음에 맞춰졌다는 뜻.", "",
              "| 모델 | 고를 때 | 실제 | 낙관 폭 | 파라미터 수(중앙값) |", "|---|---|---|---|---|"]
    for n in [ref] + FAMILY:
        if n not in info:
            continue
        inner = np.mean([s["best_inner_subset"] for s in info[n]])
        outer = np.mean([np.mean(s["outer_test_subset_per_seed"]) for s in info[n]])
        npar = np.median([s["refit"].get("n_params", np.nan) for s in info[n]])
        lines.append(f"| {n} | {inner:.3f} | {outer:.3f} | {inner - outer:.3f} | {npar:,.0f} |")
    lines.append("")

    # ---------------------------------------------------------------- 6 attention
    att_models = [n for n in [ref] + FAMILY if n in info and "attn_by_type" in info[n][0]["refit"]]
    if att_models:
        tok = [f.replace("A_", "").replace("dist_", "d_") for f in C.FIELD_ORDER]
        lines += ["## 5. 변이유형별 attention 가중치 (해석용, 판정 아님)", "",
                  "바깥 test 행에서 토큰 9개에 준 가중치의 평균(head·seed·fold 평균). 균등이면 각 0.111.", ""]
        for n in att_models:
            lines += [f"**{n}**", "", "| 유형 | " + " | ".join(tok) + " |",
                      "|---|" + "---|" * len(tok)]
            for t in C.TYPE3:
                ws = [s["refit"]["attn_by_type"][t] for s in info[n] if t in s["refit"]["attn_by_type"]]
                m = np.mean(ws, axis=0)
                lines.append(f"| {t} | " + " | ".join(f"{v:.3f}" for v in m) + " |")
            lines.append("")

    # ---------------------------------------------------------------- 7 하이퍼파라미터
    lines += ["## 6. 선택된 하이퍼파라미터 (바깥 fold 마다) · 예산 곡선 · fANOVA", ""]
    ns = [1, 5, 10, 25, 50, 100]
    for n in [ref] + FAMILY + [c for c in CONTROLS if c in info]:
        if n not in info or not info[n][0]["best_params"]:
            continue
        sel = info[n]
        df = pd.DataFrame([s["best_params"] for s in sel])
        desc = []
        for c in df.columns:
            v = df[c]
            if v.dtype == object or c in ("d_s", "extrapolate", "n_bins"):
                desc.append(f"{c}: " + ", ".join(f"{k}×{c_}" for k, c_ in v.value_counts().items()))
            else:
                desc.append(f"{c}: 중앙값 {v.median():.3g} (범위 {v.min():.3g}–{v.max():.3g})")
        tfiles = [os.path.join(a.results, n, f"r{s['repeat']}_k{s['fold']}", "trials.csv") for s in sel]
        tdfs = [pd.read_csv(f) for f in tfiles]
        curves = [expected_max(t["value"].to_numpy(float), ns) for t in tdfs]
        em = {k: np.mean([c[k] for c in curves if k in c]) for k in ns}
        nfail = sum(t["state"].ne("COMPLETE").sum() for t in tdfs)
        lines += [f"**{n}** (실패 trial {nfail}개, trial 수 {len(tdfs[0])})", "",
                  "- " + "\n- ".join(desc),
                  "- 기대 최고 inner subset (trial 수 n): " +
                  " · ".join(f"n={k}: {v:.4f}" for k, v in em.items() if k <= len(tdfs[0])), ""]
        if not a.no_fanova and n != "TYPE":
            imps = []
            for s in sel:
                db = os.path.join(a.results, n, f"r{s['repeat']}_k{s['fold']}", "study.db")
                st = optuna.load_study(study_name="hpo", storage=f"sqlite:///{db}")
                try:
                    imps.append(optuna.importance.get_param_importances(
                        st, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0)))
                except Exception as e:
                    lines.append(f"- fANOVA 생략: {type(e).__name__}")
                    break
            if imps:
                keys = sorted({k for i in imps for k in i})
                mean = {k: np.mean([i.get(k, 0.0) for i in imps]) for k in keys}
                lines.append("- fANOVA 중요도(평균): " + " · ".join(
                    f"{k} {v:.2f}" for k, v in sorted(mean.items(), key=lambda kv: -kv[1])))
                lines.append("")

    out = os.path.join(a.results, "REPORT_E1_E3.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    tag = ""
    rows = [{"A": A_, "B": B_, "metric": k, **v}
            for (A_, B_), m in B["paired"].items() for k, v in m.items()]
    pd.DataFrame(rows).to_csv(os.path.join(a.results, f"paired{tag}.csv"), index=False)
    pd.DataFrame([{"model": n, "metric": k, "point": v[0], "lo": v[1], "hi": v[2]}
                  for n, m in B["single"].items() for k, v in m.items()]).to_csv(
        os.path.join(a.results, f"single{tag}.csv"), index=False)
    print("\n".join(lines))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
