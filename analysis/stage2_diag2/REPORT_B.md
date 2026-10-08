# Stage 2 정체 원인 진단 B: 수치 오차 vs 순위 지표, 보정 크기 (2026-10-07)

> **후속 보고서**: 이 진단 이후 제한적 Stage1 공동 fine-tuning(실제 vs 고정 구조 ablation)을
> `analysis/stage2_joint_ablation/REPORT_D.md`에서, 3D 구조 이웃(anchor+8) FiLM 보정을
> `analysis/stage2_neighborhood/REPORT_E.md`에서 실행했다. 이 보고서는 그대로 보존했다.

범위: Stage 1 baseline, R1, R2, 참고용 R0 (이미 학습된 `runs/stage2_reverse/{r0,r1,r2}_*/seed42`). 새 모델·공동 fine-tuning·OOF 재학습은 하지 않았다. test split은 로드하지 않았다. 기존 결과 디렉터리(`runs/stage2_reverse/`, `analysis/stage2_diag/`)는 읽기만 했고, 이 보고서와 산출물은 `analysis/stage2_diag2/`에 새로 저장했다.

## 0. 공통 준비에서 확인한 것

- checkpoint: 세 모델 모두 `best_stage2.pt` 하나만 저장돼 있다(best epoch=1인 checkpoint). **last epoch(11)의 가중치는 저장되지 않았다** — `train_stage2_reverse.py`가 best state만 저장하기 때문이다. 그래서 last epoch에서는 학습 중 기록된 집계 지표(§2)만 쓸 수 있고, per-sample 추론(§2 train 쪽, §3 alpha blending)은 best epoch에서만 가능하다. 재학습으로 채우지 않았다.
- split/variant ID: `src/stage2/diagnostics.load_setup`로 train 4124 / val 881행을 cohort에서 복원하고, DataLoader(`shuffle=False`)의 var_id 순서가 cohort 순서와 정확히 같은지 매 스크립트에서 `assert`로 확인했다(통과). 모든 예측은 var_id로 라벨과 묶었고, 행 순서만으로 합치지 않았다.
- subset metric: `mean(spearman_missense, spearman_indel)` (synonymous 제외), 기존 Stage 1/Stage 2와 동일 정의.
- Huber: `delta=1.0`, `reduction='mean'`, sample weighting 없음. weight decay(0.01)는 AdamW 내부에만 적용되고 `train_task_loss`에는 더해지지 않는다 — 로그의 `train_task_loss`는 순수 task Huber다. `lambda_sp`(L2-SP)는 세 모델 모두 `frozen_stage1` 모드라 0이다(`train_l2sp_penalty` 전부 0 확인).
- target: z-score 그대로, Stage 1 예측(y1)은 `y_std·pred+y_mean`으로 이미 원래 scale로 변환돼 있다(label과 같은 단위). 별도 정규화 없음.
- best checkpoint 선택 기준: validation subset Spearman, `min_delta=0`, `patience=10`. 세 모델 모두 `best_epoch=1`, `stop_reason="patience"`, 총 11 epoch(epoch 0 제외) 학습 후 종료.
- **train-mode vs eval-mode 주의**: `train_task_loss`는 학습 중 매 minibatch 이후(파라미터가 계속 바뀌는 도중) 누적 평균한 값이라, "epoch 끝의 고정된 모델 하나"의 loss가 아니다. eval-mode로 정확히 재현 가능한 지점은 가중치가 저장된 **best epoch(=epoch 1)뿐**이다.

## 1. 진단 A: 수치 오차와 순위 지표가 같이 움직이는가?

산출물: `epoch_metrics_val.csv`(매 epoch, val 전체), `best_epoch_full_metrics_with_stage1.csv`(best epoch, train+val, eval-mode 추론), `best_vs_last_summary.csv`, `curves_huber_mae.png`, `curves_spearman.png`.

### best epoch(=1)에서 이미 벌어진 gap (eval-mode, train+val 동시 비교)

| | train Huber | train MAE | train subset | val Huber | val MAE | val subset |
|---|---|---|---|---|---|---|
| Stage 1 | 1.4077 | 1.8302 | 0.79963 | 1.8652 | 2.2888 | 0.76636 |
| R0 (best=ep1) | 1.3992 | 1.8218 | 0.79976 | 1.8698 | 2.2928 | 0.76629 |
| R1 (best=ep1) | 1.3992 | 1.8218 | 0.79981 | 1.8688 | 2.2919 | 0.76632 |
| R2 (best=ep1) | 1.3958 | 1.8183 | 0.79998 | 1.8697 | 2.2925 | 0.76622 |

- epoch 1 하나만 학습했는데도 **train은 이미 Stage 1보다 살짝 좋아지고(Huber −0.006~−0.012), val은 이미 살짝 나빠졌다(Huber +0.003~+0.005)**. subset spearman은 train·val 모두 사실상 그대로다(소수 4자리까지 거의 동일). 즉 "best"로 선택된 체크포인트조차 수치 오차 기준으로는 이미 Stage 1보다 좋지 않다 — subset spearman으로 선택했기 때문에 이 미세한 수치 악화가 가려져 있었을 뿐이다.

### best(ep1) → last(ep11), val (학습 로그에 기록된 집계치, 재추론 아님)

| | val subset best→last | val RMSE best→last | val MAE best→last | missense best→last | synonymous best→last | indel best→last |
|---|---|---|---|---|---|---|
| Stage 1 (상수) | 0.7664 | 3.723 | 2.289 | 0.6951 | 0.2214 | 0.8376 |
| R0 | 0.7663→0.7403 | 3.736→3.857 | 2.293→2.339 | 0.6950→0.6350 | 0.2244→0.2522 | 0.8376→0.8455 |
| R1 | 0.7663→0.7493 | 3.735→3.854 | 2.292→2.331 | 0.6950→0.6556 | 0.2252→0.2484 | 0.8376→0.8430 |
| R2 | 0.7662→0.7501 | 3.739→3.914 | 2.293→2.356 | 0.6948→0.6590 | 0.2253→0.2473 | 0.8376→0.8413 |

(`best_vs_last_summary.csv`. last epoch의 val Huber와 train 쪽 지표는 가중치가 없어 계산하지 못했다 — `None`/"unavailable"로 표시했다.)

### 그림

`curves_huber_mae.png`(왼쪽: train Huber 학습 중 로그 vs val MAE, 오른쪽: val subset): **train Huber는 11 epoch 내내 단조 감소**(R0/R1/R2 거의 겹침)하고, **val MAE는 epoch 1 이후 단조에 가깝게 증가**해서 Stage 1 기준선 위에 계속 머문다. 전형적인 train/val 분기(과적합) 모양이다.

`curves_spearman.png`(missense/synonymous/indel 세 패널, 세 모델 겹쳐 그림): **missense만 epoch 1 직후 급락**(0.695→0.63대)해서 Stage 1선 아래로 계속 남는다. **synonymous와 indel은 반대로 epoch 2부터 Stage 1선 위로 올라가서 11 epoch 내내 그 위에 머문다.** (synonymous는 subset 지표에 안 들어가지만 방향은 indel과 같다.)

### 패턴 판정

- **A (train↓, val 수치·순위 모두 악화)**: 전체(all-rows) 기준으로는 지지된다. train Huber는 계속 내려가고, val Huber/RMSE/MAE는 best 이후 계속 올라가며, val overall/subset spearman도 대체로 같이 내려간다.
- **C (유형별로 갈림)**: 동시에, 유형별로 쪼개면 분명히 다른 그림이 나온다. **missense(n=671, val의 76%)가 악화를 주도**하고, **synonymous(n=158)·indel(n=52)은 오히려 Stage 1보다 개선된 채로 11 epoch 내내 유지**된다. subset 지표가 missense와 indel의 평균이라서, indel의 작은 개선(+0.004~+0.008)이 missense의 큰 악화(−0.036~−0.060)에 묻혀 전체 subset이 떨어지는 것으로 보인다.
- **B(수치는 좋아지는데 순위만 나빠짐)**: 지지되지 않는다 — val Huber/MAE/RMSE가 개선되는 구간이 없다.
- **D(train도 개선 안 됨)**: 지지되지 않는다 — train Huber는 꾸준히, 뚜렷하게 개선된다. 그래서 "일반화 문제만으로는 설명 안 되는 최적화 실패"의 근거는 이번 진단에서 나오지 않았다.
- 한 패턴으로 억지로 정리하지 않는다: **"전체적으로는 A(과적합), 그 안에서 내용적으로는 C(유형별 상반된 방향)"**가 혼재한 결과다.

## 2. 진단 B: 보정 방향은 맞는데 크기만 과한가?

산출물: `alpha_metrics.csv`, `alpha_bootstrap.csv`, `alpha_delta_diagnostics.csv`, `alpha_curves.png`.

y_alpha = y_stage1 + alpha·delta_y, alpha ∈ {0, 0.1, 0.25, 0.5, 1.0}. **checkpoint는 best(epoch 1)만 가능하다** — last epoch 가중치가 없어 last는 분석하지 못했다(§0). `delta_y`는 모델이 직접 반환한 값을 썼고, 모든 체크포인트에서 `y_final ≈ y1+delta_y`(오차 ≤1e-4)를 확인했다.

### alpha 곡선 (val, best checkpoint)

`alpha_curves.png`: 세 모델 모두 **alpha가 커질수록 Huber와 MAE는 단조 증가**(alpha=0이 최선), **subset spearman은 alpha=0에서 가장 높거나(R1은 alpha=0.1에서 미세하게 더 높음, 차이 <0.00002) 거의 평평하다가 alpha=0.5~1에서 더 떨어진다.** 작은 alpha(0.1, 0.25)가 양끝(0과 1)보다 뚜렷하게 나은 구간은 없다. 변화 폭 자체가 매우 작다(Huber 변화 ~0.001~0.005, subset spearman 변화 ~0.00002~0.00014) — 실질적 개선으로 보기 어렵다.

### position 단위 paired bootstrap (B=2000, val subset metric, alpha − alpha0 차이)

| model | alpha | 관측 차이 | 95% CI | P(차이≤0) |
|---|---|---|---|---|
| R0 | 1.0 | −0.0001 | [−0.0004, 0.0003] | 0.658 |
| R1 | 1.0 | −0.0000 | [−0.0003, 0.0003] | 0.602 |
| R2 | 1.0 | −0.0001 | [−0.0007, 0.0004] | 0.683 |

(0.1/0.25/0.5의 CI도 전부 0을 포함한다. `alpha_bootstrap.csv` 전체 참고. replicate 실패 0건.) **alpha=0(Stage 1)과 alpha=1(기존 Stage 2) 사이에 유의한 차이가 없다** — 이는 이전 보고서(`analysis/stage2_diag/REPORT.md` §8)의 결론과 일치한다.

### 보조 분석 (best checkpoint, delta_y 통계)

| model | split | delta 평균 | delta RMS | corr(delta, Stage1 잔차) | \|오차\| 줄어든 표본 비율 | 평균 \|오차\| before→after |
|---|---|---|---|---|---|---|
| R0 | train | 0.055 | 0.076 | **+0.137** | **54.9%** | 1.830→1.822 |
| R0 | val | 0.052 | 0.075 | **−0.027** | **47.9%** | 2.289→2.293 |
| R1 | train | 0.049 | 0.072 | +0.138 | 54.9% | 1.830→1.822 |
| R1 | val | 0.047 | 0.071 | −0.027 | 47.8% | 2.289→2.292 |
| R2 | train | 0.063 | 0.090 | **+0.162** | 54.8% | 1.830→1.818 |
| R2 | val | 0.059 | 0.088 | **−0.034** | 47.9% | 2.289→2.293 |

- train에서는 delta_y가 Stage 1 잔차(y_true−y1)와 약하지만 같은 방향으로 움직이고(상관 +0.14~0.16), 표본의 54.9%에서 절대오차가 줄어든다.
- val에서는 그 상관이 거의 0이거나 **부호가 반대**(−0.03)이고, 표본의 **47.8~47.9%만** 절대오차가 준다 — 즉 **val에서는 절반 넘는 표본에서 보정이 오차를 더 키운다.** R2가 R1보다 상관의 절댓값이 살짝 크지만(train +0.162 vs +0.138, val −0.034 vs −0.027) 부호와 방향은 완전히 같다.
- 단순 상관 하나로 Spearman 변화를 설명하지 않는다 — 위 상관·오차감소비율은 §1의 "missense 악화/synonymous·indel 개선" 패턴과 별개로, **전체 평균 수치**만 본 것이다. 유형별로 쪼갠 상관은 `alpha_delta_diagnostics.csv`에 있다(missense 상관이 전체 평균과 비슷하게 가장 큰 음의 방향을 끈다).

### 해석

- **alpha=0이 best 체크포인트에서는 거의 항상 최선이거나 차이가 없다.** "작은 alpha가 둘보다 낫다"는 패턴은 나타나지 않았다.
- Huber/MAE가 선호하는 alpha(=0)와 subset spearman이 선호하는 alpha(R0/R2는 0, R1은 0.1이지만 차이 2e-5 수준)는 **사실상 같다** — 이번 데이터에서는 목적함수-순위지표 불일치가 뚜렷하게 나타나지 않았다.
- "alpha=0이 최선"이라는 결과를 "구조 정보가 무가치하다"로 확대하지 않는다. 이는 **이 특정 seed·best epoch(1)·frozen Stage1 조합**에서 학습된 delta_y가 유용하지 않았다는 것이고, train에서는 보정 방향이 약하게나마 맞는 방향(+상관, 54.9% 개선)이라는 근거도 함께 있다 — 다만 그 신호가 val로 일반화되지 않을 뿐이다.
- val에서 alpha를 고르는 것 자체가 validation에 맞춘 추가 tuning이다. 이 bootstrap CI는 "주어진 alpha가 고정됐을 때의 불확실성"이지, alpha를 validation에서 고르는 절차 자체의 선택 편향까지 반영하지 않는다. alpha=0.1(R1에서 가장 높았던 값)의 성능을 독립적인 일반화 성능으로 보고하지 않는다.

## 3. 불확실성·데이터 한계

- 위치(pp) 단위 paired bootstrap, B=2000, 모든 replicate 성공(실패 0건) — val 881행이 52~53개 position 블록으로 나뉘는 구조라, 실패(해당 resample에 missense나 indel이 2개 미만) 위험이 있었지만 발생하지 않았다.
- 단일 seed(42) 결과다. 여러 seed의 재현성으로 해석하지 않는다.
- **last epoch 가중치가 저장되지 않아서** last epoch의 train 쪽 지표, val Huber, alpha blending, delta_y 상관 분석을 전혀 하지 못했다. 재학습으로 채우지 않았다(요청대로). last epoch는 §1의 학습 로그 집계치(subset/RMSE/MAE/spearman by-group)로만 다뤘다.
- train 쪽 매 epoch(best 제외) 수치는 전혀 복원하지 못했다(`train_task_loss`만 train-mode로 존재). 재학습 없이는 채울 수 없다.

## 4. 질문별 답

**1. Validation의 수치 오차도 악화되는가, 순위 지표만 악화되는가?**
둘 다 악화되지만 **크기가 다르다**. best epoch(1)에서 이미 val Huber/MAE/RMSE가 Stage 1보다 미세하게 나빠져 있고(subset spearman은 거의 그대로), epoch가 진행될수록 Huber/MAE/RMSE가 꾸준히 나빠지면서 subset spearman도 대체로 같이 나빠진다. 순위 지표만 떨어지고 수치는 그대로이거나 좋아지는 패턴(B)은 관찰되지 않았다.

**2. 작은 alpha가 Stage1보다 나은가, 아니면 보정하지 않는 것이 최선인가?**
best 체크포인트에서는 **보정하지 않는 것(alpha=0)이 사실상 최선**이다. alpha 0.1~1.0 전 구간에서 Stage 1 대비 유의한 이득이 없고(bootstrap CI가 전부 0 포함), Huber/MAE는 alpha가 커질수록 단조 악화된다. 다만 train에서는 delta_y가 Stage1 잔차와 약하게 같은 방향(correlation +0.14~0.16)이라는 근거가 있어, "구조 정보 자체가 무가치하다"는 결론까지는 내리지 않는다 — 그 신호가 val로 일반화되지 않는다는 것이 더 정확한 설명이다.

**3. R1과 R2의 실패 패턴은 같은가?**
**같다.** 둘 다 best_epoch=1, val subset이 epoch 2부터 떨어지다 일부 회복, missense가 악화를 주도하고 synonymous·indel은 Stage 1보다 개선된 채 유지되는 유형별 분기, alpha=0이 사실상 최선, val에서 delta_y와 잔차의 상관이 거의 0이거나 음수, 보정이 val 표본의 절반 미만에서만 오차를 줄이는 것까지 패턴이 동일하다. 차이는 크기뿐이다 — R2가 용량·학습 가능한 τ 덕분에 delta_y를 더 크게(train/val RMS 0.072→0.088~0.090), train 상관을 더 강하게(+0.138→+0.162) 만들었지만, val 상관은 오히려 더 음의 방향으로(−0.027→−0.034) 움직였다. 즉 **R2는 R1과 같은 방향으로, 더 세게** 반응한다.

**4. 현재 결과가 지지하는 설명과 여전히 구분하지 못하는 설명은?**
- *지지됨*: (a) train/val 오차가 분기하는 과적합 패턴(§1 패턴 A), (b) missense와 synonymous/indel이 반대 방향으로 움직이는 유형별 분기(§1 패턴 C), (c) 학습된 delta_y가 train에서는 약하게 올바른 방향이지만 val에서는 그렇지 않다는 것(§2), (d) R1·R2가 동일한 실패 패턴을 공유한다는 것(attention 설계 자체가 원인이 아니라는 기존 결론과 일치).
- *구분하지 못함*: (i) 이 분기가 순전히 표본 수 차이(missense 671개로 더 세밀한 신호를 학습할 여지가 큰 반면 synonymous/indel은 표본이 적어 더 거친/안정적인 보정만 학습됐을 가능성) 때문인지, 구조 feature 자체가 missense에서만 체계적으로 오도하는 신호를 담고 있는지는 이번 진단으로 구분할 수 없다. (ii) `analysis/stage2_diag/REPORT.md`에서 제기된 "train residual이 in-sample이라 val과 다르다"는 가설과 이번 결과의 관계 — 이번 진단은 그 가설을 직접 건드리지 않았다(§2는 alpha blending이지 residual target 교체가 아니다). (iii) missense 악화가 특정 position/도메인에 몰려 있는지는 측정하지 않았다.

**5. 다음 실험으로 제한적 Stage1 공동 fine-tuning을 진행할 근거가 있는가?**
**아직 근거가 부족하다.** 이유:
- R1과 R2가 attention 용량·날카로움이 크게 다른데도 **완전히 같은 실패 패턴**을 보인다는 것은, 문제가 Stage 2의 attention/구조 融合 설계가 아니라 **더 상류(잔차 target 또는 데이터)**에 있다는 기존 가설과 일치한다. Stage 1 자체를 건드리는 것은 이 상류 문제를 겨냥한 조치이긴 하지만,
- 더 직접적이고 비용이 작은 미시험 경로(OOF 잔차 target의 fold 모델을 강화해 재시도 — `analysis/stage2_diag/REPORT.md` §9)가 아직 제대로 시도되지 않았다. 지난 OOF 실행은 fold 모델이 약해서(val subset 0.685~0.743 대 shipped 0.766) 결론이 교란됐다.
- 공동 fine-tuning은 frozen Stage 1(161,986 파라미터)의 일부를 다시 학습 가능하게 만드는 것이라, 지금의 작은 val(881행)에서 **과적합 위험을 한 축 더 늘린다**. 이번 진단에서 본 과적합 속도(epoch 1만에 이미 train 개선·val 악화가 시작)를 고려하면, 상류 문제를 먼저 좁히지 않고 자유도를 늘리는 것은 순서가 이르다.
- 권장 순서: (1) OOF fold 모델을 shipped 수준으로 강화해 재시도 → (2) 그래도 정체되면, missense만 악화되는 이유를 position/구조 feature 단위로 더 쪼개보기 → (3) 그 다음에야 제한적 Stage 1 공동 fine-tuning을 고려.

## 산출물

- `analysis/stage2_diag2/epoch_metrics_val.csv`, `best_epoch_full_metrics.csv`, `best_epoch_full_metrics_with_stage1.csv`, `best_vs_last_summary.csv`, `stage1_baseline_full_metrics.json`
- `analysis/stage2_diag2/alpha_metrics.csv`, `alpha_bootstrap.csv`, `alpha_delta_diagnostics.csv`
- `analysis/stage2_diag2/curves_huber_mae.png`, `curves_spearman.png`, `alpha_curves.png`
- 스크립트: `scripts/stage2_diag2_curves.py`(진단 A), `scripts/stage2_diag2_alpha.py`(진단 B)
- 재실행: `python scripts/stage2_diag2_curves.py && python scripts/stage2_diag2_alpha.py`
- 관련 기존 보고서: `analysis/stage2_diag/REPORT.md` (R0/R1/R2 구현·초기 비교, OOF 실험 포함)
