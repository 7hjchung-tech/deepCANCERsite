# R1 + 제한적 Stage1 공동 fine-tuning 비교 (A/B/C/D, 2026-10-07)

> **후속 보고서**: 3D 구조 이웃(anchor+8) FiLM 보정(E0/E1/E2)은
> `analysis/stage2_neighborhood/REPORT_E.md`. 이 보고서는 그대로 보존했다.

단일 seed(Stage2=42, Stage1 checkpoint=seed44 W10)의 탐색적 결과다. OOF·ESM fine-tuning·대규모 sweep은 하지 않았다. test split은 로드하지 않았다. 기존 결과(`runs/stage2_reverse/`, `analysis/stage2_diag/`, `analysis/stage2_diag2/`)는 보존했고, 이번 산출물은 `runs/stage2_joint_ablation/seed42/`와 `analysis/stage2_joint_ablation/`에 새로 저장했다.

## 0. 실험 구성

Fusion은 R1(서열 쿼리 → 구조 K/V, 1 head×32) 고정. 4개 조건:

| | Stage1 | 구조 입력 |
|---|---|---|
| A | frozen | 실제 구조 |
| B | 공동 학습(content_builder+pooling+head) | 실제 구조 |
| C | 공동 학습(동일) | 고정값(train median/mode) |
| D | frozen | 고정값(train median/mode) |

- **고정값**: C/D는 `src/stage2/structure.FixedStructureStore`로 구현했다 — 연속형 8개는 train 행의 중앙값, 범주형(이차구조)은 train 행의 최빈값을 모든 샘플에 동일하게 먹인다. tokenizer의 PLE bin 경계는 **항상 실제 train 분포로 fit**했다(상수만 넣으면 분위수 경계를 만들 수 없어서 — 고정값은 fit 이후 입력 단계에서만 적용). field-ID(tokenizer의 field_emb)와 variant-type(`type_emb`, 별도 `type_id`로 전달)은 구조값과 무관하므로 그대로 유지된다. 실제 구조값만 상수로 바뀐다.
- **공동 학습 unfreeze**: `content_builder`(86,368) + `pooling`(129, query q0와 log_tau 포함) + `head`(66,561) = **153,058 파라미터**. `layer_embedding`·`metadata_encoder`(합 8,928)는 그대로 frozen.
- ESM은 모든 조건에서 frozen(raw hidden은 캐시에서 읽고, Stage1의 자체 projection은 매 forward마다 다시 계산돼 gradient가 content_builder/pooling까지 연결된다 — `forward_batch`가 K/V를 캐싱하지 않고 매번 `handle.model(batch, return_extras=True)`를 다시 호출하는 기존 구조 그대로다).
- Stage2 lr = 기존 R1 설정(1e-4), Stage1 lr = 그 1/10(1e-5, `stage1_lr_ratio=0.1`, 기존 `base.yaml` 값 그대로).
- L2-SP: `Huber(y_final,y) + λ·mean((θ_stage1−θ_ref)²)`. **기존 코드의 기본 reduction은 `sum`이고 `lambda_sp=0.0`(사실상 미사용, untuned)이었다** — `README_STAGE2.md`와 `src/stage2/engine.l2sp_penalty`에서 확인. 이번 실험을 위해 `reduction` 인자를 추가해(`sum`/`mean` 선택 가능, 기존 호출부는 `sum`이 기본이라 하위 호환) λ=1.0, reduction=`mean`을 탐색적 시작값으로 썼다(요청대로). θ_reference는 각 조건 시작 시(= 매번 새로 로드한 Stage1 checkpoint) 복사해 고정했다.
- 초기화: 모든 조건이 **동일 시드로 동일하게 초기화된 fresh Stage2/tokenizer**에서 시작한다(과거 R1의 11-epoch 가중치를 재사용하지 않음). Stage2 residual head는 zero-init이라 epoch 0 예측은 Stage1과 완전히 같다.
- epoch 0을 평가해 best 후보에 포함했다(선택 기준은 기존과 동일하게 val subset, `min_delta=0`, strict `>`). best와 last(=조기종료 직전 epoch) 체크포인트를 모두 저장했다.
- 동일 split·batch 순서: 4개 조건 모두 train 4124 / val 881행, 동일 batch_size(32)이고, 매 조건 시작 시 동일 seed(42)로 재시드해 DataLoader shuffle 순서와 모델/토크나이저 초기화가 동일하도록 맞췄다.

### 구현 중 발견·수정한 버그

최초 구현에서는 `Setup`(캐시·구조테이블·Stage1 handle)을 4개 조건이 **공유**해서, B(공동학습)가 Stage1 가중치를 in-place로 바꾼 뒤 C가 그 "오염된" 가중치를 그대로 이어받는 문제가 있었다(1-epoch smoke test에서 C/D의 epoch-0 `val_stage1_subset`이 서로 달라지는 것으로 발견). **조건마다 Stage1 checkpoint를 새로 로드**하도록 고쳤다(`load_stage1_handle` 재호출, ESM·cache 재사용은 그대로 — 가벼움). 수정 후 4개 조건 모두 epoch 0의 `val_stage1_subset`이 0.7664로 동일함을 확인했다.

## 1. Best/last 요약 (`summary.csv`)

| 조건 | which | epoch | val subset | val RMSE | val MAE | missense | synonymous | indel | Stage1 가중치 변화(L2) |
|---|---|---|---|---|---|---|---|---|---|
| A frozen+real | best | 0 | 0.7664 | 3.723 | 2.289 | 0.6951 | 0.2211 | 0.8376 | — |
| A | last | 11 | 0.7493 | 3.854 | 2.331 | 0.6556 | 0.2484 | 0.8430 | — |
| B joint+real | best | 0 | 0.7664 | 3.723 | 2.289 | 0.6951 | 0.2211 | 0.8376 | 0.0000 |
| B | last | 11 | 0.7456 | 4.083 | 2.415 | 0.6511 | 0.2947 | 0.8400 | 0.2057 |
| C joint+fixed | best | 0 | 0.7664 | 3.723 | 2.289 | 0.6951 | 0.2211 | 0.8376 | 0.0000 |
| C | last | 11 | 0.7457 | 4.083 | 2.415 | 0.6514 | 0.2900 | 0.8400 | 0.2058 |
| D frozen+fixed | best | 1 | 0.7664 | 3.735 | 2.292 | 0.6952 | 0.2237 | 0.8376 | — |
| D | last | 11 | 0.7614 | 3.841 | 2.313 | 0.6801 | 0.2499 | 0.8426 | — |

- **네 조건 모두 best epoch는 0(A/B/C) 또는 1(D)이고, 값은 Stage1(0.7664)과 사실상 같다.** 어떤 조건도 Stage1 자체 baseline을 넘지 못했다 — 공동 fine-tuning도 예외가 아니었다. 이것이 이번 실험의 1차 결론이다.
- Stage1 가중치 변화(L2 norm)는 B=0.2057, C=0.2058로 **거의 동일**하다. 11 epoch 동안 153,058개 파라미터가 눈에 띄게(0.2 수준) 움직였지만, 실제 구조값을 썼는지 고정값을 썼는지는 그 움직임의 크기에 거의 영향을 주지 못했다.
- L2-SP penalty: λ=1.0, reduction=mean일 때 `train_l2sp_penalty` ≈ 2.8×10⁻⁷ (B/C epoch 11, Stage1 가중치 변화 L2=0.206 → mean squared diff = 0.206²/153058 ≈ 2.76e-7), task loss(≈1.06~1.40)에 비해 **7자리 이상 작다**. `train_stage1_grad_norm_mean`(클리핑 전, 153k개 파라미터 전체의 norm)은 epoch 11에 B=42.05, C=42.08로 꽤 크다. 즉 **이 λ=1·mean 설정에서는 L2-SP 앵커가 사실상 작동하지 않았다** — Stage1은 거의 전적으로 task loss(Huber)만으로 끌려갔다. 이는 기록해 두되, λ를 올려 재시도하는 것은 이번 범위 밖이다(요청대로 추가 탐색은 하지 않았다).

## 2. 학습 곡선

`learning_curves.png`(4개 조건의 val subset, epoch 0–11): **B(빨강)와 C(주황)가 11 epoch 내내 거의 완전히 겹친다.** D(초록)는 epoch 3 이후 계속 A(파랑)보다 위에 있다. 네 조건 모두 Stage1 선(점선) 아래로 빠르게 떨어진 뒤 부분적으로만 회복한다.

`by_type_curves.png`(RMSE, missense/synonymous/indel Spearman, 4개 조건): 같은 패턴이 모든 패널에서 반복된다 — B/C는 모든 지표에서 거의 포개지고, missense·RMSE에서는 D가 A보다 눈에 띄게, 그리고 **지속적으로**(한두 epoch가 아니라 2~11 전체 구간에서) 낫다. indel은 네 조건이 뒤섞여 뚜렷한 순서가 없고, synonymous는 B/C(공동 학습)가 A/D(frozen)보다 전 구간에서 높다.

## 3. 핵심 비교

### (a) epoch 11(last)에서의 position 단위 paired bootstrap (B=2000, val subset)

| 비교 | 관측 차이 | 95% CI | P(차이≤0) |
|---|---|---|---|
| B−A (공동 학습 효과) | −0.0037 | [−0.0348, 0.0263] | 0.59 (유의하지 않음) |
| B−C (공동 학습에서 구조정보 효과) | −0.0001 | [−0.0004, 0.0001] | 0.78 (유의하지 않음, 사실상 0) |
| A−D (frozen에서 구조정보 효과) | **−0.0121** | **[−0.0247, −0.0007]** | **0.982 (유의함 — A가 D보다 나쁨)** |
| (B−C)−(A−D) | **+0.0120** | **[0.0007, 0.0246]** | **0.98 (유의함 — 양의 방향)** |

(`contrasts_bootstrap.csv`. best epoch에서는 네 조건 모두 Stage1과 정확히 같아 네 비교 모두 차이가 0에 가깝고 CI가 0을 포함해 싣지 않는다 — `summary.csv`에 남아 있다.)

### (b) 궤적 전체 평균(epoch 1–11, 단일 숫자가 아니라 추세로도 같은 결론인지 확인)

| 지표 | A | B | C | D | B−A | B−C | A−D |
|---|---|---|---|---|---|---|---|
| val subset | 0.7460 | 0.7463 | 0.7463 | **0.7611** | +0.0003 | +0.0000 | **−0.0151** |
| val RMSE | 3.845 | **4.113** | 4.113 | 3.814 | **+0.268**(악화) | +0.000 | +0.031 |
| missense Spearman | 0.6476 | 0.6546 | 0.6546 | **0.6800** | +0.0070 | −0.0000 | **−0.0324** |
| synonymous Spearman | 0.2527 | **0.2828** | 0.2826 | 0.2416 | +0.0301 | +0.0002 | +0.0111 |
| indel Spearman | 0.8444 | 0.8380 | 0.8379 | 0.8423 | −0.0064 | +0.0000 | +0.0021 |

11개 epoch 전체 평균으로 봐도 endpoint 결과와 같은 결론이다: **B−C는 모든 지표에서 0에 수렴**하고, **A−D는 subset·RMSE·missense에서 뚜렷하게 D가 유리**하다(synonymous·indel은 거의 중립이거나 반대).

## 4. 해석

**(1) 개선은 없었다.** 네 조건 중 어느 것도 Stage1 자체(0.7664)를 넘지 못했다. 제한적 공동 fine-tuning(B)도 예외가 아니다 — "구조정보 덕분에 좋아졌다"고도 "서열 모델을 더 학습시켜서 좋아졌다"고도 말할 수 없다. 둘 다 좋아지지 않았다.

**(2) 그럼에도 비교는 뚜렷한 것을 보여준다.** B−C(공동 학습에서 실제 구조 대 고정 구조)가 **모든 지표·전 구간에서 사실상 0**이라는 것은, 공동 학습이 진행되는 동안 Stage2(R1)가 실제 구조값에서 끌어오는 유용한 정보가 (있어도) 측정 가능한 수준이 아니라는 뜻이다. 반면 A−D(frozen에서 실제 대 고정)는 last epoch에서 유의하게 음수이고(−0.012, p=0.982), 궤적 평균으로도 재현된다(subset −0.015, missense −0.032) — **frozen Stage1 위에서는 실제 구조값이 "안 쓰느니만 못한" 수준으로 해롭다.** 이 손해는 거의 전부 missense에서 나온다(synonymous·indel은 거의 중립이거나 반대 방향).

**(3) (B−C)−(A−D) > 0 (last: +0.012, 유의)이 말하는 것**: 공동 학습은 "구조정보를 유용하게 만들지" 못했지만(B≈C), frozen에서 구조정보가 끼치던 해(A<D)를 **지워버렸다**. 즉 공동 학습의 효과는 "구조를 더 잘 읽게 됐다"가 아니라, **Stage1 pooling/head가 다시 학습되면서 R1의 구조-조건부 보정이 더는 missense를 깎아먹지 않는 지점으로 전체 계(system)가 재조정됐다**는 쪽에 가깝다 — 공동 학습이 구조를 무의미하게 만들어서(B≈C) 그 해악도 같이 사라진 것이다.

**(4) 그런데 이 "재조정"은 공짜가 아니다.** RMSE 기준으로는 B−A가 +0.268(뚜렷하게 악화)이고, B/C의 RMSE(last 4.08, 궤적평균 4.11)는 A/D(3.8대)보다 전 구간에서 계속 나쁘다. **subset(순위 지표)은 공동 학습으로 "덜 나빠지지만", 수치 오차는 공동 학습으로 "더 나빠진다."** `analysis/stage2_diag2/REPORT_B.md`에서 본 수치-순위 분기(패턴 A/C 혼재)가 여기서도 반복된다 — 다만 이번엔 Stage1 자체가 재학습되면서 생기는 분기라는 점이 다르다.

**요약하면: 이번 결과가 보여주는 "개선"은 구조정보 활용 능력의 향상이 아니라, 공동 학습이 frozen 상태에서 구조정보가 끼치던 (주로 missense의) 해를 순위 지표 기준으로 상쇄한 것이고, 그 대가로 수치 오차(RMSE/MAE)는 더 나빠졌다.** 둘 다 Stage1 단독 baseline보다는 못하다.

## 5. 한계 (단일 seed·탐색적 결과)

- Stage2 seed 42, Stage1 checkpoint seed44 하나만 썼다. 위 bootstrap CI는 **고정된 모델 쌍 사이의 val 표본 불확실성**이지, 여러 Stage2 시드에 대한 재현성이 아니다.
- λ=1.0·mean reduction에서 L2-SP 앵커가 사실상 비활성이었다(§1) — 앵커가 실제로 작동하는 λ에서 B/C가 어떻게 달라지는지는 이번 범위에서 보지 않았다.
- position 단위 bootstrap은 val의 52~53개 position 블록에 의존한다 — 실패한 replicate는 없었지만(`n_failed=0`), 블록 수 자체가 적어 CI가 넓다(B−A의 경우 [−0.035, 0.026]).
- C/D의 "고정값"은 median/mode 하나이지 "구조정보 제거"의 유일한 방법은 아니다. 다른 null(예: 샘플별 랜덤 셔플)과 결과가 같은지는 확인하지 않았다(`analysis/stage2_diag/REPORT.md` §5의 position shuffle 실험과는 설계가 다르다 — 그쪽은 학습된 best 체크포인트에 대한 입력 교란이고, 이번은 학습 자체를 고정값으로 처음부터 다시 한 것이다).
- test split은 전혀 쓰지 않았다(요청대로).

## 산출물

- `runs/stage2_joint_ablation/seed42/{A,B,C,D}/`: `best_stage2.pt`, `last_stage2.pt`, `stage1_init.pt`(B/C만), `history.json/csv`, `config.json`
- `analysis/stage2_joint_ablation/summary.csv`, `contrasts_bootstrap.csv`, `learning_curves.png`, `by_type_curves.png`
- 스크립트: `train_stage2_r1_joint_ablation.py`(실행), `scripts/stage2_joint_ablation_summary.py`(집계·bootstrap)
- 재실행: `python train_stage2_r1_joint_ablation.py --out-root runs/stage2_joint_ablation/seed42 && python scripts/stage2_joint_ablation_summary.py`
- 관련 기존 보고서: `analysis/stage2_diag/REPORT.md`(R0/R1/R2, OOF), `analysis/stage2_diag2/REPORT_B.md`(수치오차 vs 순위, alpha blending)
