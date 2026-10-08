# Stage 2 진단 보고서 (seed 42, Stage 1 seed44 W10, validation 전용)

> **후속 보고서**: 수치 오차(Huber/MAE/RMSE) vs 순위 지표(Spearman)가 따로 움직이는지, 보정(delta_y) 크기를
> alpha로 줄이면 나아지는지는 `analysis/stage2_diag2/REPORT_B.md`에서 R1/R2(+R0 참고)로 추가로 다뤘다.
> R1 + 제한적 Stage1 공동 fine-tuning(실제 vs 고정 구조 ablation)은 `analysis/stage2_joint_ablation/REPORT_D.md`.
> 이 보고서의 내용은 그대로 두고 덮어쓰지 않았다.

작성일 2026-10-07. 이번 진단과 R0/R1 실험에서는 test split을 로드하거나 평가하지 않았다.
모든 결과는 단일 seed 기준의 탐색적 결과다. 기존 run 디렉터리(`runs/stage2/`, `runs/stage2_tau0.1/`)는 읽기만 했다.

## 0. 요약

| 질문 | 판정 | 근거 |
|---|---|---|
| ESM key가 한 샘플 안에서 서로 비슷한가 | **확인됨** | 잔기 key 간 cosine 평균 0.946(최솟값 0.91). layer embedding(norm 10.8)과 PE(5.7)가 content(1.1)보다 훨씬 크다. |
| 구조 query가 key 차이를 읽는가 | **못 읽음 (확인됨)** | query의 대부분이 type embedding이다(norm 11.3, 구조 성분 2.4–2.6). nine-query끼리 cosine 0.75. |
| τ 구현 오류 | **반박됨** | 수동 softmax(cos/τ)와 구현 결과의 차이 ≤5e-7. τ를 바꾸면 entropy가 예상대로 변한다. |
| gradient 차단 | **반박됨** | 모든 모듈이 업데이트된다. 첫 ~10 step은 zero-init 때문에 상류 gradient가 ~1e-5이고, step 100 전후로 커진다. |
| 학습시간이 짧은 이유 | **실행 오류 아님** | epoch당 129 step, 약 18초(그중 데이터 로딩 13.4초). best epoch 1 + patience 10으로 11 epoch에서 종료된다. |
| train 전용 residual 패턴에 과적합 | **지지되지만 확정되지 않음** | Stage 1 residual 평균이 train +0.27, val −0.75. 네 가지 구조 모두 train loss↓, val↓. |
| 역방향(R1)이 해결책인가 | **이번 결과로는 아님** | R0와 R1 모두 best epoch 1, val subset 0.7663으로 forward 모델과 같다. |

핵심: attention 선택성이 낮은 것은 사실이다. 그러나 **현재 성능 정체의 주된 원인으로는 보이지 않는다.**
attention 경로를 uniform 평균으로 바꿔도 예측은 거의 그대로였고(아래 §3), attention 구조를 바꾼 네 모델이 모두 같은 궤적을 보였다.
가장 유력한 공통 원인은 train 행의 Stage 1 예측이 in-sample이어서 residual target 분포가 val과 다르다는 점이다. 이는 아직 가설이다(§5).

이전 대화에서 제가 "128차원에서 query와 key가 거의 직교해서 cosine이 0 근처에 몰린다"고 추정했는데, 측정해 보니 틀렸다.
실제로는 key끼리 **너무 비슷하고**(cos 0.95), query 방향은 구조가 아니라 type embedding이 정한다.

## 1. 확인한 설정

- Stage 1: `runs/stage1_v2/unified_reference_delta/W10/shipped_split/seed44/best.pt` (sha256 앞 16자리 `e968769e37a11148`).
  W10, layers=[33], init_seed 1234, 학습된 Stage 1 τ = 0.662.
- Stage 2 학습 seed 42는 `train_stage2.py`의 기본값이다. Stage 1 checkpoint seed44는 Stage 1 validation 최고값으로 고른 것이다.
  둘이 다른 것은 의도된 조합이며, 이번 비교는 모두 같은 checkpoint를 썼다.
- 데이터: manifest sha `4e6129d9ba8f7c80`, 구조 테이블 sha `92fafdcfce71a5f7`. 구조 행은 var_id로 join했고, split이 manifest와 일치하는지 검사했다.
- 표본 수: train 4124 (missense 3195 / synonymous 675 / indel 254), val 881 (671 / 158 / 52).
- 기존 4개 run 모두 config에서 query_mode, seed, Stage 1 경로를 확인했다. τ_init은 null(=0.69) 또는 0.1이다. checkpoint 해시는 4개 모두 서로 다르다(공유나 재사용 없음).
- 학습 파라미터: Stage 2 21,346 + tokenizer 1,472 = 22,818. Stage 1은 requires_grad가 0개이고 eval mode다. 학습 전후 해시가 같다(`cc332b8a015c853a`).
- optimizer는 AdamW 한 그룹(Stage 2 + tokenizer), grad accumulation 없음, drop_last=False, num_workers=0.
- 선택 기준: val subset(missense와 indel Spearman의 평균), min_delta 0, patience 10. **기록된 epoch 1은 1 epoch 학습 *후* 값이다.**
  이번에 epoch 0(학습 전)을 따로 측정했다: val subset 0.7664 = Stage 1.
- git: 작업 트리가 dirty다(model.py의 tau_init, structure/schema 변경, 이번 진단 파일들). 기존 Stage 2 run의 config에는 git commit이 기록되어 있지 않다.

### Attention 구조 (실측)
- K, V: [B, 22, 128]. 22 = 잔기 슬롯 21개(W10, layer 1개이므로 Lyr·A = 1×21) + edit metadata token 1개(마지막 위치).
  indel이나 서열 끝 근처 변이는 슬롯 수가 더 적다(N_valid 최솟값 12). padding은 mask로 처리하고 softmax에서 제외된다.
- K = content + PE + layer_emb, V = content(잔기). metadata token은 K와 V 모두 metadata encoder의 출력이다.
- attention weight shape: single [B,1,22], nine [B,9,22]. softmax 축은 N이다(합 −1 오차 ≤1.2e-7).
- entropy는 dropout 없는 실제 softmax 확률로 계산했다(이 attention에는 dropout이 없다).

## 2. 진단 A: key와 value의 다양성 (val 진단 표본, 샘플 내부 기준)

진단 표본은 type별 최대 150개로, 위치가 고르게 퍼지도록 골랐다. ID는 `diag_sample_ids.csv`에 있다.

| 지표 (샘플 내부) | 평균 | p05–p95 |
|---|---|---|
| 잔기 key 간 cosine (off-diagonal) | 0.946 | 0.942–0.948 |
| 같은 지표, content만 | 0.692 | 0.50–0.90 |
| key centered variance ratio (토큰 간 변동 에너지 비율) | 0.098 | 0.095–0.103 |
| value centered variance ratio | 0.460 | 0.28–0.67 |
| norm: layer_emb / PE / content | 10.8 / 5.66 / 1.10 | |
| Stage 1 자체 attention의 H/log N | 0.993 | 0.990–0.997 |

- key의 90%가 모든 토큰에 공통인 성분이다. layer가 하나뿐이라 layer_emb는 모든 토큰에 같은 상수로 더해지고, PE의 cosine도 0.74다.
- 한편 value(content)는 토큰마다 꽤 다르다(분산 비율 0.46). 즉 "읽을 만한 정보는 V에 있지만, K로는 고를 수 없는" 상태다.
- Stage 1 자신도 거의 균등하게 pooling하고 있었다(H/log N 0.993). Stage 1은 사실상 평균 pooling 모델이다.
- metadata token의 key는 잔기 key와 거의 직교한다(cos ≈ 0). 잔기끼리는 cos 차이가 작으므로, query별 cos 범위(평균 0.13)는 대부분 metadata 대 잔기 차이에서 나올 것으로 추정한다. 토큰별 분해는 측정하지 않았다.
  heatmap: `key_cosine_heatmaps.png`.

## 3. 진단 B: query, logit, τ

| 지표 (val, τ0.1 학습 모델, best=epoch 1) | single | nine |
|---|---|---|
| ‖e_type‖ / ‖mean U‖ | 11.3 / 2.6 | 11.3 / 2.6 |
| 9개 query 간 cosine | – | 0.755 |
| 구조 토큰 S 간 cosine | −0.009 | −0.010 |
| query별 cos(q,K) 범위(max−min) | 0.133 | 0.130 |
| logit 범위 | 1.31 | 1.29 |
| H/log N, max p | 0.986, 0.080 | 0.985, 0.083 |
| ‖z_attn − z_uniform‖ / ‖z_uniform‖ | 0.117 | 0.112 |
| \|δ_attn − δ_uniform\| (δ 크기 ~0.07) | ~5e-4 | ~5e-4 |

- 구조 토큰 자체는 서로 다르다(S cosine ≈ 0). 하지만 adapter 출력의 평균에 norm 11의 type embedding을 더하면서 query 방향이 type으로 정해진다.
  `nn.Embedding` 기본 초기화가 N(0,1)이라 128차원에서 norm이 √128 ≈ 11이 된다.
- single과 nine의 내부 출력은 **동일하지 않다**: z 차이 0.012 (‖z‖ 0.87). 그러나 예측 차이는 평균 1.5e-4로, 성능 지표가 소수점 4자리까지 같아지는 수준이다.
- **τ 대조 실험**(같은 checkpoint, 같은 batch, eval mode, τ만 교체):

  | τ | 0.69 | 0.1 | 0.03 | 0.01 |
  |---|---|---|---|---|
  | H/log N (τ0.1_nine) | 1.000 | 0.985 | 0.828 | 0.457 |
  | max p | 0.050 | 0.083 | 0.247 | 0.528 |
  | δ 평균 | 0.0094 | 0.0089 | 0.0071 | 0.0050 |

  τ는 의도대로 logit에 반영되고, scaling이 중복으로 적용되지도 않는다. 저장된 cfg의 tau_init(0.1)과 실제 τ(0.1009)도 일치한다.
  τ를 0.01까지 낮추면 attention은 날카로워지지만 δ는 거의 변하지 않는다.
  learnable scale s와 1/τ는 같은 함수족이므로, scale을 따로 도입하는 것은 해결책이 아니다.
- "entropy 3.04가 같았던 이유": τ0.1 run들의 기록값(3.04)은 val 샘플별 raw entropy(nats)의 평균이다. N_valid가 샘플마다 달라서 log 22보다 조금 낮게 나온다.
  τ0.69 run은 3.08로 서로 다르다(이전 대화에서 제가 이 값을 3.04로 잘못 적었다).
  single과 nine이 같은 값을 보인 것은 위에서 본 것처럼 두 attention 분포가 거의 같기 때문이다. 반올림이나 구현 문제는 아니다.

## 4. 진단 C: gradient, 업데이트, 보정 경로 (계측 재현 run, 11 epoch)

`instrument_stage2.py`로 기존 설정(τ_init 0.1)을 그대로 재현했다. val subset 궤적이 기록과 epoch 1에서 일치한다(0.7663).
이후 epoch에서는 0.0002–0.006 범위에서 차이가 나는데, GPU 비결정성으로 추정한다(확인하지는 않았다).

- gradient norm(clip 전, single 기준):
  - step 1–5에서는 head 마지막 층만 0.01–0.1이고 나머지는 0. FiLM과 head의 마지막 층이 zero-init이라서 그렇다.
  - step 10에서는 상류가 1e-5 수준, step 100에서 adapter 6.5e-4, FiLM 7e-3이다.
  - step 500에서는 tokenizer 0.27, adapter 0.14, FiLM 0.76이다. **차단되지 않고 지연될 뿐이다.**
- clipping(max_norm 1.0)은 1419 step 중 399 step(28%)에서 일어났고, 처음 일어난 것은 step 214다.
- 초기값 대비 변화(epoch 11): tokenizer 1.17, adapter 1.89, type_emb 0.33, FiLM-out 3.17, head-in 1.76, **log_tau 0.047**.
  τ는 0.100에서 0.1046으로 움직였다. gradient가 계속 ~1e-4 수준이라 τ는 사실상 학습되지 않는다.
- 학습 경과(val): δ RMS는 0.08(ep1)에서 0.76(ep11)으로 커진다. 그동안 val Huber는 1.865에서 1.956으로 나빠지고, train Huber는 1.408에서 1.332로 좋아진다.
- attention이 예측에 미치는 영향 \|δ_attn − δ_uniform\|는 ep1 0.0007, ep11 0.025다(δ RMS 0.76의 약 3%).
  즉 학습 후반에도 δ는 대부분 FiLM(c)의 global 경로에서 나온다.
- **epoch 1의 δ는 거의 type별 상수다**: missense +0.076, synonymous −0.065, indel +0.031, type 내 표준편차 ≤0.011.
  Stage 1 train residual의 type별 평균(missense +0.42, synonymous −0.44)과 부호가 같다.
  val residual 평균은 missense −0.79, synonymous −0.59로 반대 방향이다.
- δ와 Stage 1 residual의 상관: train 0.133(missense 0.235), val −0.020.
- 구조 position-level shuffle(9개 feature를 위치 단위로 다른 위치와 교환, 5회): val subset 변화 ≤0.0002, 예측 변화 평균 0.002.
  best checkpoint는 구조 입력에 거의 반응하지 않는다(입력 교란에 대한 민감도이며, 인과효과가 아니다).

### Residual 선형 probe (`residual_probe.json`)
Ridge 회귀, 입력은 구조 9개 feature + type.
- train residual로 학습하고 val로 평가: Pearson −0.09, MSE 15.0. 0을 예측할 때의 MSE 13.9보다 나쁘다.
  **train residual 패턴은 val로 옮겨지지 않는다.**
- val 내부 5-fold cross-fit(진단용, 모델 선택에 쓰지 않음): Pearson 0.15, missense 내부 Spearman 0.14. type만 쓰면 −0.15다.
  → 구조에는 out-of-sample residual 신호가 **약하게** 있다. 다만 n=881이라 불확실성이 크다.

## 5. 학습시간

- ESM은 실행되지 않는다. `raw_cache.pt`(11 GB, mmap)의 hidden state를 읽고, Stage 1은 frozen 상태로 매 batch forward만 한다.
- 장치: NVIDIA TITAN Xp (CUDA). peak GPU 메모리 122 MB.
- epoch당: train 129 batch(bs 32, 4124행, gradient accumulation 1)에 약 15.2초가 걸린다.
  - 데이터 로딩(윈도우 구성) 13.4초
  - Stage 1 forward 0.63초
  - Stage 2 forward/backward/step 1.13초
  - val 평가 3.0초
  - 합계 약 18초/epoch, step당 약 0.12초.
- 11 epoch × 129 = 1419 optimizer step, 약 3.5분이 걸린다. 여기에 cache 로드와 test 1회 평가가 더해진다.
- 결론: 짧은 시간은 (1) Stage 2가 2.3만 파라미터로 가볍고 (2) ESM과 Stage 1이 캐시되거나 frozen이며 (3) best epoch 1 + patience 10으로 조기 종료되기 때문이다.
  subset, batch limit, debug, resume 같은 실행 축소 요인은 없었다.
  train은 계속 좋아지고 val은 나빠지므로 epoch를 늘리는 것은 해결책이 아니다. 그래서 장기 run은 하지 않았다.

## 6. 가설 판정

| 가설 | 판정 | 근거 수치 |
|---|---|---|
| ESM key가 샘플 안에서 비슷해서 선택성이 낮다 | 확인됨 | §2: 잔기 key cos 0.946, 변동 에너지 9.8%, 공통 성분 layer_emb·PE |
| 구조 query가 key 차이를 못 읽는다 | 확인됨 | §3: query를 type embedding이 지배, nine-query 간 cos 0.75 |
| τ 적용 버그 / scaling 중복 | 반박됨 | 수동 계산과 차이 ≤5e-7, τ 대조에서 entropy가 단조적으로 변함 |
| mask/padding 버그 | 반박됨 | padding weight가 정확히 0, softmax 합 1, metadata token 위치 확인 |
| gradient 차단 / Stage 1 변형 | 반박됨 | 모든 모듈 업데이트, Stage 1 해시 동일·eval 유지 |
| τ가 학습되지 않는다 | 확인됨(관측) | 11 epoch 동안 log_tau 변화 0.047 |
| attention이 예측에 기여하지 않는다 | 확인됨 (best), 대체로 확인 (last) | \|δ_attn−δ_unif\| 6e-4 (ep1), 0.025 (ep11, δ RMS의 3%) |
| 실행 문제로 학습시간이 짧다 | 반박됨 | §5 |
| train residual(in-sample Stage 1)에 과적합하여 val이 나빠진다 | 지지되지만 확정되지 않음 | train/val residual 평균의 부호 반대, Stage 1 Huber train 1.41 / val 1.87, 4개 구조 모두 같은 궤적, probe train→val 실패 |
| 구조 정보가 residual 신호를 갖지 않는다 | 근거 부족 (약하게 반대) | val cross-fit r=0.15 |
| "데이터 부족", "고차원 직교성" | 결론 내리지 않음 | 직교성 가설은 측정 결과와 맞지 않음 |

## 7. 수정 사항

- 버그 수정은 없다. 구현, 로딩, mask, gradient 문제는 발견되지 않았다.
- 기존 코드에 대한 변경은 모두 기본 동작을 바꾸지 않는 추가 사항이다.
  - `src/stage2/model.py`: `tau_init` 옵션(이전 세션). `configs/stage2/base.yaml`은 `tau_init: 0.1`.
  - `src/stage2/engine.py`: `train_stage2(..., evaluate_test=True)` 인자 추가. 기본값은 기존과 같고, R0/R1에서만 False로 쓴다.
- 새 파일:
  - `src/stage2/diagnostics.py`
  - `diagnose_stage2.py`
  - `instrument_stage2.py`
  - `scripts/stage2_residual_probe.py`
  - `scripts/stage2_diag_summary.py`
  - `src/stage2/reverse.py`
  - `train_stage2_reverse.py`
  - `tests/test_stage2_reverse.py`
- 테스트: `tests/test_stage2_smoke.py` 13개 + `tests/test_stage2_reverse.py` 5개, 모두 통과.

## 8. 역방향 모델 R0/R1

**진행 근거**: 구현과 학습 경로는 정상이었다(§4). 기존 fusion에는 구체적인 병목이 확인됐다.
ESM key는 선택할 차이가 거의 없고(§2), 반대로 구조 토큰은 서로 직교에 가깝고 content(value)는 다양하다.
그래서 "서열이 구조를 읽는" 방향을 시험할 근거가 있다고 판단했다.

**구현** (`src/stage2/reverse.py`):
- h_i는 Stage 1 content token(= 잔기 V, PE·layer_emb 이전 값)이다. metadata token은 FiLM과 pooling에서 **두 모델 모두 제외**했다.
- R0: c_i = mean_j W_V s_j.
- R1: q_i = W_Q h_i, k_j = W_K s_j(d_a=32), softmax_j(q·k/√32)는 9개 구조 토큰에 대해 계산하고, c_i = Σ a_ij W_V s_j다.
  cosine이 아닌 scaled dot-product만 쓴다.
- 공통: [γ_i, β_i] = g([c_i; e_type]) (256→32→256, 마지막 층 zero-init). h_mod_i = (1+γ_i)h_i + β_i.
  z = 유효 잔기 masked mean, δ = Head(z) (128→32→1, zero-init), y_final = y1 + δ.
- 파라미터: R0 25,441, R1 30,625. 차이 5,184 = W_Q(128×32+32 = 4,128) + W_K(32×32+32 = 1,056). dummy 파라미터는 없다. tokenizer는 1,472로 같다.
- 동일 조건: seed 42, Stage 1 seed44 W10, base.yaml 설정, val 선택, test 미평가, epoch 0 기록.

**결과** (`model_comparison_val.csv`, `training_curves.png`):

| 모델 | Stage 2 params | best epoch | val subset (best) | Stage 1 | val subset (ep11) |
|---|---|---|---|---|---|
| forward single (τ0.1) | 21,346 | 1 | 0.7663 | 0.7664 | 0.7427 |
| forward nine (τ0.1) | 21,346 | 1 | 0.7663 | 0.7664 | 0.7427 |
| R0 구조 평균 → token FiLM | 25,441 | 1 | 0.7663 | 0.7664 | 0.7403 |
| R1 서열 Q → 구조 K/V → token FiLM | 30,625 | 1 | 0.7663 | 0.7664 | 0.7493 |

- 네 모델 모두 epoch 1이 선택됐고, 그 값은 Stage 1과 사실상 같다. 이후 val은 내려가고 train loss는 내려간다(R1이 가장 많이 내려간다: 1.30).
- R1 attention entropy: ep1 2.194 (log 9 = 2.197의 99.9%, 거의 균등), ep11 1.969 (90%).
- 초기값에서 best checkpoint까지 모든 파라미터 텐서가 움직였다(W_Q 0.37, W_K 0.20). gradient는 정상적으로 흐른다.
- ep11에서 R1이 R0보다 높은 것(0.749 대 0.740)은 선택되지 않은 epoch이고 단일 seed라서 의미를 두지 않는다.
- **결론**: 방향을 바꿔도 이번 설정의 정체는 풀리지 않았다. 이 결과는 병목이 attention 방향보다 상류(residual target)에 있다는 가설과 맞는다.
  R1과 forward 모델의 차이를 순수한 방향 효과로 해석할 수 없다는 점은 처음부터 전제했다.

## 8.1 R2: R1을 강화한 multi-head reverse attention (2026-10-07 추가, 단일 seed, val only)

§9(아래)에서 제안한 OOF 실험은 이미 실행했다(§8.2). 이번은 그와 별개로, "attention 용량·날카로움이 부족해서 구조 신호를 못 읽는 게 아닐까"라는 남은 의심을 직접 검증하기 위해 R1을 강화했다.

**구현** (`src/stage2/reverse.py`, `ReverseMultiHeadFiLMModel`, mode `r2_multihead_query`):
- R1 대비 두 가지를 키웠다: (1) attention head를 1개(d_a=32)에서 4개(d_head=16, 총 64)로 늘리고 head마다 독립된 Q/K/V projection을 쓴 뒤 출력을 W_O(64→128)로 섞는다(표준 multi-head cross-attention). (2) 고정된 1/√d_head scaling 대신, Stage 1/forward와 같은 학습 가능한 τ = softplus(log_tau)+1e-4를 logit에 추가로 나눈다.
- 나머지(FiLM, head, type embedding, zero-init 마지막 층, metadata token 제외)는 R0/R1과 동일하다. W_O는 zero-init하지 않았다 — film/head의 마지막 층이 이미 초기 δ=0을 보장하므로, W_O까지 0으로 두면 불필요한 gating이 하나 더 생긴다.
- 파라미터: R2 42,018 (W_Q 8,256 + W_K 2,112 + W_V 2,112 + W_O 8,320 + log_tau 1 + 나머지는 R1과 동일). R1(30,625)보다 11,393개 많다.
- 조건은 R0/R1과 같다: seed 42, Stage 1 seed44 W10, 기존 target(OOF 아님), val 선택, test 미평가.

**결과**:

| 모델 | heads × d_head | params | best epoch | val subset (best) | entropy ep1 → ep11 (log 9 = 2.197) |
|---|---|---|---|---|---|
| R1 | 1 × 32 | 30,625 | 1 | 0.7663 | 2.194 → 1.969 (90%) |
| R2 | 4 × 16 | 42,018 | 1 | 0.7662 | 1.895 → 1.297 (**59%**) |

(`r2_comparison.csv`, `r1_vs_r2_entropy_and_val.png`)

- **용량과 날카로움은 실제로 커졌다.** R2는 11 epoch 안에 entropy를 log9의 59%까지 낮췄다. R1은 같은 기간에 90%에서 멈췄다. gradient는 모든 모듈(W_Q 0.68, W_K 0.33, W_V 0.36, W_O 0.72, log_tau 0.016)에 정상적으로 흘렀다.
- **그런데도 val 패턴은 바뀌지 않았다.** best epoch는 여전히 1이고 그 값(0.7662)은 Stage 1(0.7664)과 같다. 이후 val은 0.736–0.756 사이를 오가며, 어떤 epoch도 Stage 1을 넘지 못했다(R1의 ep9 0.7556, R2의 ep9 0.7556 — 둘 다 선택되지 않은 epoch).
- attention이 더 날카로워질수록 δ도 커졌다(delta_abs_mean ep1 0.085 → ep11 0.618, R1보다 크다). 즉 모델은 분명히 "더 적극적으로" 구조를 읽도록 바뀌었지만, 그 결과가 val에 도움이 되지는 않았다.
- **해석**: attention 용량이나 sharpness 부족은 정체의 원인이 아니었다. R1에서 "거의 균등해서 신호를 못 읽는 것 아니냐"는 가설은 R2로 반박됐다 — 훨씬 날카롭게 만들어도 같은 지점(Stage 1 수준, epoch 1)에서 멈춘다.
  이는 §8.2(OOF)의 결론과 같은 방향을 가리킨다: 병목은 attention 설계가 아니라 train 잔차 target 쪽에 더 가깝다. 다만 이번도 단일 seed이고, OOF 실험 자체에 fold 모델이 약하다는 교란이 있어 어느 쪽도 확정은 아니다.

## 8.2 OOF 잔차 target 실험 (요약, 전체 결과는 `analysis/stage2_oof/`)

같은 세션에서 §9의 1번 제안을 먼저 실행했다. 5-fold로 Stage 1을 다시 학습해 train 행마다 out-of-fold y1을 만들고, forward single/nine과 R0/R1을 그 위에서 다시 학습했다(여전히 val은 기존 Stage 1, test 미사용).

- train 잔차 평균이 +0.27(in-sample) → −0.64(OOF)로 바뀌어 val(−0.75)과 부호가 같아졌다. epoch가 지나도 val이 떨어지던 현상이 사라지고, 32 epoch까지 Stage 1 근처에서 유지됐다.
- 하지만 Stage 1 대비 이득(+0.0007 ~ +0.0019)은 위치 단위 paired bootstrap(B=2000)에서 95% CI가 모두 0을 포함했다(P(이득≤0) 37–46%).
- **교란 요인**: OOF fold 모델이 shipped 모델보다 약했다(val subset 0.685–0.743 대 0.766, 학습 데이터 80%·조기 조기종료). 그래서 "잔차 target이 유일한 원인"이라고 확정할 수 없다 — fold 모델이 더 강했다면 결과가 달라질 수 있다.

종합하면, 지금까지의 증거(R0/R1/R2 모두 같은 지점에서 멈춤 + OOF로 하락은 없어지지만 유의한 이득도 없음)는 "attention 설계보다 잔차 target/구조 신호 자체의 한계가 더 크다"는 가설 쪽에 무게를 싣지만, 두 실험 모두 단일 seed와 각자의 교란 요인이 있어 확정은 아니다.

## 9. 다음에 할 가치가 있는 실험 (일부는 이미 실행됨 — 위 §8.1/§8.2 참고)

1. ~~Out-of-fold Stage 1 residual target.~~ **실행함 (§8.2).** 하락은 사라졌지만 유의한 이득은 없었고, fold 모델이 약하다는 교란이 남았다.
2. ~~R1 강화(multi-head + 학습 가능한 τ).~~ **실행함 (§8.1, R2).** attention을 훨씬 날카롭게 만들어도 같은 지점에서 멈췄다.
3. **더 강한 fold 모델로 OOF 재실행.** §8.2의 fold 모델(val 0.685–0.743)을 shipped 수준(0.766)에 가깝게 올린 뒤(예산·patience 확대, fold 수 조정) 다시 비교한다. 지금 결과의 가장 큰 교란 요인을 없애는 실험이다.
4. **seed 확장.** 1·2·3 중 가장 유망한 조합(현재는 forward+OOF, 이득 +0.0019)을 3–5 seed로 재현해 bootstrap CI(현재 ±0.015)보다 작은 효과를 구분한다.
5. (3·4가 긍정적일 때) **attention 입력 정리 후 forward 재비교.**
   Stage 2 attention에서 샘플별로 key를 centering(공통 layer_emb·PE 성분 제거)하고, type embedding의 scale을 구조 성분 수준으로 맞춘다.
   그 다음 single 대 nine을 다시 비교한다. 지금은 target 문제가 지배적이라 이 변경만 단독으로 시험하면 효과를 구분할 수 없다.

## 산출물

- 진단 수치: `analysis/stage2_diag/`
  - `audit.json`
  - `keys_*.csv`
  - `queries_*.csv`
  - `tau_override_summary.csv`
  - `single_vs_nine_*.csv`
  - `residual_per_sample.csv`
  - `report_numbers.json`
  - `residual_probe.json`
  - `structure_shuffle_val.csv`
  - `model_comparison_val.csv`
  - `diag_sample_ids.csv`
- 그림: `key_cosine_heatmaps.png`, `cos_range_and_entropy.png`, `tau_override_entropy.png`, `training_curves.png`
- 계측 run: `runs/stage2_diag/instrumented/{single,nine}_query/` (steps.csv, epochs.csv, summary.json, last_state.pt)
- R0/R1: `runs/stage2_reverse/{r0_struct_mean,r1_seq_query}/seed42/`
- 재실행 순서:
  1. `python diagnose_stage2.py`
  2. `python instrument_stage2.py --query-mode {single_query|nine_query} --out ...`
  3. `python train_stage2_reverse.py --mode {r0_struct_mean|r1_seq_query} --out ...`
  4. `python scripts/stage2_residual_probe.py`
  5. `python scripts/stage2_diag_summary.py`

### 측정하지 못한 항목
- GPU utilization(%)은 기록하지 않았다. peak memory만 기록했다.
- 계측 재현 run에서 epoch 2 이후 생긴 작은 차이(≤0.006)의 원인이 GPU 비결정성인지는 확인하지 않았다.
- 진단 A의 token kind별 분해는 잔기와 metadata 두 종류만 했다. layer는 하나뿐이라 layer별 분해는 해당 사항이 없다.
