# 구조 토크나이저 검증 실험

담당: 조승원. 배포용 토크나이저(`../tokenizer.py`)를 고르고 검증한 실험이다. 팀원이 알면 좋은 것만 남겼다.
(pooling 비교처럼 9/30 설계 변경으로 필요 없어진 실험과 코드는 뺐다.)

## 결론 먼저

1. **인코딩 방식은 결과를 가르지 않는다.** 5가지(L / Q / Qk / T / PLR)가 실용적으로 같아서, 해석이 쉬운 **Qk**를 쓴다.
2. **구조 9개 값에는 정보가 있다.** 구조만으로 subset 0.50, 위치와 구조의 대응을 섞으면 0 근처. 최선의 모델(GBDT)은 0.54.
3. **Stage 1과 합칠 때는 "어떻게 합치느냐"가 갈랐다.** 동결 Stage 1 뒤에 붙이면 효과가 없었고(E7),
   처음부터 같이 학습하면 구조 정보가 쓰였다(E8, 섞은 구조 대비 +0.029). 다만 "구조 없음" 대비 순이득은 아직 확인되지 않았다.

subset = (missense Spearman + indel Spearman) ÷ 2. 차이의 95% CI 는 위치 단위 bootstrap.
b = 구조 없음, c = 구조, d = c 와 같은 모델에 위치↔구조 대응을 섞은 구조 (c − d = 구조 "정보"의 효과).

## 실험별 요약

### E1 · E3 — 구조만으로 어디까지 / 인코딩 5가지 비교 ([보고서](reports/REPORT_E1_E3.md))

위치 기준 교차검증(5조각 × 2반복, 안쪽 4조각으로 하이퍼파라미터 탐색 — 인코딩 후보 50회, GBDT 100회), 고른 설정으로 seed 5개.
probe = 변이유형별 query 3개가 토큰 9개를 cross-attention 으로 읽음 → MLP head (ESM 없이 구조만).

| 모델 | subset [95% CI] | 비고 |
|---|---|---|
| GBDT (원래 9개 값) | 0.539 [0.488, 0.571] | 천장 |
| Qk (현재) | 0.501 [0.448, 0.540] | 기준 |
| Q / PLR / T / L | 0.500 / 0.497 / 0.495 / 0.494 | Qk 대비 −0.007 ~ −0.002, 모두 CI 가 ±0.04 안 → 실용적 동등 |
| Qk + 섞은 구조 | −0.032 | 구조 정보가 실제로 쓰인다는 대조군 |
| 변이유형만 | 0.000 | 바닥 |

- 인코딩 후보: L = 분위수-정규 변환 후 선형, Q = PLE(분위수 경계), **Qk = PLE(분위수 + pLDDT 공식 경계 50/70/90)**,
  T = PLE(결정나무가 정답을 보고 경계), PLR = 주기 함수.
- 고를 때 점수 − 실제 점수(낙관 폭)는 T 가 0.037 로 가장 컸다(정답을 보고 경계를 정함, 나머지 0.023–0.029) → T 는 권하지 않음.
- 천장과의 차이 0.04 는 주로 indel 쪽. 인코딩을 바꿔 더 짜낼 여유는 크지 않다.

### E2 — PLE 구현 검증 (`tests/test_encoders.py`)

- 원저자 공식 구현 `rtdl_num_embeddings`와 경계·인코딩 값이 1e-6 이하로 같다.
- 합성 데이터(pLDDT 70 에서 꺾이는 반응) 복원 R²: 순수 선형 0.683, **선형 토큰 + LayerNorm 0.997**, PLE 1.000.
  LayerNorm 덕분에 L 도 꺾임을 거의 잡는다 → E3 에서 L 이 PLE 만큼 나온 이유.

### E7 — 학습된 Stage 1(동결) 뒤에 구조 붙이기 ([보고서](reports/REPORT_E7.md))

호준 Stage 1(C·창 10, seed 44 `best.pt`)을 동결하고 Stage 2 가 구조를 읽어 고침값을 더한다 = 팀 Stage 2 1단계의 최소판.
배포 분할(train 학습 / val 선택 / test 한 번), 고침값 마지막 층 0 초기화, Huber.

| 비교 | seed 44 | Stage 1 seed 5개 합침 |
|---|---|---|
| c − d (구조 정보 효과) | +0.001 [−0.006, +0.008] | +0.005 [−0.003, +0.013] |
| c − b (실제 이득) | +0.003 [−0.006, +0.012] | −0.001 [−0.009, +0.007] |

- 이득 없음. 동결 Stage 1 이 train 을 일부 외워(남은 오차 SD train 2.8 vs test 4.4) Stage 2 가 배울 게 적다.
  팀 Stage 2 의 `frozen_stage1` 모드도 같은 조건이다.
- 호준 HF 체크포인트에는 변이별 예측이 없어서, 호준 코드 + 민선 ESM 캐시로 다시 계산했다(val 이 기록과 ≤5e-6 로 일치).

### E8 — 구조를 Stage 1 과 처음부터 같이 학습 ([보고서](reports/REPORT_E8.md))

호준 Stage 1 코드·설정 그대로, Stage 1 요약 z 가 query 로 구조 토큰 9개를 읽어 z 에 더한 뒤 Stage 1 head 로. seed 42–46.

| 팔 | test subset | missense |
|---|---|---|
| c: 구조 같이 학습 | 0.683 | 0.598 |
| b: 구조 없음 (다시 학습) | 0.675 | 0.571 |
| d: 섞은 구조 | 0.654 | 0.554 |

- **c − d = +0.029 [+0.002, +0.056]** (missense +0.045) → 같이 학습하면 구조 정보가 쓰인다.
- c − b = +0.008 [−0.036, +0.051] → 순이득은 학습 잡음(±0.02)에 묻혀 미확인.
- 주의: c 는 val 에서 b 보다 낮고(0.724 vs 0.760) test 에서 높다. 구조 값이 위치마다 거의 고유해 위치를 외울 위험이 있어,
  val 하나로 모델을 고르면 구조를 버리게 된다.

### 뺀 실험 (결과만)

- E6 (Stage 1 이 없을 때 흉내 낸 ESM 인코더 + 구조): c − d +0.022 유의. E6b (민선 분할·어댑터): 재현 안 됨.
  E7·E8 이 실제 Stage 1 로 같은 질문에 답하므로 코드는 뺐다.
- pooling 비교(mean / flatten / attention), 선형 head 비교: 9/30 설계 변경으로 필요 없어짐. 결과는 attention 판과 같은 결론.

## 실행

```bash
cd structure_tokenizer
python build_dataset.py                       # ../data/v2_dataset.csv
cd experiments
pip install optuna scikit-learn scipy rtdl_num_embeddings==0.0.12
python -m pytest tests -q                     # 38개 (E7 출력이 없으면 1개 건너뜀)

bash run_e1_e3.sh                             # E1·E3 → results/REPORT_E1_E3.md (CPU, 수 시간)

# E7·E8: 외부 입력이 필요 (경로는 config.py, 환경변수로 변경)
#   STAGE1_HF : hf download DeepCANCERsite/stage1-checkpoints --local-dir experiments/external/stage1_hf
#   FROZEN_PT : 브랜치 minseon/esm-module 의 handoffs/module_a_hr_v1/payload/frozen.pt (git lfs pull)
python e7_stage1_reconstruct.py && bash run_e7_all.sh && python e7_stage2.py --analyze   # Mac CPU 약 50분
python e8_joint.py --pack && python e8_joint.py --check && bash run_e8_all.sh            # 약 10분
```

## 파일

| 파일 | 역할 |
|---|---|
| `config.py` | 스키마·프로토콜·탐색공간·경로 (각 값의 근거를 주석으로) |
| `data.py`, `splits.py` | 데이터 로드·가정 검사·셔플 대조군, 위치 기준 nested CV |
| `encoders.py` | 인코딩 5가지의 전처리 (PLE 경계: 분위수 / pLDDT 공식 / 트리) — train 행에서만 |
| `model.py` | 실험용 토크나이저 + cross-attention 읽기 + probe head. M 개 모델을 한 번에 학습 (배포판과 토큰이 같음) |
| `train.py`, `learners.py`, `cv.py`, `launch.py` | 학습 루프, 학습기(인코딩 5종 / GBDT / TYPE), nested CV 한 칸, 병렬 실행 |
| `metrics.py`, `stats.py`, `compare.py` | subset 지표, 위치 bootstrap·corrected t·Holm, E1·E3 보고서 |
| `e7_*.py`, `model_e7.py`, `run_e7_all.sh` | E7 |
| `e8_*.py`, `run_e8_all.sh` | E8 |
