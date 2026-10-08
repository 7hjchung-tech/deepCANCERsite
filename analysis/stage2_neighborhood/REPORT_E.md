# 3D 구조 이웃(anchor + 8 nearest) FiLM 보정 실험 (E0/E1/E2, 2026-10-07)

단일 seed(Stage2=42, Stage1 checkpoint=seed44 W10)의 탐색적 결과다. test split은 로드하지 않았다. 기존 결과(`runs/stage2_reverse/`, `runs/stage2_joint_ablation/`, `analysis/stage2_diag*/`)는 전혀 건드리지 않았다. 새 산출물은 `runs/stage2_neighborhood/`, `data/structure/results/wt_neighbor_cache.npz`, `analysis/stage2_neighborhood/`에 저장했다.

## 0. 질문에 대한 답

1. **기준 Stage1 성능이 재현됐는가?** 예. E0(Stage1 단독, 재학습 없이 checkpoint 그대로 평가) val subset = **0.7664**로, 기존에 보고된 값과 정확히 같았다. checkpoint·split·metric 정의를 바꿀 필요가 없었다.
2. **E2가 E1/E0보다 개선됐는가?** 아니다. best epoch(=2) 기준 E1=0.7673, E2=0.7668, E0=0.7664로 순서는 **E1 > E2 > E0**였고, position 단위 bootstrap에서 E2−E1(핵심 비교)은 −0.0005[CI −0.0017, 0.0006], E2−E0은 +0.0004[CI −0.0036, 0.0078]로 **둘 다 유의하지 않다**. last epoch에서는 셋 다 E0보다 낮다(E1=0.7588, E2=0.7563). 8개 이웃을 추가한 것이 anchor 하나보다 나은 신호는 이번 결과에서 보이지 않았다.
3. **3D 이웃은 어떤 범위의 residue를 추가했는가?** 8개 이웃의 거리 중앙값 5.95Å(8번째 이웃 중앙값 7.36Å), 서열상 2칸 이내(`|offset|≤2`)인 이웃은 40.4%, 10칸 넘게 떨어진(순수 3차원 접촉) 이웃은 26.5%였다. 즉 이웃의 상당수(약 1/4)가 서열로는 멀지만 3차원에서는 가까운, ESM의 W10 window로는 못 보는 진짜 비서열적 정보였다.
4. **실제 neighbor 입력에 예측이 민감한가?** 네, 하지만 **슬롯 순서에는 완전히 불변**(순열 바꿔도 예측 차이 ~1e-7, 부동소수점 오차 수준)이고 **내용(어떤 이웃인지)에는 민감**하다(donor-swap 시 평균 절대 예측 변화 best=0.073, last=0.263, 최대 1.0). 다만 last checkpoint에서는 **진짜 이웃 대신 무작위 donor의 이웃을 넣어도 val subset이 더 높게 나왔다**(0.7563 → 0.7603) — 모델이 실제 이웃에서 뭔가를 읽고는 있지만, 그게 유용한 신호라는 근거는 아니다.
5. **지지되는 설명 / 구분 못하는 설명**: §6 참고.

## 1. 저장소 확인 (재구현 전 검증)

- 기준 checkpoint: `runs/stage1_v2/unified_reference_delta/W10/shipped_split/seed44/best.pt`. `cfg`에서 직접 읽음: `window_radius=10, layers=[33], d_esm=1280, bottleneck_dim=32, head_hidden=256`. **다른 checkpoint로 바꾸지 않았다.**
- `h_base`(pooled hidden, prediction head 입력): `Stage1Model.forward()`의 `z_seq`(= `self.pooling(K,V,valid)`의 출력). **항상** 반환됨(`return_extras` 여부와 무관) — `src/stage2/stage1_adapter.py:stage1_outputs()`에 `h_base: out["z_seq"]`로 추가해 노출시켰다(기존 `y1/K/V/valid` 반환은 그대로 유지, 하위호환 확인·테스트 통과).
- **hidden dimension은 하드코딩하지 않았다.** `train_stage2_neighborhood.py`가 실제 배치로 `stage1_outputs(...)["h_base"].shape[-1]`을 읽어 `NeighborhoodFiLMModel(d=...)`에 넘긴다. 실행 결과 `d_model=128`로 확인됐다(`TOKEN_DIM`과 일치하지만, 이 값 자체를 checkpoint 실행으로 검증했다).
- `unified_reference_delta`의 reference/delta/flags/edit metadata 처리, variant type schema(`missense/synonymous/indel`), train/val/test split, `StructureTokenizer`(friend's `structure_tokenizer/tokenizer.py`, Qk 방식) — 모두 기존 코드를 그대로 재사용했다(아래 §2 참고).

## 2. WT 구조와 residue 정합 (§4)

- 기존 추출 코드(`data/structure/code/block_b.py :: build_block_a`)를 **재사용**해, AlphaFold 모델(`AF-O43502-F1.pdb`, 376 residue) 전체에 대해 **label 없이** Block A 9-field(여기선 11컬럼: plddt, ss_helix/sheet/loop, rsasa, 6개 site 거리)를 뽑았다. 변이별 feature table(`rad51c_struct_features.csv`)만 있던 것을, 전체 WT residue용으로 **새 스크립트**(`build_wt_neighbor_cache.py`)에서 확장 호출했다.
- **PDB residue 번호 ↔ WT sequence index 정합을 가정하지 않고 직접 검증**했다: `positions`가 1..376 연속(gap·insertion code 없음), `wt_sequence.txt`(376자)와 길이 일치, 376개 residue 전원에서 PDB 3-letter 잔기명이 WT 서열의 해당 위치와 **완전히 일치**(mismatch 0건). 단일 체인(AlphaFold 모델이라 체인 분기 없음)도 확인했다. 이 검증은 `build_wt_neighbor_cache.py`에 하드코딩된 assert로 남아 있어, 다른 구조 파일로 바꾸면 자동으로 실패한다.
- 결측: 6개 기능 부위 거리 컬럼의 NaN 0건, 376개 전원이 CB(또는 CA fallback) 좌표를 가짐(`build_block_a`가 CA 없는 residue는애초에 제외하므로 구조상 "좌표 없음"이 발생하지 않음) — 표로 보고(§3).
- Variant ID → WT anchor position: 기존 pipeline 그대로 manifest의 **`pp`**(1-based, `rad51c_struct_features.csv`/`build_dataset.py`가 쓰는 것과 동일 정의)를 썼다. indel도 포함해 **기존 anchor 정의를 그대로 유지**했고(새로 재정의하지 않음), 가상의 WT 좌표를 만들지 않았다 — 이웃은 항상 실제 WT residue에서만 가져온다.

## 3. 고정 8개 3D 이웃 (§5) — 캐시: `data/structure/results/wt_neighbor_cache.npz`

- 좌표 규칙: `build_block_a`의 대표 좌표(`rep`, **Cβ, 없으면 Cα**)로 만든 376×376 거리행렬(Å)을 그대로 썼다(새로 좌표를 뽑지 않고 기존 계산을 재사용).
- `N(p) = {p} ∪ {p를 제외한 최근접 8개}`, self 제외, 동일 거리 tie는 WT sequence index로 해결(`np.lexsort`), 슬롯0=anchor·나머지는 거리 오름차순.
- **376개 anchor 전원이 8개 이웃을 모두 확보**(부족분 0건 — 보고만 하고 추가 조치는 하지 않음).
- 각 슬롯 저장: WT residue index, 9-field raw feature, anchor까지의 실제 거리(Å), signed offset(`i-p`), anchor flag, valid mask.
- 캐시에는 PDB·annotation 파일 sha256, k=8, feature schema, tie-break 규칙이 메타데이터로 붙어 있어 잘못된 재사용을 막는다.

### 이웃 품질 (학습 전, 보고만 함 — `neighbor_quality.png/json`)

| 지표 | 값 |
|---|---|
| 이웃 거리(8×376=3008개), 중앙값 / 평균 / 최대 | 5.95 / 6.46 / 24.1 Å |
| 8번째(가장 먼) 이웃 거리, 중앙값 / 평균 / 최대 | 7.36 / 8.17 / 24.1 Å |
| `|offset|≤2` 비율 | 40.4% |
| `|offset|>10` 비율(순수 3D 접촉) | 26.5% |
| WT 전체 pLDDT, 평균 / 중앙값 | 84.4 / 92.4 |

예시(anchor 100): 이웃이 offset 1, -1, 14, 15, 38, 41, 42, 179인 residue들로, 서열로는 전혀 가깝지 않은 잔기들이 3D에서는 4.4–6.2Å 안에 있다 — 3D 전용 정보가 실제로 존재함을 확인했다(`neighbor_examples.csv`). 반대로 서열 말단(anchor 2, 376)은 이웃이 전부 서열상 인접 잔기였다(사슬 말단이 풀려 있어 3D 압축 구조가 없는 것으로 보인다 — 해석일 뿐 추가 실험은 하지 않았다).

## 4. 모델 (§7–9) — `src/stage2/neighbor_structure.py`, `src/stage2/neighbor_model.py`

```
T          in R^[B, 9 residue, 9 field, 32]     ResidueStructureTokenizer.inner (기존 Qk 모듈, 재사용)
s_i = A_res(mean_field(T_i))   in R^[B, 9, 32]  A_res = Linear(32,32), 9개 residue에 공유

Q  = W_Q h_base                 [B, 32]   W_Q: 128->32, bias 없음
K_content_i = W_K s_i            [B, 9, 32] W_K: 32->32, bias 없음
V_i = W_V s_i                    [B, 9, 32] W_V: 32->32, bias 없음
K_i = K_content_i + 0.1 * ( E_seq(i-p) + E_dist(d_ip) + E_anchor(is_anchor_i) )
  E_seq    signed sinusoidal(offset), 32차원, 파라미터 없음 (src/stage1/positional.py 재사용)
  E_dist   Gaussian RBF(16 center, 0~30Å) -> Linear(16,32,bias無)
  E_anchor Embedding(2,32)
logits_i = Q·K_i / sqrt(32)              (cosine 정규화·learnable temperature 없음)
a = masked_softmax(logits, residue축);  padding weight = 0; all-masked = 오류
c_struct = sum_i a_i V_i                 [B, 32]

[gamma,beta] = FiLM(c_struct)   32->32->2*128, 마지막 층 zero-init
h_mod = (1+gamma)*h_base + beta
delta_y = Head(concat(h_mod, c_struct))  160->32->1, 마지막 층 zero-init
y_final = y_base + delta_y
```

- `A_res`는 anchor·neighbor 구분 없이 9개 residue에 완전히 공유된다(residue별 개별 가중치 없음). attention은 **residue축**(9개 슬롯)에서만 일어나고, field축(9개)은 tokenizer 내부에서 이미 평균으로 사라진다 — 두 축을 코드 주석에 명시했다.
- metadata(`E_seq+E_dist+E_anchor`)는 고정 scale 0.1을 곱해 Key에만 더한다(Value에는 더하지 않음, learnable temperature 없음) — E1/E2 모두 같은 scale을 쓴다.
- `d_model=128`(§1에서 checkpoint로 확인), 총 파라미터 23,937개(tokenizer 포함), **E1/E2 완전히 동일**.
- ESM·Stage1 전체 frozen/eval 유지(이번 실험은 공동학습을 하지 않는다). 학습 대상은 `ResidueStructureTokenizer`(Qk inner + `A_res`)와 `NeighborhoodFiLMModel`뿐이다.

## 5. 실행과 검증 (§10, §13)

- seed=42, 기존 train/val split, Huber(delta=1.0, reduction=mean), target은 원래 z-score, optimizer/batch/epoch 예산은 **기존 R1 설정을 그대로 재사용**(`configs/stage2/neighborhood.yaml`에 lr=1e-4, weight_decay=0.01, batch_size=32, grad_clip=1.0, min_delta=0, patience=10, max_epochs=120을 명시적으로 기록). epoch 0을 best 후보에 포함했고, best/last checkpoint를 모두 저장했다. 매 epoch의 validation 예측을 var_id와 함께 저장했다(`val_predictions_by_epoch.csv`).
- **E1/E2는 별도 `Stage1Handle`/모델/tokenizer 인스턴스**를 쓴다(이전 공동학습 ablation에서 handle을 공유해 생긴 버그를 교훈 삼아, 이번엔 애초에 공유하지 않도록 설계). E2는 `--init-state-from`으로 E1의 **학습 전** 초기 state(`init_state.pt`)를 그대로 복사해 시작한다(둘 다 동일 초기 가중치, 다른 인스턴스).
- 합성 배치 테스트 12개(`tests/test_stage2_neighborhood.py`) + 실제 캐시 테스트 5개(`tests/test_stage2_neighborhood_real.py`), 전체 레포 테스트 151개 모두 통과:
  - residue축/field축 분리, padding attention=0·valid 합=1, all-masked 오류 처리
  - epoch 0에서 `y_final=y_base`(E1/E2 모두)
  - E1/E2 파라미터 수 동일, 별도 인스턴스에 동일 초기 state 복사 확인
  - **실제 학습 로그로 확인한 gradient 분리**: epoch1 첫 step은 FiLM·Head 마지막 층이 zero-init이라 `head.2`(진짜 마지막 층) 말고는 전부 gradient 0 — 이는 이전 실험(forward/R1/R2)에서도 확인된 동일한 zero-init cascade이지 이번 모델만의 문제가 아니다. epoch2 첫 step에서는: **E1은 `w_q/w_k/dist_proj/anchor_emb` 4개만 정확히 0, 나머지 16/20은 nonzero**(E1은 valid key가 1개라 softmax weight가 항상 1 — spec이 "정상"이라고 명시한 바로 그 현상), **E2는 20/20 전부 nonzero**(여러 key 중 선택이 가능해 Q/K 경로에도 gradient가 흐름). 버그가 아니라 설계대로의 동작임을 실측으로 확인했다.
  - Cβ→Cα fallback과 WT↔PDB 매핑은 §2처럼 캐시 빌드 스크립트 자체의 assert로 검증했다(별도 재구현 없이 기존 `build_block_a` 로직을 신뢰해 재사용).

## 6. 학습 결과

### Best/last 요약 (`summary.csv`)

| 조건 | which | epoch | val subset | val RMSE | val MAE | missense | synonymous | indel |
|---|---|---|---|---|---|---|---|---|
| E0 (Stage1만) | best | 0 | 0.7664 | 3.723 | 2.289 | 0.6951 | 0.2214 | 0.8376 |
| E1 (anchor만) | best | 2 | **0.7673** | 3.769 | 2.317 | 0.6926 | 0.2611 | 0.8419 |
| E1 | last | 12 | 0.7588 | 3.804 | 2.324 | 0.6660 | 0.2134 | 0.8517 |
| E2 (anchor+8) | best | 2 | 0.7668 | 3.767 | 2.316 | 0.6917 | 0.2555 | 0.8419 |
| E2 | last | 12 | 0.7563 | 3.804 | 2.325 | 0.6609 | 0.2159 | 0.8517 |

### position 단위 paired bootstrap (B=2000, val subset Spearman)

| 비교 | best 값 | best 95% CI | best P(≤0) | last 값 | last 95% CI | last P(≤0) |
|---|---|---|---|---|---|---|
| E1−E0 (anchor-only 효과) | +0.0009 | [−0.003, 0.008] | 0.42 | −0.0075 | [−0.025, 0.011] | 0.79 |
| **E2−E1 (3D 이웃 추가 효과)** | **−0.0005** | **[−0.0017, 0.0006]** | **0.78** | **−0.0025** | **[−0.0092, 0.0047]** | **0.81** |
| E2−E0 (전체 효과) | +0.0004 | [−0.004, 0.008] | 0.50 | −0.0101 | [−0.028, 0.008] | 0.86 |

**어느 비교도 유의하지 않다.** 특히 핵심 질문인 E2−E1은 best에서 CI가 [−0.0017, 0.0006]로 매우 좁고 0을 포함하며, 방향도 음수(이웃 추가가 오히려 살짝 나쁨)다. `learning_curves.png`: E1과 E2 곡선이 거의 겹치며 둘 다 epoch 2에서 짧게 Stage1을 넘었다가 이후 다른 모든 Stage2 실험과 같은 패턴으로 떨어진다.

### 학습 중 진단 (attention mass, entropy — `history.csv`)

- E2 epoch 0(학습 전) attention: anchor_mass=0.111(≈1/9, 균등), local_mass=0.367, nonlocal_mass=0.522 — 초기화 시점에 이미 attention이 사실상 균등(entropy_norm=1.0000)하다.
- epoch 12까지 entropy_norm은 1.000→0.983으로 **1.7%만** 날카로워졌고, anchor_mass 0.111→0.122, nonlocal_mass 0.522→0.493로 아주 조금씩만 움직였다. **attention이 "날카로워졌다"고 말할 수준이 아니다** — 거의 균등을 유지한 채 best epoch(2)을 지난다.
- E1은 valid key가 1개뿐이라 `anchor_mass≡1.0`, `entropy_norm`은 정의상 N/A(NaN으로 기록, §11 요구사항대로).

## 7. 구조 context 민감도 (§12, E2 best/last 체크포인트 — `sensitivity_summary.json`)

| | best(epoch2) | last(epoch12) |
|---|---|---|
| 정상 입력 val subset | 0.76679 | 0.75630 |
| **슬롯 순열** val subset / 평균 절대 예측차 | 0.76678 / **6.96e-9** | 0.75629 / **2.79e-8** |
| **donor-swap** val subset / 평균 절대 예측차 | 0.76672 / 0.0727 | **0.76028** / 0.2627 |

- **순열 불변성 확인**: 슬롯 순서를 바꿔도 예측은 수치 오차 수준(1e-7~1e-8)에서 완전히 같다 — 설계대로 위치 자체가 아니라 offset/거리가 slot과 함께 이동하기 때문이다(버그 없음).
- **donor-swap 민감도**: 다른 anchor의 이웃 묶음(9개 feature를 따로 섞지 않고 통째로, donor 자신의 offset/거리 포함)으로 바꾸면 예측이 꽤 바뀐다(평균 0.07~0.26, 최대 1.0) — 모델이 입력 내용을 무시하지 않는다는 뜻이다. 이 교란은 인과효과가 아니라 입력 민감도로만 해석한다(donor의 이웃 환경을 다른 anchor에 붙이는 것은 비현실적 조합일 수 있다).
- **그러나 last checkpoint에서는 donor-swap이 val subset을 오히려 높였다**(0.7563→0.7603). 즉 실제 이웃보다 무작위 다른 anchor의 이웃을 넣는 편이 (근소하게) 더 나은 val 성능을 냈다 — 모델이 학습 후반에 실제 이웃에서 읽어내는 것이 유용한 신호라는 근거가 되지 못한다는 뜷렷한(비록 작지만) 신호다.

## 8. 종합 해석

- **지지되는 설명**: (a) 구현은 정확하다 — Stage1 재현(0.7664 정확히 일치), 순열 불변성(1e-7 수준), E1 zero-gradient/E2 nonzero-gradient 구분이 설계대로 작동한다. (b) 이웃은 실제로 3D 전용 정보를 담고 있다(26.5%가 서열 10칸 이상 떨어진 접촉). (c) 모델이 입력 내용에 민감하다(donor-swap으로 예측이 크게 바뀜). (d) 그럼에도 8개 이웃 추가가 anchor-only보다 낫다는 증거는 없다(E2−E1 비유의, 방향도 음수). (e) 이전 R0/R1/R2/joint-ablation 실험과 같은 패턴(best epoch 2 근처에서 Stage1과 비슷, 이후 지속적으로 악화)이 이번에도 재현됐다 — 네 번째로 다른 설계에서 같은 패턴이 나온 셈이다.
- **아직 구분하지 못하는 설명**: (i) attention이 거의 균등을 유지한 것(entropy_norm 0.983)이 "구조 신호가 약해서"인지 "Q·K 스케일/학습률/λ 선택이 날카로워지기엔 불충분해서"인지는 이번 실험으로 가르지 못한다(이전 forward-attention 실험에서 τ를 낮추거나 head를 늘려도 비슷한 정체가 재현됐다는 사실과 일관되지만, 메커니즘 자체는 다시 검증하지 않았다). (ii) last checkpoint의 donor-swap이 더 나은 현상이 "실제 이웃 신호가 체계적으로 해롭다"는 것인지 "단일 seed의 우연"인지는 반복 실험 없이는 알 수 없다. (iii) 말단 residue(순수 서열-인접 이웃만 가진)와 내부 residue(진짜 3D 접촉 이웃을 가진) 사이에 효과가 다른지는 유형별로 더 쪼개보지 않았다.

## 9. 한계 (단일 seed·탐색적 결과)

- Stage2 seed 42 하나, Stage1 checkpoint seed44 하나다. 위 bootstrap CI는 **이 고정된 모델 쌍 사이의 val 표본 불확실성**이지 여러 seed의 재현성이 아니다. best checkpoint 선택은 validation에서 이뤄졌으므로, 그 선택 편향은 CI에 반영되지 않는다.
- 이번 실행은 E0/E1/E2로 제한했다 — head 수·k(이웃 개수) sweep, nonlocal-only 변형, 공동 학습, OOF 재학습은 하지 않았다(요청대로).
- Cβ→Cα fallback 로직 자체(`build_block_a`)는 기존 코드를 그대로 신뢰해 재사용했고, 이번 작업에서 그 내부 로직을 독립적으로 재검증하지는 않았다.
- donor-swap은 anchor와 donor 환경을 억지로 결합하므로 "실제로 있을 수 있는 구조"가 아니다 — 민감도 측정으로만 쓰고 인과 해석을 하지 않았다.

## 산출물

- 캐시: `data/structure/results/wt_neighbor_cache.npz`(+ 빌드 스크립트 `build_wt_neighbor_cache.py`)
- 모델/엔진 구현: `src/stage2/neighbor_structure.py`, `src/stage2/neighbor_model.py`, `src/stage2/neighbor_engine.py`
- 설정: `configs/stage2/neighborhood.yaml`
- 실행: `train_stage2_neighborhood.py`, `scripts/neighborhood_quality_report.py`, `scripts/neighborhood_summary.py`, `scripts/neighborhood_sensitivity.py`
- 테스트: `tests/test_stage2_neighborhood.py`(12개, 합성), `tests/test_stage2_neighborhood_real.py`(5개, 실제 캐시)
- 결과: `runs/stage2_neighborhood/{E0,E1,E2}/seed42/`(history.json/csv, best.pt, last.pt, init_state.pt, val_predictions_by_epoch.csv, config.json, fit_positions.json)
- 분석: `analysis/stage2_neighborhood/`(neighbor_quality.{json,csv,png}, neighbor_examples.csv, summary.csv, contrasts_bootstrap.csv, learning_curves.png, sensitivity_{best,last}.csv, sensitivity_summary.json)
- 재실행: `python build_wt_neighbor_cache.py && python scripts/neighborhood_quality_report.py && python train_stage2_neighborhood.py --condition E0 ... && python train_stage2_neighborhood.py --condition E1 ... && python train_stage2_neighborhood.py --condition E2 --init-state-from <E1 out dir> ... && python scripts/neighborhood_summary.py && python scripts/neighborhood_sensitivity.py`
