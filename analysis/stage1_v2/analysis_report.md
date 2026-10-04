# Stage 1 Multi-Seed Sweep -- Analysis Report
Generated from 45 / 45 planned runs under `runs/stage1_v2`.

## 1. Configuration별 요약 (파라미터/속도/정확도)
전체 수치는 `summary_by_configuration.csv` 참고. 핵심 열:

| model                   |   window |   n_seeds |   best_epoch_median |   stopped_epoch_median |   median_seconds_per_epoch_median |   val_subset_best_mean |   val_subset_best_sd |   test_subset_mean |   test_subset_sd |   test_subset_ci95_lo |   test_subset_ci95_hi |
|:------------------------|---------:|----------:|--------------------:|-----------------------:|----------------------------------:|-----------------------:|---------------------:|-------------------:|-----------------:|----------------------:|----------------------:|
| branched_projection     |        5 |         5 |             10.0000 |                20.0000 |                           10.8000 |                 0.7089 |               0.0091 |             0.5757 |           0.0142 |                0.5647 |                0.5865 |
| branched_projection     |       10 |         5 |             12.0000 |                22.0000 |                           15.1000 |                 0.7038 |               0.0045 |             0.6446 |           0.0119 |                0.6350 |                0.6530 |
| branched_projection     |       20 |         5 |             15.0000 |                25.0000 |                           14.2500 |                 0.6775 |               0.0090 |             0.6341 |           0.0096 |                0.6273 |                0.6423 |
| paired_delta            |        5 |         5 |             20.0000 |                30.0000 |                           10.8000 |                 0.6972 |               0.0059 |             0.5498 |           0.0107 |                0.5407 |                0.5566 |
| paired_delta            |       10 |         5 |             16.0000 |                26.0000 |                           15.2000 |                 0.7064 |               0.0109 |             0.6398 |           0.0083 |                0.6332 |                0.6462 |
| paired_delta            |       20 |         5 |             17.0000 |                27.0000 |                           14.3000 |                 0.6847 |               0.0165 |             0.6289 |           0.0085 |                0.6222 |                0.6353 |
| unified_reference_delta |        5 |         5 |             32.0000 |                42.0000 |                           10.5000 |                 0.7426 |               0.0052 |             0.6487 |           0.0112 |                0.6392 |                0.6566 |
| unified_reference_delta |       10 |         5 |             52.0000 |                62.0000 |                           14.9000 |                 0.7579 |               0.0119 |             0.6956 |           0.0048 |                0.6918 |                0.6991 |
| unified_reference_delta |       20 |         5 |             14.0000 |                24.0000 |                           14.0000 |                 0.7146 |               0.0299 |             0.6561 |           0.0196 |                0.6435 |                0.6732 |

## 2. Fixed-budget (고정 epoch) validation 비교
각 셀은 해당 epoch까지 실제로 도달한 seed만의 평균 -- 조기 종료로 도달 못한 seed는 보간/추정하지 않고 NaN으로 남김 (`fixed_budget_comparison.csv` 참고).

| model                   |   window |   ep10_mean |   ep10_n |   ep20_mean |   ep20_n |   ep30_mean |   ep30_n |   ep40_mean |   ep40_n |   ep60_mean |   ep60_n |   ep80_mean |   ep80_n |
|:------------------------|---------:|------------:|---------:|------------:|---------:|------------:|---------:|------------:|---------:|------------:|---------:|------------:|---------:|
| branched_projection     |        5 |      0.7082 |        5 |      0.6888 |        5 |    nan      |        0 |    nan      |        0 |    nan      |        0 |      nan    |        0 |
| branched_projection     |       10 |      0.698  |        5 |      0.6822 |        5 |    nan      |        0 |    nan      |        0 |    nan      |        0 |      nan    |        0 |
| branched_projection     |       20 |      0.6519 |        5 |      0.6663 |        4 |      0.6731 |        1 |    nan      |        0 |    nan      |        0 |      nan    |        0 |
| paired_delta            |        5 |      0.6848 |        5 |      0.6941 |        5 |      0.674  |        3 |    nan      |        0 |    nan      |        0 |      nan    |        0 |
| paired_delta            |       10 |      0.6801 |        5 |      0.6894 |        5 |      0.6778 |        1 |    nan      |        0 |    nan      |        0 |      nan    |        0 |
| paired_delta            |       20 |      0.6452 |        5 |      0.6788 |        4 |      0.6936 |        1 |    nan      |        0 |    nan      |        0 |      nan    |        0 |
| unified_reference_delta |        5 |      0.7109 |        5 |      0.719  |        5 |      0.7161 |        3 |      0.7324 |        3 |    nan      |        0 |      nan    |        0 |
| unified_reference_delta |       10 |      0.6963 |        5 |      0.7094 |        5 |      0.7336 |        5 |      0.7346 |        4 |      0.7457 |        3 |      nan    |        0 |
| unified_reference_delta |       20 |      0.6746 |        5 |      0.6685 |        3 |      0.6849 |        2 |      0.7182 |        1 |      0.7386 |        1 |        0.74 |        1 |

## 3. Best epoch / stopped epoch 분포 (9 configuration 전체)
`best_epoch_analysis.csv` 및 `plots/comparison/best_epoch_boxplot.png`, `stopped_epoch_boxplot.png` 참고.

## 4. best/test score와 best_epoch의 연관성 (scatter)
- val_subset_best vs best_epoch: Spearman rho=0.419 (p=0.004, n=45) -- 연관성일 뿐 인과관계 아님.
- test_subset vs best_epoch: Spearman rho=0.278 (p=0.065, n=45) -- 연관성일 뿐 인과관계 아님.

## W10 심층 분석 (paired_delta)
**1. seed42의 76 epoch가 다른 seed에서도 반복되는가?**
- W10 seed별 stopped_epoch: {42: 34, 43: 26, 44: 25, 45: 28, 46: 25}
- median=26.0, sd=3.4, range=[25, 34]

**2. W10의 best_epoch가 W5/W20보다 일관되게 늦은가?**
- W5: best_epoch median=20.0, mean=20.2, values=[18, 19, 20, 22, 22]
- W10: best_epoch median=16.0, mean=17.6, values=[15, 15, 16, 18, 24]
- W20: best_epoch median=17.0, mean=15.6, values=[4, 15, 17, 18, 24]

**3-4. 장기간 점진적 향상 vs plateau 중 fluctuation:**
- seed42: n_patience_resets=12, min_reset_delta=0.0004178364882653218, best_minus_plateau=0.03653713767305489
- seed43: n_patience_resets=9, min_reset_delta=0.0008921187949471054, best_minus_plateau=0.04028252272974697
- seed44: n_patience_resets=12, min_reset_delta=0.007135387639070956, best_minus_plateau=0.07115153809428387
- seed45: n_patience_resets=9, min_reset_delta=0.0023307681265495317, best_minus_plateau=0.04504880504518638
- seed46: n_patience_resets=10, min_reset_delta=0.0020238177535654156, best_minus_plateau=0.04315547028466826
  (min_reset_delta이 작을수록, 그리고 best_minus_plateau이 작을수록 '작은 fluctuation이 patience를 반복 리셋'시켰을 가능성이 큼 -- 반대로 reset마다 delta가 꾸준히 크면 점진적 향상 쪽에 더 가까움. min_delta=0으로 실행되었으므로 위 delta가 곧 실제 개선 판정 기준이었다는 점에 유의.)

**5. best epoch가 늦은 seed일수록 test 성능도 높은가? (같은 config 내 paired, n=5 -- 매우 작은 표본)**
- W10 내부 5-seed Spearman(best_epoch, test_subset) = -0.410 (p=0.493, n=5) -- n=5라 신뢰구간이 매우 넓음, 참고용.

**6. epoch 수-성능 관계가 특정 seed 하나에 의해 주도되는가? (leave-one-out)**
- seed42 제외 시 rho=0.211 (p=0.789, n=4)
- seed43 제외 시 rho=-0.632 (p=0.368, n=4)
- seed44 제외 시 rho=-0.400 (p=0.600, n=4)
- seed45 제외 시 rho=-0.316 (p=0.684, n=4)
- seed46 제외 시 rho=-0.800 (p=0.200, n=4)

**7. W10의 우위가 전체 test 성능인지 missense/indel(subset)에 국한되는지:**
- W5: test_spearman(전체) mean=0.5178, test_missense mean=0.4642, test_indel mean=0.6355, test_subset mean=0.5498
- W10: test_spearman(전체) mean=0.5841, test_missense mean=0.5446, test_indel mean=0.7349, test_subset mean=0.6398
- W20: test_spearman(전체) mean=0.5941, test_missense mean=0.5558, test_indel mean=0.7020, test_subset mean=0.6289

**결론 프레이밍**: 위 수치만으로 '더 오래 학습해서 좋아졌다'를 단정하지 않는다. 가능성: (a) 추가 epoch의 실제 기여, (b) W10 representation이 좋은 성능과 장기 개선을 동시에 유발(공통원인), (c) validation noise가 early stopping을 지연시켰을 뿐(min_delta=0의 영향), (d) seed42 단일 outlier, (e) validation 기준 model selection의 winner's curse. 5-seed로는 이 중 하나를 확정할 수 없고, 위 세부 수치(리셋 delta 크기, leave-one-out 안정성, 그룹별 분해)가 각 가설에 대한 상대적 증거를 제공할 뿐이다.

## W10 심층 분석 (branched_projection)
**1. seed42의 76 epoch가 다른 seed에서도 반복되는가?**
- W10 seed별 stopped_epoch: {42: 23, 43: 21, 44: 20, 45: 26, 46: 22}
- median=22.0, sd=2.1, range=[20, 26]

**2. W10의 best_epoch가 W5/W20보다 일관되게 늦은가?**
- W5: best_epoch median=10.0, mean=10.4, values=[10, 10, 10, 10, 12]
- W10: best_epoch median=12.0, mean=12.4, values=[10, 11, 12, 13, 16]
- W20: best_epoch median=15.0, mean=15.6, values=[4, 15, 15, 18, 26]

**3-4. 장기간 점진적 향상 vs plateau 중 fluctuation:**
- seed42: n_patience_resets=9, min_reset_delta=0.006519017220670276, best_minus_plateau=0.05249058427298248
- seed43: n_patience_resets=5, min_reset_delta=0.011125193994527938, best_minus_plateau=0.04666735223038565
- seed44: n_patience_resets=10, min_reset_delta=0.0012875515111960834, best_minus_plateau=0.07252410253224162
- seed45: n_patience_resets=7, min_reset_delta=0.003906915074942785, best_minus_plateau=0.041748800285474785
- seed46: n_patience_resets=7, min_reset_delta=0.004612556524175138, best_minus_plateau=0.04608987384864649
  (min_reset_delta이 작을수록, 그리고 best_minus_plateau이 작을수록 '작은 fluctuation이 patience를 반복 리셋'시켰을 가능성이 큼 -- 반대로 reset마다 delta가 꾸준히 크면 점진적 향상 쪽에 더 가까움. min_delta=0으로 실행되었으므로 위 delta가 곧 실제 개선 판정 기준이었다는 점에 유의.)

**5. best epoch가 늦은 seed일수록 test 성능도 높은가? (같은 config 내 paired, n=5 -- 매우 작은 표본)**
- W10 내부 5-seed Spearman(best_epoch, test_subset) = 0.100 (p=0.873, n=5) -- n=5라 신뢰구간이 매우 넓음, 참고용.

**6. epoch 수-성능 관계가 특정 seed 하나에 의해 주도되는가? (leave-one-out)**
- seed42 제외 시 rho=-0.400 (p=0.600, n=4)
- seed43 제외 시 rho=0.000 (p=1.000, n=4)
- seed44 제외 시 rho=0.600 (p=0.400, n=4)
- seed45 제외 시 rho=0.200 (p=0.800, n=4)
- seed46 제외 시 rho=0.000 (p=1.000, n=4)

**7. W10의 우위가 전체 test 성능인지 missense/indel(subset)에 국한되는지:**
- W5: test_spearman(전체) mean=0.5352, test_missense mean=0.4929, test_indel mean=0.6585, test_subset mean=0.5757
- W10: test_spearman(전체) mean=0.5810, test_missense mean=0.5520, test_indel mean=0.7372, test_subset mean=0.6446
- W20: test_spearman(전체) mean=0.5953, test_missense mean=0.5579, test_indel mean=0.7103, test_subset mean=0.6341

**결론 프레이밍**: 위 수치만으로 '더 오래 학습해서 좋아졌다'를 단정하지 않는다. 가능성: (a) 추가 epoch의 실제 기여, (b) W10 representation이 좋은 성능과 장기 개선을 동시에 유발(공통원인), (c) validation noise가 early stopping을 지연시켰을 뿐(min_delta=0의 영향), (d) seed42 단일 outlier, (e) validation 기준 model selection의 winner's curse. 5-seed로는 이 중 하나를 확정할 수 없고, 위 세부 수치(리셋 delta 크기, leave-one-out 안정성, 그룹별 분해)가 각 가설에 대한 상대적 증거를 제공할 뿐이다.

## W10 심층 분석 (unified_reference_delta)
**1. seed42의 76 epoch가 다른 seed에서도 반복되는가?**
- W10 seed별 stopped_epoch: {42: 65, 43: 33, 44: 62, 45: 66, 46: 42}
- median=62.0, sd=13.5, range=[33, 66]

**2. W10의 best_epoch가 W5/W20보다 일관되게 늦은가?**
- W5: best_epoch median=32.0, mean=27.0, values=[11, 13, 32, 35, 44]
- W10: best_epoch median=52.0, mean=43.6, values=[23, 32, 52, 55, 56]
- W20: best_epoch median=14.0, mean=28.0, values=[9, 9, 14, 23, 85]

**3-4. 장기간 점진적 향상 vs plateau 중 fluctuation:**
- seed42: n_patience_resets=18, min_reset_delta=0.00017908936979305068, best_minus_plateau=0.050396710426325764
- seed43: n_patience_resets=10, min_reset_delta=0.00131081663658561, best_minus_plateau=0.0528269176478362
- seed44: n_patience_resets=17, min_reset_delta=0.0001045555252703334, best_minus_plateau=0.05938794802374325
- seed45: n_patience_resets=20, min_reset_delta=6.302877594377421e-05, best_minus_plateau=0.0467187947489347
- seed46: n_patience_resets=16, min_reset_delta=7.0091687269902e-05, best_minus_plateau=0.05451471465053659
  (min_reset_delta이 작을수록, 그리고 best_minus_plateau이 작을수록 '작은 fluctuation이 patience를 반복 리셋'시켰을 가능성이 큼 -- 반대로 reset마다 delta가 꾸준히 크면 점진적 향상 쪽에 더 가까움. min_delta=0으로 실행되었으므로 위 delta가 곧 실제 개선 판정 기준이었다는 점에 유의.)

**5. best epoch가 늦은 seed일수록 test 성능도 높은가? (같은 config 내 paired, n=5 -- 매우 작은 표본)**
- W10 내부 5-seed Spearman(best_epoch, test_subset) = -0.700 (p=0.188, n=5) -- n=5라 신뢰구간이 매우 넓음, 참고용.

**6. epoch 수-성능 관계가 특정 seed 하나에 의해 주도되는가? (leave-one-out)**
- seed42 제외 시 rho=-0.800 (p=0.200, n=4)
- seed43 제외 시 rho=-0.400 (p=0.600, n=4)
- seed44 제외 시 rho=-0.800 (p=0.200, n=4)
- seed45 제외 시 rho=-1.000 (p=0.000, n=4)
- seed46 제외 시 rho=-0.400 (p=0.600, n=4)

**7. W10의 우위가 전체 test 성능인지 missense/indel(subset)에 국한되는지:**
- W5: test_spearman(전체) mean=0.5914, test_missense mean=0.5592, test_indel mean=0.7381, test_subset mean=0.6487
- W10: test_spearman(전체) mean=0.6141, test_missense mean=0.5750, test_indel mean=0.8161, test_subset mean=0.6956
- W20: test_spearman(전체) mean=0.6035, test_missense mean=0.5774, test_indel mean=0.7349, test_subset mean=0.6561

**결론 프레이밍**: 위 수치만으로 '더 오래 학습해서 좋아졌다'를 단정하지 않는다. 가능성: (a) 추가 epoch의 실제 기여, (b) W10 representation이 좋은 성능과 장기 개선을 동시에 유발(공통원인), (c) validation noise가 early stopping을 지연시켰을 뿐(min_delta=0의 영향), (d) seed42 단일 outlier, (e) validation 기준 model selection의 winner's curse. 5-seed로는 이 중 하나를 확정할 수 없고, 위 세부 수치(리셋 delta 크기, leave-one-out 안정성, 그룹별 분해)가 각 가설에 대한 상대적 증거를 제공할 뿐이다.

## 5. 통계적 주의사항
- seed 5개는 여전히 작은 표본. 아래는 강한 유의성 주장이 아니라 방향성 참고용.
- synonymous 그룹 test 표본 수: min=140, median=140 -- 작을 경우 synonymous Spearman은 매우 불안정하니 강조하지 말 것.
- 같은 configuration 내 seed 간 test_subset SD 평균=0.0110 vs configuration 평균값들의 SD=0.0433 -- configuration 간 차이가 seed 분산보다 큰 편.
- val 성능과 test 성능은 항상 분리해서 봐야 함 (`val_subset_best*` vs `test_subset*` 컬럼 구분).
- 동일 split/seed를 공유하는 configuration끼리는 seed로 매칭한 paired difference가 가능 -- `summary_by_run.csv`를 (model,window)로 pivot해 직접 계산 권장 (예: 같은 seed의 paired_delta W10 vs unified_reference_delta W10 test_subset 차이).
