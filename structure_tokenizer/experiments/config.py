"""
config.py — 검증 실험(E1·E2·E3·E7·E8)의 상수·프로토콜·탐색공간·경로. 단일 진실 공급원.

다른 파일에서 feature 이름·분할 규칙·탐색공간을 문자열/숫자로 다시 적지 않는다.
각 값 옆에 **근거(레퍼런스)** 를 적는다. 근거가 "우리 결정"인 것은 그렇게 적는다.

"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                 # structure_tokenizer/experiments
TOK_DIR = os.path.dirname(HERE)                                    # structure_tokenizer
REPO_ROOT = os.path.dirname(TOK_DIR)                               # deepCANCERsite 레포 루트
# 데이터는 레포에 올리지 않는다(엠바고). ../build_dataset.py 가 레포 안의 입력으로 만든다.
DATA_PATH = os.environ.get("STRUCTTOK_DATA", os.path.join(TOK_DIR, "data", "v2_dataset.csv"))
# 이 파일의 sha256. 데이터가 바뀌면 이전 결과와 비교가 성립하지 않으므로 로드 시 대조한다.
DATA_SHA256 = "0b4f7c60aba08b9609a9e4cd2ccd21be58c6b3568addc20a9f3017ce339f381d"
RESULTS_DIR = os.path.join(HERE, "results")

# 외부 입력 (E7·E8 만 필요). 기본값은 레포 안 위치, 환경변수로 바꿀 수 있다.
#   ESM_HANDOFF : 민선 handoff (브랜치 minseon/esm-module 의 handoffs/module_a_hr_v1, git lfs pull 필요)
#   FROZEN_PT   : 그 안의 동결 ESM 캐시
#   STAGE1_HF   : 호준 Stage 1 체크포인트 (HF DeepCANCERsite/stage1-checkpoints 를 받아 둔 폴더)
ESM_HANDOFF = os.environ.get("ESM_HANDOFF", os.path.join(REPO_ROOT, "handoffs", "module_a_hr_v1"))
FROZEN_PT = os.environ.get("FROZEN_PT", os.path.join(ESM_HANDOFF, "payload", "frozen.pt"))
STAGE1_HF = os.environ.get("STAGE1_HF", os.path.join(HERE, "external", "stage1_hf"))

# ------------------------------------------------------------------ 스키마
# 명세 §3 의 # 순서. 인덱스 = field-ID embedding 의 행 번호.
FIELD_ORDER = [
    "A_plddt",                 # 0  연속
    "A_ss_class",              # 1  범주 3-class
    "A_rsasa",                 # 2  연속
    "A_dist_walker_a",         # 3  연속
    "A_dist_walker_b",         # 4  연속
    "A_dist_atp_contact",      # 5  연속
    "A_dist_ssdna_binding",    # 6  연속
    "A_dist_bcdx2_interface",  # 7  연속
    "A_dist_cx3_interface",    # 8  연속
]
SS_FIELD = "A_ss_class"
CONT_FIELDS = [f for f in FIELD_ORDER if f != SS_FIELD]      # 연속 8개, FIELD_ORDER 상대순서 유지
SS_CLASSES = ["helix", "sheet", "loop"]
TYPE3 = ["missense", "synonymous", "indel"]
TARGET = "z_score_D4_D14"
POS = "anchor_pos"

N_TOKENS, N_CONT = len(FIELD_ORDER), len(CONT_FIELDS)
SS_TOKEN = FIELD_ORDER.index(SS_FIELD)                        # 1
PLDDT_COL = CONT_FIELDS.index("A_plddt")                      # 0

assert N_TOKENS == 9 and N_CONT == 8, "명세 §3: 연속 8 + 범주 1"
assert SS_TOKEN == 1 and PLDDT_COL == 0, "model.py 의 토큰 조립이 이 배치를 가정한다"

# 명세 §4.3 (확정) — AlphaFold 공식 pLDDT 신뢰구간 경계
PLDDT_DOMAIN_BINS = np.array([0.0, 50.0, 70.0, 90.0, 100.0])

# 명세 §2.1 의 개수표. data.py 가 로드 시 대조한다.
EXPECTED_COUNTS = {  # (split, type3) -> n
    ("train", "missense"): 3195, ("val", "missense"): 671, ("test", "missense"): 689,
    ("train", "synonymous"): 675, ("val", "synonymous"): 158, ("test", "synonymous"): 140,
    ("train", "indel"): 254, ("val", "indel"): 52, ("test", "indel"): 53,
}
N_ROWS, N_POSITIONS = 5887, 375

# ------------------------------------------------------------------ 프로토콜
PROTOCOL = {
    # 바깥 CV: position 기반 K-fold, fold seed 를 바꿔 반복.
    #   근거: ProteinGym(Notin+ 2023) modulo/contiguous = position 격리,
    #         Bouthillier+ 2021 = 분할도 변동원으로 반복해야 함
    "outer_k": 5,
    "repeats": 2,
    "fold_seed0": 2026,
    # 안쪽 CV: 선택 전용. 바깥 test 는 선택에 절대 안 씀 (Cawley & Talbot 2010)
    "inner_k": 4,
    # 탐색: Optuna TPE 100회 (Gorishniy+ 2022 부록 E 와 동일)
    "n_trials": 100,
    "hpo_seed0": 7,
    # 선택된 설정으로 바깥 train 전체에 다시 학습할 때의 seed 개수.
    #   Gorishniy+ 2022 는 15개. 비용 때문에 5개 — 우리 결정
    "refit_seeds": 5,
    # 학습 루프: AdamW, 스케줄 없음, patience 16 (Gorishniy+ 2022 부록 E, TabM 동일)
    "batch_size": 128,        # 고정. 논문도 데이터셋별 고정값 사용 — 값 자체는 우리 결정
    "max_epochs": 300,
    "patience": 16,
    # 명세 §6 — Stage 2 로 넘기는 차원
    "fusion_dim": 128,
    # probe head 용량(고정, 튜닝 안 함). 계획서 §2.2
    "head_hidden": 128,
    # 셔플 대조군 seed = shuffle_seed0 + repeat
    "shuffle_seed0": 1000,
}

# ------------------------------------------------------------------ 탐색공간
# 공통 (모든 NN 후보에 동일하게 적용 — Melis+ 2018: 같은 예산·같은 공간)
SPACE_COMMON = {
    "lr": ("log", 1e-4, 5e-3),            # TabM (Gorishniy+ ICLR 2025)
    # Gorishniy+ 2022 는 {0, LogU[1e-6,1e-3]}. fANOVA 가 조건부 파라미터를 다루기
    # 어렵기 때문에 LogU[1e-6,1e-2] 하나로 둔다(1e-6 ≈ 0) — 우리 결정
    "weight_decay": ("log", 1e-6, 1e-2),
    # Gorishniy+ 2022: linear 출력 차원 UniformInt[1,128]. 해석 편의상 5단계 — 우리 결정
    "d_s": ("cat", [8, 16, 32, 64, 128]),
}
SPACE_MLP_HEAD = {"dropout": ("float", 0.0, 0.5)}               # Gorishniy+ 2022 MLP 공간
# PLE: Gorishniy+ 2022 는 [2,256]. 학습 position 이 ~300개라 64 초과는 구간당
# position 5개 미만 → [2,64] — 우리 결정. 경계검사(계획서 §5-6)로 끝값 확인.
SPACE_PLE = {"n_bins": ("intlog", 2, 64), "extrapolate": ("cat", [False, True])}
# T-PLE: Gorishniy+ 2022 부록 E.8 (leaf [2,256], min items [1,128], min gain LogU[1e-9,0.01])
SPACE_TREE = {"n_bins": ("intlog", 2, 64), "min_samples_leaf": ("intlog", 1, 128),
              "min_impurity_decrease": ("log", 1e-9, 1e-2),
              "extrapolate": ("cat", [False, True])}
# PLR: k 는 Gorishniy+ 2022 [1,128], σ 는 rtdl_num_embeddings README 의
# "research project" 권장 LogU[0.01, 10]
SPACE_PLR = {"n_frequencies": ("intlog", 1, 128), "sigma": ("log", 0.01, 10.0)}

# E7 (동결 Stage 1 위의 Stage 2) 탐색공간. lr 하한은 1e-5: 동결 Stage 1 이 train 에 맞춰져 있어
# Stage 2 가 몇 에폭 만에 멈추므로 더 느린 학습도 시험 (본 실행 결과 보기 전에 정함)
SPACE_STAGE2 = {"lr": ("log", 1e-5, 3e-3), "weight_decay": ("log", 1e-6, 1e-2),
                "dropout": ("float", 0.0, 0.5)}
SPACE_STAGE2_STRUCT = {"d_s": ("cat", [8, 16, 32, 64, 128]), "n_bins": ("intlog", 2, 64)}
