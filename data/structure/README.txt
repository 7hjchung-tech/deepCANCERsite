DeepRAD51C — 구조 정보 Encoding  (진행상황 스냅샷)
================================================================

■ 먼저 볼 것
  Block_B_설명.pdf     ← 조원 설명용. 그림으로 Block B 원리 설명 (indel 생물학 + 각 feature 의미)
  진행상황_리포트.pdf   ← 프로젝트 전체 진행상황 (개요/결과/파일설명/앞으로 할 일)

■ 한 줄 요약
  RAD51C 변이(SAV + in-frame indel)를 "구조 feature만"으로 25차원 벡터로 인코딩.
  구조 정보만으로 test Pearson r ~ 0.50, depleted 분류 AUC ~ 0.80 확인.

■ 최종 벡터 25-dim = Block A(11) + Block B(11) + 타입 one-hot(3)
  - Block A : 앵커 한 점 구조 (pLDDT, 2차구조, 표면노출, 기능자리까지 거리)
  - Block B : indel '재봉합 사건' 인코더 (SAV는 전부 0)
  - one-hot : [sav, del, ins] 3-way

■ 폴더 구조
  code/     구현 코드
    block_b.py         ★ 핵심. Block A + Block B(indel 11-dim) + one-hot = 25-dim 인코더
    assemble.py        조립기 데모 (sav/del/ins → 25-dim 검증)
    build_dataset.py   split_manifest.csv → 전체 [4914,25] feature 표 생성
    analysis.py        구조 feature 진단 + Ridge 예측 baseline
    generate_report.py 진행상황_리포트.pdf 생성
    explain_blockb.py  Block_B_설명.pdf (조원 설명용 그림) 생성

  results/  나온 결과물
    rad51c_X.npy               모델 입력 [4914, 25] 구조 feature
    rad51c_y.npy               라벨 [4914] z_score_D4_D14
    rad51c_struct_features.csv 사람용 통합표 (var_id, split, 라벨 + 25 feature)
    rad51c_meta.csv            변이 메타
    rad51c_blockA.csv          residue별 Block A 테이블 (376행)
    analysis_results.json      분석 수치 원본 (상관/baseline/AUC)

  inputs/   입력 파일 (재현용)
    AF-O43502-F1.pdb                RAD51C AlphaFold WT 구조
    rad51c_residue_annotation.csv   기능 자리 정의 (거리 계산용)
    split_manifest.csv              SGE 데이터 (위치/타입/변이서열/z_score/split)
    wt_sequence.txt                 WT 서열 (376 aa)

■ 재현 방법
  의존성: numpy, pandas, biotite  (+ PDF: reportlab). torch/scipy 불필요.
  inputs/ 파일들을 code/ 와 같은 작업 폴더에 두고:
    python build_dataset.py       # → rad51c_X.npy [4914,25], rad51c_y.npy, features.csv
    python analysis.py            # → analysis_results.json
    python generate_report.py     # → 진행상황_리포트.pdf
    python explain_blockb.py      # → Block_B_설명.pdf

■ 앞으로 (자세한 건 진행상황_리포트.pdf 5장)
  1. 다운스트림 MLP 연결 (torch 필요, GPU 서버 권장)
  2. ESM WT-difference 임베딩 등 다른 블록과 통합
  3. multi-residue indel 많은 다른 유전자로 확장 (span pooling·junction feature 진가 발휘)

■ 원본 작업 폴더
  /Users/choseungwon/deeprad51c_struct/  (venv 포함, 실행은 그 안의 venv/bin/python)
