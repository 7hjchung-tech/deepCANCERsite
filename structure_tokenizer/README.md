# structure_tokenizer — RAD51C 구조 토크나이저 (Qk)

담당: 조승원. 변이 위치의 구조 값 9개를 Stage 2 가 읽을 토큰 9개로 바꾼다.
이 토크나이저를 고르고 검증한 실험은 [`experiments/`](experiments/README.md)에 있다.

> 데이터 주의: SGE 데이터는 엠바고 상태다(`README_P1.md`). 데이터 파일은 올리지 않고(`.gitignore`),
> `build_dataset.py`로 레포 안의 입력에서 만든다.

## 무엇을 만드나

| 항목 | 내용 |
|---|---|
| 입력 | 변이 시작 위치(앵커) 잔기의 구조 값 9개. 같은 위치의 변이는 값이 같다 |
| 연속 8개 | pLDDT, rSASA, 기능 부위까지 거리 6개 (Walker A, Walker B, ATP 접촉, ssDNA 결합, BCDX2 접촉면, CX3 접촉면; Å) |
| 범주 1개 | 2차 구조 (helix / sheet / loop) |
| 연속값 인코딩 (Qk) | PLE (Gorishniy+ 2022). pLDDT 경계는 AlphaFold 공식 [0, 50, 70, 90, 100] 고정, 나머지 7개는 train 분위수 경계. 구간마다 0~1 로 자름 |
| 토큰 | token = LayerNorm(값 인코딩 + 항목 이름 임베딩), FT-Transformer 방식 |
| 출력 | `[B, 9, d_s]`, 토큰 순서 = `FIELD_ORDER` (pLDDT, ss, rSASA, 거리 6개) |
| 잠정 하이퍼파라미터 | 구간 수 4 (pLDDT 제외), d_s 32 — 최종값은 Stage 2 설계가 정해진 뒤 확정 |

토큰 9개는 하나로 합치지(pooling) 않는다. Stage 2 가 변이 쪽 query(예: Stage 1 요약 벡터)로
cross-attention 해서 읽고, 변이유형은 Stage 2 의 조건으로 넣는다 (2026-09-30 설계).

## 준비

```bash
cd structure_tokenizer
pip install -r requirements.txt
python build_dataset.py          # data/v2_dataset.csv (변이 5,887행) 생성
python -m pytest tests -q        # 7개
```

`build_dataset.py`는 `data/split_manifest.csv`, `data/structure/inputs/`의 AlphaFold 구조와 기능 부위 주석,
`data/structure/code/block_b.py`의 잔기별 구조 계산을 쓴다. 결과 파일의 sha256 이 검증 실험에 쓴 파일과 같은지도 알려 준다.

## 쓰는 법

```python
import pandas as pd, torch
from tokenizer import StructureTokenizer, features_from_frame

df = pd.read_csv("data/v2_dataset.csv")
cont, ss = features_from_frame(df)                      # cont [N, 8] 원래 단위, ss [N] 0/1/2
train = (df.split == "train").to_numpy()

tok = StructureTokenizer.from_train(cont[train])        # 구간 경계는 반드시 train 행만으로
tokens = tok(torch.as_tensor(cont), torch.as_tensor(ss))   # [5887, 9, 32]

tok.save("structure_tokenizer.pt")                      # 경계 + 가중치
tok = StructureTokenizer.load("structure_tokenizer.pt")
```

- 토크나이저의 가중치(구간별 사영, ss·항목 임베딩, LayerNorm)는 Stage 2 와 함께 학습한다. 경계는 학습하지 않는다.
- 다른 분할(교차검증 등)을 쓸 때는 그 분할의 train 행으로 경계를 다시 만든다.
- 변이는 `var_id`로 Stage 1·Stage 2 입력과 맞춘다.

## 파일

| 파일 | 역할 |
|---|---|
| `tokenizer.py` | `fit_qk_bins`, `StructureTokenizer`, `features_from_frame` |
| `build_dataset.py` | 레포 입력 → `data/v2_dataset.csv` |
| `tests/test_tokenizer.py` | 경계(pLDDT 고정, train 분위수), PLE 값, 토큰 순서, 기울기, 저장·불러오기 |
