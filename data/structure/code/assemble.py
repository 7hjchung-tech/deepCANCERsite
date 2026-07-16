"""
assemble.py — 변이 -> 최종 22-dim 구조 feature 벡터 조립

  22-dim = Block A(11, 앵커 한 점) + Block B(11, indel 재봉합 사건)
  변이 타입(sav/del/ins) one-hot은 struct에서 빠짐 — meta stream이 담당.

실제 SGE 데이터를 붙이기 전, 조립기가 sav/del/ins에 대해
올바른 22-dim을 만드는지 확인하는 데모.
"""

import numpy as np
import pandas as pd
from block_b import build_block_a, BlockBEncoder, FULL_COLS, FULL_DIM

ba = build_block_a()
enc = BlockBEncoder(ba)

# 예시 변이 (나중엔 SGE 파싱 결과가 이 자리에 들어온다)
variants = [
    {"variant": "R258H",         "anchor_pos": 258, "var_type": "sav"},
    {"variant": "K130_L132del",  "anchor_pos": 130, "var_type": "del", "del_len": 3},
    {"variant": "A350_ins2",     "anchor_pos": 350, "var_type": "ins", "ins_len": 2},
]

X = enc.encode_full_table(variants)
print(f"조립 결과: {X.shape}   (변이 x {FULL_DIM}-dim)")
assert X.shape == (len(variants), FULL_DIM)

# 사람이 읽을 수 있는 표로
df = pd.DataFrame(X, columns=FULL_COLS)
df.insert(0, "variant", [v["variant"] for v in variants])

# 블록 경계 (Block A 11 + Block B 11)
A_END = 11
B_END = A_END + 11
assert B_END == FULL_DIM
print(f"\n[블록 경계 확인]  (총 {FULL_DIM}-dim)")
for v, row in zip(variants, X):
    a_nz = int((row[0:A_END] != 0).sum())
    b_nz = int((row[A_END:B_END] != 0).sum())
    print(f"  {v['variant']:<14} A:{a_nz:>2}/11 nonzero | B:{b_nz:>2}/11 nonzero")

# SAV 는 Block A 는 채워지고 Block B 는 전부 0 이어야 함
sav = X[0]
print(f"\nSAV 검증: Block A 채워짐={bool((sav[0:A_END]!=0).any())}, "
      f"Block B 전부 0={bool((sav[A_END:B_END]==0).all())}")

# 저장 (모델 입력 후보 + 사람 확인용)
np.save("rad51c_features_demo.npy", X)
df.round(3).to_csv("rad51c_features_demo.csv", index=False)
print(f"\nsaved: rad51c_features_demo.npy  [M,{FULL_DIM}],  rad51c_features_demo.csv")
print("컬럼 순서:", ", ".join(FULL_COLS[:6]), "...", ", ".join(FULL_COLS[-3:]))
