"""
build_dataset.py — 토크나이저 입력(변이별 구조 값 9개)을 레포 안의 입력으로 만든다.

데이터 파일은 레포에 올리지 않는다 (SGE 데이터 엠바고, README_P1.md). 대신 이 스크립트로 누구나
같은 파일을 만들고, sha256 으로 같은 파일인지 확인한다.

입력 (모두 레포 안)
  data/split_manifest.csv                              변이 5,887개, 배포 분할, 라벨
  data/structure/inputs/AF-O43502-F1.pdb               AlphaFold 구조 (RAD51C, 376 잔기)
  data/structure/inputs/rad51c_residue_annotation.csv  기능 부위 표시
  data/structure/code/block_b.py :: build_block_a      잔기별 Block A (pLDDT, ss one-hot 3, rSASA, 부위 거리 6)

출력: structure_tokenizer/data/v2_dataset.csv  (변이 5,887행)
  var_id, split, var_type, type3, anchor_pos, del_len, ins_len, z_score_D4_D14,
  slim_consequence, functional_classification, A_* 구조 값, A_ss_class
  앵커 = 변이 시작 위치(pp). 같은 위치의 변이는 구조 값이 같다 (assert).

실행:  python build_dataset.py      (biotite 필요)
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import numpy as np
import pandas as pd

from tokenizer import SS_CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)                                       # 레포 루트
MANIFEST = os.path.join(R, "data", "split_manifest.csv")
PDB = os.path.join(R, "data", "structure", "inputs", "AF-O43502-F1.pdb")
ANNOT = os.path.join(R, "data", "structure", "inputs", "rad51c_residue_annotation.csv")
BLOCK_B = os.path.join(R, "data", "structure", "code", "block_b.py")
OUT = os.path.join(HERE, "data", "v2_dataset.csv")
# 검증 실험(E0–E8)에 쓴 파일의 sha256. 같으면 그 결과와 같은 입력이다.
EXPECTED_SHA256 = "0b4f7c60aba08b9609a9e4cd2ccd21be58c6b3568addc20a9f3017ce339f381d"
N_ROWS, WT_LEN = 5887, 376

CONS2TYPE = {"missense": "sav", "synonymous": "syn", "codon_deletion": "del",
             "clinical_inframe_deletion": "del", "clinical_inframe_insertion": "ins"}
TYPE4_TO_TYPE3 = {"sav": "missense", "syn": "synonymous", "del": "indel", "ins": "indel"}


def _load_block_b():
    spec = importlib.util.spec_from_file_location("block_b", BLOCK_B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    bb = _load_block_b()
    m = pd.read_csv(MANIFEST)
    m = m[m.slim_consequence.isin(CONS2TYPE)].copy().reset_index(drop=True)
    m["var_type"] = m.slim_consequence.map(CONS2TYPE)
    m["type3"] = m.var_type.map(TYPE4_TO_TYPE3)
    m["anchor_pos"] = m.pp.astype(int)
    dlen = WT_LEN - m.mut_seq.str.len()
    m["del_len"] = dlen.clip(lower=0).astype(int)
    m["ins_len"] = (-dlen).clip(lower=0).astype(int)

    ba = bb.build_block_a(PDB, ANNOT)
    pos2row = {int(p): i for i, p in enumerate(ba["positions"])}
    A = ba["feats"][np.array([pos2row[p] for p in m.anchor_pos])]
    out = pd.DataFrame(A, columns=[f"A_{c}" for c in bb.FEAT_COLS])
    oh = out[["A_ss_helix", "A_ss_sheet", "A_ss_loop"]].to_numpy()
    assert (oh.sum(1) == 1).all(), "ss one-hot 이 정확히 하나가 아님"
    out["A_ss_class"] = [SS_CLASSES[i] for i in oh.argmax(1)]

    meta = m[["var_id", "split", "var_type", "type3", "anchor_pos", "del_len", "ins_len",
              "z_score_D4_D14", "slim_consequence", "functional_classification"]]
    df = pd.concat([meta, out], axis=1)
    assert len(df) == N_ROWS and df.isna().sum().sum() == 0
    acols = [c for c in df.columns if c.startswith("A_")]
    assert df.groupby("anchor_pos")[acols].nunique().max().max() == 1, "같은 위치에서 구조 값이 다름"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    print(f"saved {OUT}: {len(df)} rows, sha256 {sha[:12]}…")
    if sha == EXPECTED_SHA256:
        print("검증 실험에 쓴 파일과 같습니다.")
    else:
        print("주의: 검증 실험에 쓴 파일과 sha256 가 다릅니다 (입력이나 라이브러리 버전 확인).")


if __name__ == "__main__":
    main()
