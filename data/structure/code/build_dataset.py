"""
build_dataset.py — split_manifest.csv -> 변이별 22-dim 구조 feature 표

  scope: SAV(missense) + synonymous + in-frame indel(deletion/insertion).
         synonymous는 var_type="sav"로 매핑 (WT와 서열 동일 -> Block A는 해당
         위치의 WT 구조, Block B는 SAV와 동일하게 전부 0).
  파서:  pp -> anchor_pos, slim_consequence -> var_type,
         mut_seq 길이 - 376 -> del_len / ins_len
  출력:  rad51c_struct_features.csv  (var_id, split, label + 22 feature)
         rad51c_X.npy [N,{FULL_DIM}], rad51c_y.npy [N], rad51c_meta.csv
"""

import numpy as np
import pandas as pd
from block_b import build_block_a, BlockBEncoder, FULL_COLS, FULL_DIM


def spearman(a, b):
    """scipy 없이 Spearman rank correlation."""
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])

WT_LEN = 376
CONS2TYPE = {
    "missense": "sav",
    "synonymous": "sav",
    "codon_deletion": "del",
    "clinical_inframe_deletion": "del",
    "clinical_inframe_insertion": "ins",
}

m = pd.read_csv("split_manifest.csv")
m["dlen"] = WT_LEN - m["mut_seq"].str.len()          # >0 deletion, <0 insertion

# scope 필터 (여전히 stop_gained/frameshift/splice 등은 제외)
m = m[m["slim_consequence"].isin(CONS2TYPE)].copy().reset_index(drop=True)
m["var_type"] = m["slim_consequence"].map(CONS2TYPE)
m["anchor_pos"] = m["pp"].astype(int)
m["del_len"] = m["dlen"].clip(lower=0).astype(int)
m["ins_len"] = (-m["dlen"]).clip(lower=0).astype(int)

print(f"scope 변이: {len(m)}")
print(m["var_type"].value_counts().to_string())

# 구조 feature 조립
enc = BlockBEncoder(build_block_a())
X = enc.encode_full_table(m[["anchor_pos", "var_type", "del_len", "ins_len"]])
assert X.shape == (len(m), FULL_DIM) and not np.isnan(X).any()
print(f"\nfeature 행렬 X: {X.shape}   NaN={np.isnan(X).any()}")

# 저장 (모델용 + 사람용)
feat_df = pd.DataFrame(X, columns=FULL_COLS)
meta = m[["var_id", "split", "var_type", "anchor_pos", "del_len", "ins_len",
          "z_score_D4_D14", "functional_classification"]].reset_index(drop=True)
out = pd.concat([meta, feat_df.round(4)], axis=1)
out.to_csv("rad51c_struct_features.csv", index=False)
np.save("rad51c_X.npy", X)
np.save("rad51c_y.npy", m["z_score_D4_D14"].to_numpy(dtype=np.float32))
meta.to_csv("rad51c_meta.csv", index=False)
print(f"saved: rad51c_struct_features.csv, rad51c_X.npy [N,{FULL_DIM}], rad51c_y.npy, rad51c_meta.csv")

# split x type 교차표
print("\n[split x var_type]")
print(pd.crosstab(m["split"], m["var_type"]))

# --- sniff test: 결실의 구조 disruption이 z_score와 상관있나 ---
# (z_score가 음수일수록 depleted = 기능 손상. 구조 disruption↑ 이면 z↓ 기대)
dele = m[m.var_type == "del"].copy()
Xd = enc.encode_full_table(dele[["anchor_pos", "var_type", "del_len", "ins_len"]])
z = dele["z_score_D4_D14"].to_numpy()
ok = ~np.isnan(z)
print(f"\n[결실 {ok.sum()}개: 구조 feature vs z_score Spearman ρ]  (음수 ρ = 예상 방향)")
for name in ["B_junction_strain", "B_span_external_contacts", "B_span_min2_rsasa",
             "B_seam_mean_rsasa", "A_plddt"]:
    col = Xd[:, FULL_COLS.index(name)]
    rho = spearman(col[ok], z[ok])
    print(f"  {name:26s} ρ={rho:+.3f}")
