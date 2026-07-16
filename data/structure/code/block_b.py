"""
block_b.py — RAD51C indel 구조 인코더 (v4: 타입 one-hot 제거, 22-dim)

설계 전제 (사용자 스킴):
  - Block A = 앵커 한 점의 per-residue 구조 feature (indel=시작 위치, SAV=바뀌는 residue).
  - Block B = indel 전용 "재봉합 사건" 인코더. SAV는 전부 0.
  - 변이 타입(sav/del/ins) one-hot은 struct feature에서 제거됨 — meta stream이 담당.

v3 변경 (feature 중복 정리):
  - packing_density 제거: rsasa와 residue-level 상관 -0.92로 사실상 중복.
  - plddt 중복 제거: A_plddt / span_min_plddt / seam_min_plddt 가 0.99+ 겹쳐서
    span 쪽은 span_max2_plddt 하나로, seam_min_plddt 는 삭제.
  - min -> min2/max2: 극단치 robust하게 2개 평균 (span_min2_rsasa, span_max2_plddt).
  - span pooling: 시작점만이 아니라 영향 구간 [p..q] 전체 (양끝 포함) 로 집계.
  - insertion 은 별도 그룹 없이 flank {p,p+1} 을 span 으로 삼아 통합.

구조 feature만 사용 (ESM 임베딩 / LLR / annotation membership 미포함).
의존성: numpy, pandas, biotite  (torch 불필요)
"""

import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

PDB_PATH = "AF-O43502-F1.pdb"
ANNOT_PATH = "rad51c_residue_annotation.csv"
SITE_COLS = ["walker_a", "walker_b", "atp_contact",
             "ssdna_binding", "bcdx2_interface", "cx3_interface"]

# Block A per-residue feature (packing_density 제외 -> 11-dim)
FEAT_COLS = ["plddt", "ss_helix", "ss_sheet", "ss_loop", "rsasa"] \
            + [f"dist_{c}" for c in SITE_COLS]
C_PLDDT, C_HELIX, C_SHEET, C_LOOP, C_RSASA = 0, 1, 2, 3, 4
C_SITE = slice(5, 11)                                    # 6개 site 거리 컬럼

# Block B 11-dim feature 이름 (BlockBEncoder.encode 레이아웃과 1:1 대응)
B_COLS = [
    "junction_strain", "junction_orient_cos", "seam_both_loop",
    "span_min2_rsasa", "span_max2_plddt", "span_frac_ss", "span_min_site_dist",
    "seam_mean_rsasa", "span_external_contacts",
    "del_len_norm", "ins_len_norm",
]

# 최종 벡터: Block A 11 + Block B 11 = 22  (변이 타입 one-hot은 meta stream이 담당)
FULL_COLS = [f"A_{c}" for c in FEAT_COLS] + [f"B_{c}" for c in B_COLS]
FULL_DIM = len(FULL_COLS)                                # 22

CONTACT_CUTOFF = 10.0                                    # Å, 접촉 판정
CA_BOND = 3.8                                            # Å, Cα-Cα virtual bond

MAX_ASA = {
    "ALA":129.0,"ARG":274.0,"ASN":195.0,"ASP":193.0,"CYS":167.0,
    "GLU":223.0,"GLN":225.0,"GLY":104.0,"HIS":224.0,"ILE":197.0,
    "LEU":201.0,"LYS":236.0,"MET":224.0,"PHE":240.0,"PRO":159.0,
    "SER":155.0,"THR":172.0,"TRP":285.0,"TYR":263.0,"VAL":174.0,
}


def _pool_extreme(vals, k=2, largest=False):
    """가장 작은(또는 큰) k개의 평균. 극단치 하나에 덜 민감한 robust min/max."""
    s = np.sort(np.asarray(vals, dtype=float))
    if largest:
        s = s[::-1]
    return float(s[:min(k, len(s))].mean())


# ---------------------------------------------------------------------------
# Block A : per-residue 구조 feature (11-dim) + Cα 좌표 + 거리행렬
# ---------------------------------------------------------------------------
def build_block_a(pdb_path=PDB_PATH, annot_path=ANNOT_PATH):
    """Returns dict: positions [N], feats [N,11], ca_coords [N,3], dmat [N,N], df."""
    f = pdb.PDBFile.read(pdb_path)
    arr = f.get_structure(model=1, extra_fields=["b_factor"])   # b_factor = pLDDT
    arr = arr[struc.filter_amino_acids(arr)]

    ca = arr[arr.atom_name == "CA"]
    res_ids, res_names, n = ca.res_id, ca.res_name, len(ca.res_id)
    ca_coords = ca.coord.copy()

    # 접촉/packing 계산용 대표 좌표: CB (없으면 CA)
    rep = ca_coords.copy()
    cb = arr[arr.atom_name == "CB"]
    cb_map = {rid: c for rid, c in zip(cb.res_id, cb.coord)}
    for i, rid in enumerate(res_ids):
        if rid in cb_map:
            rep[i] = cb_map[rid]

    plddt = ca.b_factor
    sse = struc.annotate_sse(arr)
    ss_helix = (sse == "a").astype(np.float32)
    ss_sheet = (sse == "b").astype(np.float32)
    ss_loop  = (sse == "c").astype(np.float32)

    res_sasa = struc.apply_residue_wise(arr, struc.sasa(arr), np.nansum)
    rsasa = np.clip([res_sasa[i] / MAX_ASA.get(res_names[i], 200.0) for i in range(n)], 0, 1)

    dmat = np.linalg.norm(rep[:, None, :] - rep[None, :, :], axis=-1)
    packing = (dmat < CONTACT_CUTOFF).sum(1) - 1          # 사람용 CSV 참고 컬럼 (feature엔 미포함)

    df = pd.DataFrame({
        "position": res_ids, "aa": res_names, "plddt": np.round(plddt, 2),
        "ss_helix": ss_helix.astype(int), "ss_sheet": ss_sheet.astype(int),
        "ss_loop": ss_loop.astype(int),
        "rsasa": np.round(rsasa, 4), "packing_density": packing,
    })

    annot = pd.read_csv(annot_path)
    pos_to_idx = {int(p): i for i, p in enumerate(res_ids)}
    for col in SITE_COLS:
        site_pos = annot.loc[annot[col] == 1, "position"].astype(int).tolist()
        idxs = [pos_to_idx[p] for p in site_pos if p in pos_to_idx]
        if not idxs:
            df[f"dist_{col}"] = np.nan
            print(f"[warn] no residues flagged for {col} -> NaN")
        else:
            df[f"dist_{col}"] = np.round(dmat[:, idxs].min(1), 3)

    return {
        "positions": df["position"].to_numpy(dtype=np.int64),
        "feats": df[FEAT_COLS].to_numpy(dtype=np.float32),
        "ca_coords": ca_coords.astype(np.float32),
        "dmat": dmat.astype(np.float32),
        "df": df,
    }


# ---------------------------------------------------------------------------
# Block B : indel 재봉합 사건 -> 11-dim 구조 벡터
# ---------------------------------------------------------------------------
class BlockBEncoder:
    """indel별 11-dim 구조 벡터. SAV는 zeros(11).

    영향 구간 R:  deletion = [p..q] (제거되는 residue),  insertion = {p, p+1} (삽입 지점 flank)
    seam (봉합 flank): deletion = {p-1, q+1},  insertion = {p, p+1}

    레이아웃 (index : feature):
        0  junction_strain         ‖Cα(left)-Cα(right)‖ / 3.8Å  (봉합 가능성; raw gap은 ×3.8)
        1  junction_orient_cos     두 사슬 끝 방향 정렬 cos  [-1,1]
        2  seam_both_loop          seam 둘 다 loop  {0,1}
        3  span_min2_rsasa         구간에서 가장 묻힌 2개 평균 rSASA (낮을수록 손상)
        4  span_max2_plddt         구간에서 가장 단단한 2개 평균 pLDDT/100 (높을수록 손상)
        5  span_frac_ss            구간 (helix+sheet) 비율
        6  span_min_site_dist      구간→기능자리 최소거리 (Å)
        7  seam_mean_rsasa         seam 평균 rSASA (흉터가 묻혔나)
        8  span_external_contacts  제거 구간이 나머지와 맺는 접촉 / L (insertion=0)
        9  del_len_norm            결실 길이 / L
        10 ins_len_norm            삽입 길이 / L
    """
    DIM = 11

    def __init__(self, block_a):
        self.positions = block_a["positions"]
        self.feats = block_a["feats"]
        self.ca = block_a["ca_coords"]
        self.dmat = block_a["dmat"]
        self.pos2row = {int(p): i for i, p in enumerate(self.positions)}
        self.first, self.last = int(self.positions.min()), int(self.positions.max())
        self.L = len(self.positions)

    def _row(self, pos):
        return self.pos2row[int(np.clip(pos, self.first, self.last))]

    def _ca(self, pos):
        return self.ca[self._row(pos)]

    def _orient_cos(self, left, right):
        a = self._ca(left) - self._ca(left - 1)          # left로 들어오는 방향
        b = self._ca(right + 1) - self._ca(right)        # right에서 나가는 방향
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def encode(self, anchor_pos, var_type, del_len=0, ins_len=0):
        vt = var_type.lower()
        v = np.zeros(self.DIM, dtype=np.float32)
        if vt == "sav":
            return v                                     # SAV -> zeros

        p = int(anchor_pos)
        if vt == "del":
            k = max(int(del_len), 1)
            q = p + k - 1
            left, right = p - 1, q + 1
            span_rows = [self._row(x) for x in range(p, q + 1)]   # 제거되는 residue
        elif vt == "ins":
            left, right = p, p + 1                       # 삽입은 p|p+1 사이
            span_rows = [self._row(p), self._row(p + 1)] # 삽입 지점 flank
        else:
            raise ValueError(f"unknown var_type: {var_type}")

        lrow, rrow = self._row(left), self._row(right)
        span = self.feats[span_rows]                     # [k, 11]

        # ① 이음매 기하
        gap = float(np.linalg.norm(self._ca(left) - self._ca(right)))
        v[0] = gap / CA_BOND
        v[1] = self._orient_cos(left, right)
        v[2] = float(self.feats[lrow, C_LOOP] * self.feats[rrow, C_LOOP])

        # ② 영향 구간 profile (양끝 포함 pooling, robust 2개 평균)
        v[3] = _pool_extreme(span[:, C_RSASA], 2, largest=False)   # 가장 묻힌 2개
        v[4] = _pool_extreme(span[:, C_PLDDT], 2, largest=True) / 100.0  # 가장 단단한 2개
        v[5] = float(span[:, [C_HELIX, C_SHEET]].sum(axis=1).mean())
        v[6] = float(np.nanmin(span[:, C_SITE]))

        # ③ 흉터 자리
        v[7] = float((self.feats[lrow, C_RSASA] + self.feats[rrow, C_RSASA]) / 2)

        # ④ 연결성 (제거 구간이 나머지와 맺는 접촉; insertion은 제거가 없어 0)
        if vt == "del":
            mask = (self.dmat[span_rows] < CONTACT_CUTOFF).any(axis=0)
            mask[span_rows] = False
            v[8] = float(mask.sum()) / self.L

        # ⑤ 크기
        v[9] = (int(del_len) / self.L) if vt == "del" else 0.0
        v[10] = (int(ins_len) / self.L) if vt == "ins" else 0.0

        return np.nan_to_num(v, nan=0.0)

    def encode_table(self, variants):
        if isinstance(variants, pd.DataFrame):
            variants = variants.to_dict("records")
        out = np.zeros((len(variants), self.DIM), dtype=np.float32)
        for i, r in enumerate(variants):
            out[i] = self.encode(r.get("anchor_pos", 0), r["var_type"],
                                 del_len=r.get("del_len", 0), ins_len=r.get("ins_len", 0))
        return out

    # -- 조립기: Block A(11) + Block B(11) = 22 --------------------------------
    def block_a_vec(self, anchor_pos):
        """앵커 한 점의 Block A 11-dim (indel=시작 위치, SAV=바뀌는 residue)."""
        return self.feats[self._row(anchor_pos)]

    def encode_full(self, anchor_pos, var_type, del_len=0, ins_len=0):
        """변이 하나 -> 최종 22-dim (FULL_COLS 순서)."""
        vt = var_type.lower()
        a = self.block_a_vec(anchor_pos)                                    # 11
        b = self.encode(anchor_pos, vt, del_len=del_len, ins_len=ins_len)   # 11
        return np.concatenate([a, b]).astype(np.float32)

    def encode_full_table(self, variants):
        """여러 변이 -> [M, 22] 행렬."""
        if isinstance(variants, pd.DataFrame):
            variants = variants.to_dict("records")
        out = np.zeros((len(variants), FULL_DIM), dtype=np.float32)
        for i, r in enumerate(variants):
            out[i] = self.encode_full(r.get("anchor_pos", 0), r["var_type"],
                                      del_len=r.get("del_len", 0), ins_len=r.get("ins_len", 0))
        return out


# ---------------------------------------------------------------------------
# demo / 검증
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ba = build_block_a()
    print(f"Block A: {ba['feats'].shape}")
    ba["df"].to_csv("rad51c_blockA.csv", index=False)

    enc = BlockBEncoder(ba)
    feats, pos = ba["feats"], ba["positions"]
    surface = pos[(feats[:, C_LOOP] == 1) & (feats[:, C_RSASA] > 0.5) & (feats[:, C_PLDDT] > 70)]
    core = pos[(feats[:, C_SHEET] == 1) & (feats[:, C_RSASA] < 0.1) & (feats[:, C_PLDDT] > 90)]
    p_surf = int(surface[len(surface) // 2]) if len(surface) else 20
    p_core = int(core[len(core) // 2]) if len(core) else 100

    iB = {n: i for i, n in enumerate(B_COLS)}
    print(f"\n예시 결실:  surface loop=@{p_surf}   core β-strand=@{p_core}")
    print(f"{'variant':<22}{'strain':>8}{'min2_rsasa':>11}{'max2_plddt':>11}{'ext_ct':>8}{'frac_ss':>8}")
    for tag, ap in [(f"del @{p_surf} (surface)", p_surf), (f"del @{p_core} (core)", p_core)]:
        w = enc.encode(ap, "del", del_len=3)
        print(f"{tag:<22}{w[iB['junction_strain']]:>8.2f}{w[iB['span_min2_rsasa']]:>11.3f}"
              f"{w[iB['span_max2_plddt']]:>11.3f}{w[iB['span_external_contacts']]*enc.L:>8.0f}"
              f"{w[iB['span_frac_ss']]:>8.2f}")

    full = enc.encode_full(258, "sav")
    print(f"\nsav @258:  Block B 전부 0={np.allclose(full[11:22], 0)}")
    print(f"Block B DIM = {enc.DIM}  |  FULL_DIM = {FULL_DIM}")
