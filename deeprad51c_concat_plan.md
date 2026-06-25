# DeepRAD51C — Concat 단계 설계 (step by step, high detail)

기준: deepCANCERsite repo 실제 코드 (`dataset.py`, `src/embeddings/*`, `structure_code.py`, `baseline_llr.py`).
이 문서는 5개 stream을 **하나의 모델 입력으로 합치는(concat)** 단계를 처음부터 끝까지 설계한다.

---

## 0. 먼저 — 업로드한 그림과 실제 코드가 다른 곳 (concat 전에 정리)

| | 그림 | 실제 코드 | 조치 |
|---|---|---|---|
| struct | 32-dim | **12-dim** (6 구조 + 6 거리) | concat 차원 12로 |
| PLLR | 1-dim | **없음.** `llr` 스칼라(masked-marginal)만 존재 | 슬롯은 `llr`로 채우거나 PLLR 나중 계산 |
| conservation | 채워짐 | **전부 0** | 0으로 진행 or ConSurf로 채움 |
| struct↔annot | 독립 | struct의 `dist_*`가 annot site에서 파생 | annot 바뀌면 struct 재생성 |

**따라서 실제 concat 차원 = `256(diff) + 1(llr) + 12(struct) + 10(annot) + 11(meta) = 290`** (그림의 310 아님).

---

## Step 1. Stream contract를 상수로 고정한다

가장 먼저 "각 stream이 뭘, 몇 차원, 어떤 key로, 어디서 오는지"를 **코드 상수 하나**로 박는다.
이게 흔들리면 weight와 feature가 어긋나 silent bug가 난다.

```python
# feature_spec.py
FEATURE_SPEC = [
    # name        dim  join_key   source_file                       kind
    ("diff_emb",  256, "var_id",  "diff_emb.pt",                    "dense"),   # 이미 LayerNorm됨
    ("llr",         1, "var_id",  "data/baseline_llr.csv",          "cont"),    # PLLR 대체 슬롯
    ("struct",     12, "pp",      "rad51c_blockA.pt",               "mixed"),   # 3 binary + 9 cont
    ("annot",      10, "pp",      "data/rad51c_residue_annotation.csv","mixed"),# 7 binary + 3 (domain) + 1 cont
    ("meta",       11, "inline",  "dataset.build_metadata",         "mixed"),   # 8 binary + 3 cont
]
CONCAT_DIM = sum(d for _, d, *_ in FEATURE_SPEC)   # = 290

# binary vs continuous 컬럼을 명시적으로 분류 (Step 4 정규화에 사용)
BINARY_IDX = {                       # stream 내부 인덱스 (0-based)
    "struct": [1, 2, 3],             # ss_helix, ss_sheet, ss_loop
    "annot":  [0,1,2, 3,4,5, 6,7,8], # domain×3 + walker_a/b + atp + ssdna/bcdx2/cx3
    "meta":   [0,1,2,3,4,5,6,7],     # variant-type one-hot ×8
}
CONT_IDX = {
    "llr":    [0],
    "struct": [0, 4,5, 6,7,8,9,10,11],   # plddt, rsasa, packing, dist_*×6
    "annot":  [9],                       # conservation
    "meta":   [8, 9, 10],                # pp_norm, cds_norm, len_norm
}   # diff_emb은 이미 LayerNorm → 정규화 대상 아님
```

> `diff_emb`은 BottleneckMLP 끝에서 LayerNorm을 거쳐 이미 scale이 잡혀 있다 → **다시 정규화 금지.**
> 나머지 raw 연속값만 Step 4에서 표준화한다.

---

## Step 2. 각 stream을 manifest 행 순서로 정렬해 캐싱한다

concat은 "변이마다 5개 조각을 모으는" 작업이다. join key가 stream마다 다르다:
**diff_emb·llr = `var_id`로, struct·annot = `pp`로, meta = inline.** Dataset이 둘 다 갖고 있어 OK.

precompute를 한 번 돌려서 `split_manifest.csv` **행 순서와 1:1 정렬된** 텐서로 저장한다.

```python
# build_feature_cache.py
import pandas as pd, numpy as np, torch
from build_annotation import load_sequence, build_residue_annotation, make_lookup
from dataset import build_metadata

def build_feature_cache(manifest="data/split_manifest.csv"):
    m = pd.read_csv(manifest)
    n = len(m)

    # --- var_id 기반 dense/scalar ---
    diff = torch.load("diff_emb.pt")          # {var_id: (256,)}  ← 호준이 덤프해야 (Step 8)
    llr_df = pd.read_csv("data/baseline_llr.csv").set_index("var_id")["llr"]

    # --- pp 기반 lookup ---
    blob = torch.load("rad51c_blockA.pt")     # {positions:[376], features:[376,12]}
    struct_by_pos = {int(p): blob["features"][i] for i, p in enumerate(blob["positions"].tolist())}
    _, get_annotation = make_lookup(build_residue_annotation())

    # --- manifest 순서대로 채우기 + 결측 mask (Step 3) ---
    X_diff   = torch.zeros(n, 256)
    X_llr    = torch.zeros(n, 1)
    X_struct = torch.zeros(n, 12)
    X_annot  = torch.zeros(n, 10)
    X_meta   = torch.zeros(n, 11)
    valid    = torch.zeros(n, 3)              # [diff_valid, llr_valid, annot_valid]

    for i, row in m.iterrows():
        vid, pp = row["var_id"], row["pp"]
        # diff_emb
        if vid in diff:
            X_diff[i] = diff[vid];            valid[i, 0] = 1.0
        # llr (missense에만 정의 / syn·indel은 0 + mask off)
        if vid in llr_df.index and not pd.isna(llr_df[vid]):
            X_llr[i, 0] = float(llr_df[vid]); valid[i, 1] = 1.0
        # struct / annot (pp lookup)
        if not pd.isna(pp) and int(pp) in struct_by_pos:
            X_struct[i] = struct_by_pos[int(pp)]
        X_annot[i] = torch.from_numpy(get_annotation(pp))
        if not pd.isna(pp): valid[i, 2] = 1.0
        # meta (inline)
        X_meta[i] = torch.from_numpy(build_metadata(row))

    torch.save({"diff": X_diff, "llr": X_llr, "struct": X_struct,
                "annot": X_annot, "meta": X_meta, "valid": valid,
                "split": m["split"].values}, "feature_cache.pt")
    print("cached:", n, "variants")
```

---

## Step 3. 결측/미정의 값을 math 전에 처리한다 (지금 안 돼 있음 — 중요)

concat에 NaN이 하나라도 들어가면 forward에서 NaN이 전파돼 **전체 loss가 NaN**이 된다. 미리 막는다.

- **diff_emb 없는 변이** (호준이 아직 안 만든 var_id) → 0으로 두고 `diff_valid=0`. 학습 시 이 행은 제외하거나 mask.
- **llr 미정의** (synonymous=0, indel=ref/alt 없음) → 0 + `llr_valid=0`. "진짜 0(중립)"과 "계산 불가"를 구분.
- **struct dist_\* future NaN** → 지금은 site가 채워져 NaN 없지만, site 비면 NaN 재발 → impute(중앙값)+경고.
- **conservation = 0** → 상수라 무해하지만 **신호 0**. ConSurf로 채우거나, 안 채우면 ablation에서 자동으로 "기여 없음"으로 드러남.
- **명시적 mask 3개** (`diff_valid, llr_valid, annot_valid`)를 feature에 포함 → 모델이 결측을 인지.

> 핵심 원칙: **결측은 "0 + valid 플래그"로**, 절대 NaN/임의값으로 두지 않는다.

---

## Step 4. Scale 정규화 — train split으로만 fit (leakage 금지)

concat 벡터엔 스케일이 천차만별인 게 섞인다: `plddt`(0–100), `rsasa`(0–1), `packing`(정수 개수),
`dist_*`(Å, 0–50+), `llr`(≈ −20–5), `meta`(0–1), binary(0/1). 표준화 없으면 큰 스케일이 지배한다.

```python
# fit_scaler.py
import torch
from sklearn.preprocessing import StandardScaler
from feature_spec import CONT_IDX

cache = torch.load("feature_cache.pt")
train = cache["split"] == "train"          # ★ train 행으로만 fit

scalers = {}
for stream, idxs in CONT_IDX.items():
    X = cache[stream][train][:, idxs].numpy()
    sc = StandardScaler().fit(X)            # mean/std from TRAIN ONLY
    scalers[stream] = (idxs, sc)
torch.save(scalers, "scalers.pt")           # 추론 때 동일 적용

# 적용 (train/val/test 모두 동일 scaler)
for stream, (idxs, sc) in scalers.items():
    cache[stream][:, idxs] = torch.tensor(sc.transform(cache[stream][:, idxs].numpy()), dtype=torch.float32)
torch.save(cache, "feature_cache_scaled.pt")
```

- **연속값만** 표준화 (`CONT_IDX`), **binary는 그대로**.
- **`diff_emb`은 건드리지 않음** (이미 LayerNorm).
- scaler는 반드시 **train만으로 fit** → val/test는 transform만.

---

## Step 5. 합치고(concat) → input projection

정해진 순서로 옆으로 이어붙인 뒤, 그림의 "Input projection"을 적용한다.

```python
import torch, torch.nn as nn

class InputProjection(nn.Module):
    """concat(290) → 256, LayerNorm으로 이종 feature 스케일 한 번 더 정렬."""
    def __init__(self, in_dim=290, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )
    def forward(self, x):           # x: (B, 290)
        return self.net(x)          # (B, 256) → Fusion MLP(residual×3)로
```

concat 순서(고정): `[diff(256), llr(1), struct(12), annot(10), meta(11), valid(3)]`.
(valid 3개 mask를 끝에 붙이면 293이 됨 — mask 포함 권장. 미포함이면 290.)

```python
def assemble_batch(cache, rows):
    parts = [cache["diff"][rows], cache["llr"][rows], cache["struct"][rows],
             cache["annot"][rows], cache["meta"][rows], cache["valid"][rows]]
    return torch.cat(parts, dim=1)      # (B, 293)
```

> **per-stream encoder(예: annot 10→16, struct 12→32)는 기본 끄고 시작.** 데이터가 작아(5,887)
> raw concat이 더 안정적일 때가 많다. encoder는 Step 6의 ablation 토글로 비교.
> 단 **`diff_emb`은 이미 BottleneckMLP로 인코딩됐으니 추가 encoder 금지**(이중 인코딩).

---

## Step 6. Ablation을 처음부터 내장한다 (P4 필수)

선배 코멘트 두 개("external score=LLR도 ablation", "annotation이 baseline 이기나")를 코드로 강제한다.
stream을 켜고 끌 수 있게 만들면 input projection의 `in_dim`만 바뀐다.

```python
STREAM_DIMS = {"diff":256, "llr":1, "struct":12, "annot":10, "meta":11, "valid":3}

def assemble(cache, rows, use):     # use: set 예) {"diff","struct","annot","meta","valid"}
    order = ["diff","llr","struct","annot","meta","valid"]
    parts = [cache[k][rows] for k in order if k in use]
    return torch.cat(parts, dim=1)

def in_dim(use): return sum(STREAM_DIMS[k] for k in use)
```

**돌려야 할 ablation (must-beat = ρ0.59 / AUC0.92 위에서):**

| 구성 | 묻는 것 |
|---|---|
| diff only | ESM 차이만으로 얼마나? |
| diff + llr | LLR(external) 효과 — 선배 지적 |
| diff + struct | 구조 효과 |
| diff + annot | **annotation이 자리값 하나** (네 핵심 결과) |
| diff + struct + annot + meta (− llr) | external 없이 |
| 전체 | 상한 |

---

## Step 7. collate_fn / DataLoader 연결

학습 루프에선 label mask까지 처리한다 (regression label은 NaN 가능, cls는 −1=unknown).

```python
def collate(batch_rows, cache, use):
    x = assemble(cache, batch_rows, use)                       # (B, in_dim)
    reg = cache_labels_reg[batch_rows]                         # (B,) NaN 가능
    cls = cache_labels_cls[batch_rows]                         # (B,) −1 가능
    reg_mask = ~torch.isnan(reg)                               # regression loss는 라벨 있는 것만
    cls_mask = cls >= 0
    return x, reg, cls, reg_mask, cls_mask
```

---

## Step 8. 실행 순서 (의존성 포함)

1. **호준(P3):** 현재는 `diff_pool`/`bottleneck` 모듈만 있고 변이별 캐시가 없음 →
   전 변이(5,887)에 대해 `diff_emb`를 `{var_id: (256,)}`로 **`diff_emb.pt` 덤프**.
   - ⚠️ indel/codon_deletion은 `diff_pool`이 WT/MUT window를 같은 인덱스로 잡아 **misalign**.
     이 subset은 alignment-aware diff 또는 global-mean stream 필요 (지금은 근사). missense부터 확정.
2. **LLR vs PLLR 슬롯 결정:** 지금은 `llr` 스칼라 사용(missense 정의). indel 커버하려면 PLLR 추후 계산.
3. **민선(P1):** conservation 채우기(ConSurf) 또는 0으로 진행(ablation에서 드러남).
4. `build_feature_cache()` → `fit_scaler`(train-only) → `feature_cache_scaled.pt`.
5. **호준(P3):** `InputProjection(in_dim)` + Fusion MLP(residual×3) + 2 head 연결, ablation use-set로 학습.

## 한눈에 — concat 체크리스트
- [ ] 차원 290(+mask 3=293), struct=12, PLLR 슬롯=llr 확정
- [ ] join: diff/llr=var_id, struct/annot=pp, meta=inline
- [ ] 결측 → 0 + valid mask 3개 (NaN 절대 금지)
- [ ] 연속값만 train-only 표준화, diff_emb·binary 제외
- [ ] concat 순서 상수 고정
- [ ] InputProjection(in_dim→256)+LN+GELU
- [ ] stream on/off ablation 내장
- [ ] indel diff_emb misalign은 알려진 한계로 표기
