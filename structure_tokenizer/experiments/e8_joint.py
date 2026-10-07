"""
e8_joint.py — E8 (안 A): 구조 토큰을 Stage 1 과 **처음부터 같이** 학습하면 보탬이 되는가.

E7 (동결 Stage 1 뒤에 붙이기) 에서는 이득이 없었다. 원인 후보 중 "동결 Stage 1 이 train 을 외워
남은 오차가 작다" 와 "ESM 인코더와 구조를 같이 학습하지 않았다" 를 함께 확인한다.

설계 (2026-10-05, 결과 보기 전 확정)
  모델  : 호준 Stage1Model (unified_reference_delta, W10, 층 33) 을 **호준 설정 그대로** 처음부터 학습.
          b = Stage 1 그대로 (구조 없음)
          c = Stage 1 의 요약 z 를 query 로 구조 토큰 9개 cross-attention → z + o → Stage 1 head
              (2026-09-30 결정 "변이 쪽 query → 구조 9개" 를 Stage 1 안에서 같이 학습)
          d = c 와 같은 모델 + 셔플 구조 (position ↔ 구조 대응을 끊음)
  학습  : 호준 base.yaml / sweep 과 같음 — AdamW lr 1e-4 wd 0.01, batch 32, Huber δ=1 (train 통계로 표준화한
          타깃), grad clip 1.0, val subset 으로 최고 에폭 선택, patience 10, 최대 120 에폭.
          구조 쪽 손잡이는 탐색하지 않고 고정: Qk, n_bins 4, d_s 32, 헤드 4 (튜닝 없음 = 세 팔 모두 같은 예산).
  seed  : 42–46 (호준과 같은 seed 목록). 팔 3 × seed 5 = 15회.
  분할  : 배포 분할, 5885 행 (민선 캐시에 있는 행). train 학습 / val 에폭 선택 / test 한 번.
  판정  : 주 가설 c − d > 0 그리고 c − b > 0 (test, position cluster bootstrap, seed 5개 평균 성능 기준).

ESM 입력: 민선 frozen.pt 의 층 33 (WT 전체 + MUT 창 21칸) 을 호준 정렬 규칙으로 모아 쓴다.
  `--pack` 으로 한 번 만들어 둔 묶음(e8_pack.pt, ~0.35 GB)만 있으면 학습 서버에서는 frozen.pt 가 필요 없다.
  `--check` : 묶음으로 만든 입력에 호준 HF best.pt(seed 44) 를 넣어 val subset 이 호준 기록과 같은지 확인.

실행:  python e8_joint.py --pack                 (Mac: 묶음 만들기)
       python e8_joint.py --check                (입력 검증)
       python e8_joint.py --arm c --seed 42      (학습 1회)
       python e8_joint.py --analyze              (→ results/REPORT_E8.md)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import config as C
from data import load_data, shuffle_structure
from encoders import prepare
from model import StructModel

REPO, FROZEN, HF = C.REPO_ROOT, C.FROZEN_PT, C.STAGE1_HF
PACK = os.path.join(C.HERE, "data", "e8_pack.pt")
OUT = os.path.join(C.RESULTS_DIR, "E8")
sys.path.append(REPO)   # 맨 뒤에: 레포 루트의 model.py·train.py 와 이름이 겹치지 않게

MODE, W, LAYER = "unified_reference_delta", 10, 33
S1_CFG = {"d_esm": 1280, "bottleneck_dim": 32, "layers": [LAYER], "head_hidden": 256, "dropout": 0.1,
          "meta_raw_dim": 10}
TRAIN = {"lr": 1e-4, "weight_decay": 0.01, "batch_size": 32, "huber_delta": 1.0, "clip": 1.0,
         "patience": 10, "max_epochs": 120}
STRUCT = {"n_bins": 4, "d_s": 32}
SEEDS, ARMS = [42, 43, 44, 45, 46], ["b", "c", "d"]


# ======================================================================== 묶음 만들기 (Mac)
def build_pack():
    import pandas as pd
    from src.stage1.alignment import anchor_relative_coordinates, build_global_alignment, select_window
    from src.stage1.dataset import build_cohort
    from src.stage1.metadata import raw_meta_features
    from src.stage1.schema import SLOT_PAD

    p = torch.load(FROZEN, map_location="cpu", weights_only=True)
    li = p["layers"].index(LAYER)
    man = pd.read_csv(os.path.join(REPO, "data/split_manifest.csv"))
    wt_seq = open(os.path.join(REPO, "data/wt_sequence.txt")).read().strip()
    vids = list(p["var_id"])
    have = set(vids)
    rows = {r["var_id"]: r for r in man.to_dict("records") if r["var_id"] in have}
    coh = build_cohort([rows[v] for v in vids], wt_seq)
    assert not coh.skipped and [e["var_id"] for e in coh.supported] == vids
    N, A = len(vids), 2 * W + 1
    wt_idx = torch.full((N, A), -1, dtype=torch.long)
    mut_slot = torch.full((N, A), -1, dtype=torch.long)
    kind = torch.full((N, A), SLOT_PAD, dtype=torch.long)
    wt_pos, mut_pos = torch.zeros(N, A, dtype=torch.long), torch.zeros(N, A, dtype=torch.long)
    ins_rank, anchor = torch.zeros(N, A), torch.zeros(N, A)
    tok_valid = torch.zeros(N, A, dtype=torch.bool)
    meta_raw = np.zeros((N, 10), np.float32)
    for r, e in enumerate(coh.supported):
        win = select_window(build_global_alignment(e["edit"]), W)
        assert len(win) <= A
        slot_of_mut = {int(m): a for a, m in enumerate(p["mut_pos"][r].tolist()) if p["mut_present"][r, a]}
        for a, s in enumerate(win):
            if s.wt_pos is not None:
                wt_idx[r, a] = s.wt_pos - 1
                wt_pos[r, a] = s.wt_pos
            if s.mut_pos is not None:
                mut_slot[r, a] = slot_of_mut[s.mut_pos]          # 민선 창에 없으면 KeyError → 중단
                mut_pos[r, a] = s.mut_pos
            kind[r, a], ins_rank[r, a], tok_valid[r, a] = s.kind, float(s.insertion_rank), True
        anchor[r, :len(win)] = torch.tensor(anchor_relative_coordinates(e["edit"], win), dtype=torch.float32)
        meta_raw[r] = raw_meta_features(e["edit"])
    pack = {"var_id": vids, "split": [rows[v]["split"] for v in vids],
            "edit_type": [e["edit"].edit_type for e in coh.supported],
            "y": torch.tensor([float(rows[v]["z_score_D4_D14"]) for v in vids], dtype=torch.float64),
            "wt_full": p["wt_full"][li].clone(), "mutwin": p["mut_windows"][:, li].clone(),
            "mutwin_row": p["row_to_mut_window"].clone(), "wt_idx": wt_idx, "mut_slot": mut_slot,
            "slot_kind": kind, "wt_pos": wt_pos, "mut_pos": mut_pos, "insertion_rank": ins_rank,
            "anchor_rel_coord": anchor, "token_valid": tok_valid, "meta_raw": torch.from_numpy(meta_raw)}
    os.makedirs(os.path.dirname(PACK), exist_ok=True)
    torch.save(pack, PACK)
    print(f"saved {PACK}: N={N}, mutwin {tuple(pack['mutwin'].shape)}, "
          f"{os.path.getsize(PACK) / 1e9:.2f} GB")


# ======================================================================== 입력 만들기
class Inputs:
    """묶음에서 호준 Stage1Model 이 받는 batch dict 를 바로 만든다 (stage1_collate 와 같은 내용)."""

    def __init__(self, pack, meta_mean, meta_std):
        self.p = pack
        mr = pack["meta_raw"].clone()
        mr[:, 5:] = (mr[:, 5:] - torch.as_tensor(meta_mean)) / torch.as_tensor(meta_std)   # 숫자 5개만 표준화
        self.meta = mr

    def batch(self, idx):
        p = self.p
        wi, ms = p["wt_idx"][idx], p["mut_slot"][idx]
        H_wt = p["wt_full"][wi.clamp(min=0)] * (wi >= 0)[..., None]
        H_mut = p["mutwin"][p["mutwin_row"][idx][:, None], ms.clamp(min=0)] * (ms >= 0)[..., None]
        dv = (wi >= 0) & (ms >= 0)
        delta = torch.where(dv[..., None], H_mut - H_wt, torch.zeros(()))
        tv = p["token_valid"][idx]
        return {"layers": [LAYER], "H_wt": H_wt[:, None], "H_mut": H_mut[:, None], "delta": delta[:, None],
                "wt_present": (wi >= 0).float(), "mut_present": (ms >= 0).float(),
                "delta_valid": dv.float(), "token_valid": tv.float(), "attention_valid": tv.float(),
                "slot_kind": p["slot_kind"][idx], "wt_pos": p["wt_pos"][idx], "mut_pos": p["mut_pos"][idx],
                "anchor_rel_coord": p["anchor_rel_coord"][idx], "insertion_rank": p["insertion_rank"][idx],
                "meta_raw": self.meta[idx]}


def load_pack():
    pack = torch.load(PACK, map_location="cpu", weights_only=False)
    sp = np.array(pack["split"])
    idx = {k: torch.as_tensor(np.flatnonzero(sp == k)) for k in ("train", "val", "test")}
    grp = np.array(["missense" if t == "missense" else "synonymous" if t == "synonymous" else "indel"
                    for t in pack["edit_type"]])
    return pack, idx, grp


def subset_spearman(y, pred, grp):
    from scipy.stats import spearmanr
    r = [spearmanr(y[grp == g], pred[grp == g]).correlation for g in ("missense", "indel")]
    return float(np.mean(r)), r


# ======================================================================== 모델
class Joint(nn.Module):
    def __init__(self, arm, seed, struct_kwargs=None):
        super().__init__()
        from src.stage1.model import build_stage1_model
        self.arm = arm
        self.s1 = build_stage1_model(MODE, S1_CFG, init_seed=seed)
        if arm in ("c", "d"):
            torch.manual_seed(seed)
            self.struct = StructModel(1, head="linear", fusion_dim=128, **struct_kwargs)
            self.q_proj = nn.Linear(128, 128)

    def forward(self, batch, x=None, ss=None, return_weights=False):
        z = self.s1(batch)["z_seq"]                                 # 호준 Stage 1 의 요약 [B,128]
        w = None
        if self.arm in ("c", "d"):
            o, w = self.struct.attend_with_query(x[None], ss[None], self.q_proj(z)[None], True)
            z = z + o[0]
        pred = self.s1.head(z)                                      # 호준 Stage 1 head
        return (pred, w) if return_weights else pred


# ======================================================================== 검증 (호준 best.pt)
def check():
    from src.stage1.checkpoint import build_and_load_stage1
    pack, idx, grp = load_pack()
    model, ck = build_and_load_stage1(os.path.join(HF, MODE, f"W{W}", "shipped_split", "seed44", "best.pt"),
                                      expected_mode=MODE)
    inp = Inputs(pack, ck["meta_scaler"]["mean"], ck["meta_scaler"]["std"])
    va = idx["val"]
    with torch.no_grad():
        pred = torch.cat([model(inp.batch(va[s:s + 256]))["pred"] for s in range(0, len(va), 256)])
    pred = pred.numpy() * ck["y_std"] + ck["y_mean"]
    sc, _ = subset_spearman(pack["y"].numpy()[va], pred, grp[va.numpy()])
    hj = json.load(open(os.path.join(HF, MODE, f"W{W}", "shipped_split", "seed44", "metrics.json")))["best"]["score"]
    print(f"val subset: 묶음 입력 {sc:.6f} vs 호준 기록 {hj:.6f} (차이 {abs(sc - hj):.1e})")
    assert abs(sc - hj) < 1e-4


# ======================================================================== 학습 1회
def run(arm, seed, device="cpu"):
    os.makedirs(OUT, exist_ok=True)
    tag = f"{arm}_seed{seed}"
    if os.path.exists(os.path.join(OUT, tag + ".json")):
        print(f"[{tag}] 이미 있음 — 건너뜀")
        return
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    pack, idx, grp = load_pack()
    y = pack["y"].numpy()
    tr, va, te = idx["train"], idx["val"], idx["test"]
    y_mean, y_std = float(y[tr].mean()), float(y[tr].std())
    num = pack["meta_raw"][tr][:, 5:].numpy()
    m_std = num.std(0)
    m_std[m_std == 0] = 1.0
    inp = Inputs(pack, num.mean(0), m_std)
    yt = torch.as_tensor((y - y_mean) / y_std, dtype=torch.float32)

    # 구조 토큰 입력 (우리 행 순서 → 묶음 행 순서)
    d0 = load_data()
    d = shuffle_structure(d0, seed=C.PROTOCOL["shuffle_seed0"] + seed) if arm == "d" else d0
    row_of = {v: i for i, v in enumerate(d.var_id)}
    drow = np.array([row_of[v] for v in pack["var_id"]])
    sk, X, SS = None, None, None
    if arm in ("c", "d"):
        tr_d = drow[tr.numpy()]
        prep = prepare("Qk", dict(STRUCT), d.cont, d.y, [tr_d])
        X = torch.as_tensor(prep.X[0][drow])                       # [N,8]
        SS = torch.as_tensor(d.ss[drow])
        sk = {"encoder": "ple", "d_s": STRUCT["d_s"], "bins": prep.bins, "extrapolate": False}

    model = Joint(arm, seed, sk).to(device)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    huber = nn.HuberLoss(delta=TRAIN["huber_delta"])
    gen = torch.Generator().manual_seed(seed)

    def predict(rows):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(rows), 256):
                r = rows[s:s + 256]
                out.append(model(inp.batch(r), X[r] if X is not None else None,
                                 SS[r] if SS is not None else None))
        return torch.cat(out).numpy() * y_std + y_mean

    best, best_ep, best_state, hist = -np.inf, 0, None, []
    for ep in range(1, TRAIN["max_epochs"] + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen)]
        run_loss, nb = 0.0, 0
        for s in range(0, len(perm), TRAIN["batch_size"]):
            r = perm[s:s + TRAIN["batch_size"]]
            pred = model(inp.batch(r), X[r] if X is not None else None, SS[r] if SS is not None else None)
            loss = huber(pred, yt[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], TRAIN["clip"])
            opt.step()
            run_loss += float(loss)
            nb += 1
        sc, (mis, ind) = subset_spearman(y[va.numpy()], predict(va), grp[va.numpy()])
        hist.append({"epoch": ep, "train_loss": run_loss / nb, "val_subset": sc, "val_mis": mis, "val_indel": ind})
        if sc > best:                                             # 호준과 같이 min_delta 0
            best, best_ep = sc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= TRAIN["patience"]:
            break
    model.load_state_dict(best_state)
    pv, pt = predict(va), predict(te)
    ts, (tmis, tind) = subset_spearman(y[te.numpy()], pt, grp[te.numpy()])
    extra = {}
    if arm in ("c", "d"):                                         # 구조 토큰 9개 attention (헤드 평균), 전체 행
        model.eval()
        allr = torch.arange(len(y))
        A = []
        with torch.no_grad():
            for s in range(0, len(allr), 256):
                r = allr[s:s + 256]
                _, w = model(inp.batch(r), X[r], SS[r], return_weights=True)
                A.append(w[0].mean(1))                             # [B,H,9] → [B,9]
        extra["attn"] = torch.cat(A).numpy()
    np.savez_compressed(os.path.join(OUT, tag + ".npz"), val_rows=va.numpy(), test_rows=te.numpy(),
                        pred_val=pv, pred_test=pt, **extra)
    n_params = sum(q.numel() for n, q in model.named_parameters()
                   if not n.startswith(("struct.type_query", "struct.type_emb", "struct.out")))
    json.dump({"arm": arm, "seed": seed, "best_epoch": best_ep, "epochs_run": len(hist), "val_subset": best,
               "test_subset": ts, "test_missense": tmis, "test_indel": tind, "n_params": n_params,
               "seconds": round(time.time() - t0, 1), "history": hist},
              open(os.path.join(OUT, tag + ".json"), "w"), indent=1, default=float)
    print(f"[{tag}] best ep {best_ep}/{len(hist)}  val {best:.4f}  test {ts:.4f} "
          f"(mis {tmis:.3f}, indel {tind:.3f})  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.pack:
        build_pack()
    elif a.check:
        check()
    elif a.analyze:
        from e8_analyze import analyze
        analyze(OUT)
    else:
        run(a.arm, a.seed)
