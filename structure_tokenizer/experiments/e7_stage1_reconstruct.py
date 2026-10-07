"""E7 준비 — 호준 Stage 1 체크포인트(HF)로 변이별 예측·표현을 복원한다.

HF `DeepCANCERsite/stage1-checkpoints`에는 best.pt와 지표만 있고 변이별 예측이 없다.
그래서 호준의 코드(deepCANCERsite main 9d22d8e, src/stage1)를 그대로 쓰되,
ESM 층 33 hidden state는 민선 handoff의 frozen.pt(같은 repo-native ESM-2 650M)에서 가져온다.

frozen.pt에는 WT 전체(376칸)와 MUT 창 21칸(±10)만 있다. 그래서
- W=10 이하만 복원할 수 있다 (W=20은 불가).
- MUT의 창 밖 칸은 NaN으로 채운다. 호준 코드가 창 밖 칸을 읽으면 NaN이 생기고 assert에 걸린다.

검증: 복원한 예측으로 계산한 val 지표가 호준 metrics.json의 best val과 같아야 한다.
val은 두 쪽 모두 881개로 같다. test는 호준 882개, 민선 881개다(Val351dup 1개가 빠짐).

출력: results/E7/stage1_{mode}_W{w}_seed{s}.npz
  var_id, split, pred(원래 z 단위), z_seq(128, Stage 2 query 후보), attn(창 토큰 + 메타 토큰 가중치)
  K, V [N, 2W+2, 128] float16 + tok_valid [N, 2W+2]: Stage 1 의 attention 토큰 (README §9 의 Stage 2 입력).
    창 토큰은 앞 칸부터, 메타 토큰은 항상 마지막 칸(2W+1). 창이 짧은 변이(말단)는 사이가 빈 칸(valid=0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as C  # noqa: E402

REPO = Path(C.REPO_ROOT)
HF = Path(C.STAGE1_HF)
FROZEN = Path(C.FROZEN_PT)
OUT = Path(__file__).resolve().parent / "results" / "E7"
sys.path.append(str(REPO))   # 맨 뒤에: 레포 루트의 model.py·train.py 와 이름이 겹치지 않게

from src.stage1.checkpoint import build_and_load_stage1  # noqa: E402
from src.stage1.dataset import Stage1Dataset, build_cohort, make_collate_fn  # noqa: E402
from src.stage1.metadata import MetaScaler  # noqa: E402

LAYER = 33


def group_of(edit_type: str) -> str:
    """src/stage1/engine.py 의 group_of 와 같음. engine 은 레포 루트의 train.py 를 불러와
    우리 train.py 와 이름이 겹치므로 import 하지 않는다."""
    return edit_type if edit_type in ("missense", "synonymous") else "indel"


class FrozenWindowCache:
    """RawStage1Cache와 같은 get_wt/get_mut 계약. MUT는 창 칸만 채우고 나머지는 NaN."""

    def __init__(self, p: dict, mut_len: dict[str, int]):
        li = p["layers"].index(LAYER)
        self.hidden_dim = p["wt_full"].shape[-1]
        self.wt = p["wt_full"][li]                                   # (376, D)
        self.mut_len = mut_len
        self.mut_entries = {}                                        # var_id -> (창 MUT 위치, (n, D))
        for r, vid in enumerate(p["var_id"]):
            keep = p["mut_present"][r]
            win = p["mut_windows"][p["row_to_mut_window"][r], li]    # (21, D)
            self.mut_entries[vid] = (p["mut_pos"][r][keep] - 1, win[keep].clone())

    def get_wt(self, layer):
        assert layer == LAYER
        return self.wt

    def get_mut(self, var_id, layer):
        # 전체 길이를 NaN으로 만들고 창 칸만 채운다 (변이 하나씩 그때그때 만들어 메모리를 아낌)
        assert layer == LAYER
        idx, win = self.mut_entries[var_id]
        H = torch.full((self.mut_len[var_id], self.hidden_dim), float("nan"))
        H[idx] = win
        return H


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def metrics(y, pred, groups):
    out = {"spearman": spearman(y, pred), "n": len(y), "by_group": {}}
    for g in ("missense", "synonymous", "indel"):
        m = groups == g
        out["by_group"][g] = spearman(y[m], pred[m])
    out["subset"] = float(np.mean([out["by_group"]["missense"], out["by_group"]["indel"]]))
    return out


def run(mode, W, seed, entries, cache):
    run_dir = HF / mode / f"W{W}" / "shipped_split" / f"seed{seed}"
    model, ck = build_and_load_stage1(run_dir / "best.pt", expected_mode=mode)
    assert ck["cfg"]["layers"] == [LAYER] and ck["cfg"]["window_radius"] == W
    assert ck["reference"]["wt_hash"] == "2d432f1bb00a6ae7"
    scaler = MetaScaler(mean=np.array(ck["meta_scaler"]["mean"]), std=np.array(ck["meta_scaler"]["std"]))
    ds = Stage1Dataset(entries, cache, W, [LAYER], meta_scaler=scaler)
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=make_collate_fn(mode))

    preds, zs, attns, nslots, Ks, Vs, TV = [], [], [], [], [], [], []
    with torch.no_grad():
        for b in dl:
            # 창 밖 칸을 읽었다면 NaN이 섞인다 → 여기서 잡는다
            assert torch.isfinite(b["H_mut"]).all(), "호준 창이 민선 21칸 밖을 읽었음"
            o = model(b, return_extras=True)
            preds.append(o["pred"].numpy()); zs.append(o["z_seq"].numpy())
            w = o["attn_weights"].numpy()
            pad = np.full((w.shape[0], 2 * W + 2), np.nan, dtype=np.float32)  # 창 최대 2W+1 + 메타 1
            A = b["H_wt"].shape[2]
            pad[:, :A] = w[:, :A]; pad[:, -1] = w[:, A]                  # 메타 토큰은 마지막 칸에
            attns.append(pad); nslots.append(b["token_valid"].sum(1).numpy())
            # K/V 를 고정 칸 수(2W+2)로 맞춘다: 창 [:A] → [:A], 메타(A번째) → 마지막 칸
            T = 2 * W + 2
            k = torch.zeros(w.shape[0], T, o["K"].shape[-1]); v = torch.zeros_like(k)
            tv = torch.zeros(w.shape[0], T, dtype=torch.bool)
            k[:, :A], v[:, :A], tv[:, :A] = o["K"][:, :A], o["V"][:, :A], o["attention_valid"][:, :A] > 0
            k[:, -1], v[:, -1], tv[:, -1] = o["K"][:, A], o["V"][:, A], True
            assert o["meta_token_index"] == A
            Ks.append(k.half().numpy()); Vs.append(v.half().numpy()); TV.append(tv.numpy())
    pred = np.concatenate(preds) * ck["y_std"] + ck["y_mean"]
    extra = dict(K=np.concatenate(Ks), V=np.concatenate(Vs), tok_valid=np.concatenate(TV))
    return pred, np.concatenate(zs), np.concatenate(attns), np.concatenate(nslots), run_dir, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="unified_reference_delta")
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    assert args.W <= 10, "frozen.pt 창은 ±10이라 W≤10만 복원 가능"

    man = pd.read_csv(REPO / "data/split_manifest.csv")
    wt_seq = (REPO / "data/wt_sequence.txt").read_text().strip()
    p = torch.load(FROZEN, map_location="cpu", weights_only=True)
    have = set(p["var_id"])
    rows = [r for r in man.to_dict("records") if r["var_id"] in have]
    coh = build_cohort(rows, wt_seq)
    assert not coh.skipped and len(coh.supported) == len(p["var_id"]) == 5885
    mut_len = {r["var_id"]: len(r["mut_seq"]) for r in rows}
    cache = FrozenWindowCache(p, mut_len)
    del p

    entries = coh.supported
    vid = np.array([e["var_id"] for e in entries])
    split = np.array([e["row"]["split"] for e in entries])
    y = np.array([float(e["row"]["z_score_D4_D14"]) for e in entries])
    grp = np.array([group_of(e["edit"].edit_type) for e in entries])

    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for s in args.seeds:
        pred, z, attn, ns, run_dir, extra = run(args.mode, args.W, s, entries, cache)
        mv = metrics(y[split == "val"], pred[split == "val"], grp[split == "val"])
        mt = metrics(y[split == "test"], pred[split == "test"], grp[split == "test"])
        hv = json.load(open(run_dir / "metrics.json"))["best"]["val"]
        ht = json.load(open(run_dir / "test_metrics.json"))
        hv_subset = float(np.mean([hv["by_group"]["missense"], hv["by_group"]["indel"]]))
        d_val = abs(mv["subset"] - hv_subset)
        report.append(dict(seed=s, val_n=mv["n"], val_subset_recon=mv["subset"], val_subset_hojun=hv_subset,
                           val_absdiff=d_val, val_mis_recon=mv["by_group"]["missense"], val_mis_hojun=hv["by_group"]["missense"],
                           test_n=mt["n"], test_subset_recon=mt["subset"], test_subset_hojun=ht["subset"],
                           test_mis_recon=mt["by_group"]["missense"], test_mis_hojun=ht["by_group"]["missense"]))
        print(json.dumps(report[-1]))
        assert mv["n"] == hv["n"] == 881
        np.savez_compressed(OUT / f"stage1_{args.mode}_W{args.W}_seed{s}.npz",
                            var_id=vid, split=split, group=grp, y=y, pred=pred, z_seq=z, attn=attn, n_slots=ns, **extra)
    pd.DataFrame(report).to_csv(OUT / f"reconstruct_check_{args.mode}_W{args.W}.csv", index=False)


if __name__ == "__main__":
    main()
