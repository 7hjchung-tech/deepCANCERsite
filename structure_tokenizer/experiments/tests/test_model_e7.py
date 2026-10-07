"""E7 Stage 2 모델과 Stage 1 출력 파일의 가정 검사."""
import os

import numpy as np
import pytest
import torch

from encoders import quantile_bins  # noqa: F401  (import 경로 확인용)
from model import StructModel
from model_e7 import Stage2Model

NPZ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "E7",
                   "stage1_unified_reference_delta_W10_seed44.npz")


def _bins(M):
    T = 4
    lo = np.tile(np.linspace(0, 0.75, T), (M, 8, 1)).astype(np.float32)
    return {"lo": lo, "width": np.full((M, 8, T), 0.25, np.float32),
            "valid": np.ones((M, 8, T), bool), "first": np.eye(T, dtype=bool)[0][None, None].repeat(M, 0).repeat(8, 1),
            "last": np.eye(T, dtype=bool)[-1][None, None].repeat(M, 0).repeat(8, 1),
            "n_bins": np.full((M, 8), T)}


def _fake_s1(N=40, T=6, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    tv = torch.ones(N, T, dtype=torch.bool)
    tv[: N // 2, 2:4] = False                                     # 말단 변이처럼 빈 칸
    return {"p1n": torch.randn(N, generator=g), "z": torch.randn(N, D, generator=g),
            "K": torch.randn(N, T, D, generator=g).half(), "V": torch.randn(N, T, D, generator=g).half(),
            "tok_valid": tv}


def _inputs(M, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(M, N, 8, generator=g)
    ss = torch.randint(0, 3, (N,), generator=g)
    typ = torch.randint(0, 3, (N,), generator=g)
    rows = torch.arange(N)[None].repeat(M, 1)
    return x, ss, typ, rows


def test_queries_T1_equals_single_query():
    M, B = 2, 5
    torch.manual_seed(0)
    sm = StructModel(M, "ple", 16, "linear", bins=_bins(M), fusion_dim=32)
    x, ss = torch.rand(M, B, 8), torch.randint(0, 3, (M, B))
    q = torch.randn(M, B, 32)
    o1, w1 = sm.attend_with_query(x, ss, q, True)
    oT, wT = sm.attend_with_queries(x, ss, q[:, :, None], True)
    assert torch.allclose(o1, oT[:, :, 0], atol=1e-6) and torch.allclose(w1, wT[:, :, 0], atol=1e-6)


@pytest.mark.parametrize("query", ["z", "tok"])
@pytest.mark.parametrize("arm", ["b", "c"])
def test_zero_init_starts_at_stage1(query, arm):
    """잔차 마지막 층이 0 → 학습 전 예측 = Stage 1 예측 그대로."""
    M, N = 3, 40
    S1 = _fake_s1(N)
    sk = {"encoder": "ple", "d_s": 16, "bins": _bins(M)} if arm == "c" else None
    m = Stage2Model(M, query, arm, S1, sk, fusion_dim=32, head_hidden=16)
    x, ss, typ, rows = _inputs(M, N)
    out = m(x, ss[rows], typ[rows], rows)
    assert torch.equal(out, S1["p1n"][rows])


def test_struct_params_learn_after_zero_init():
    """0 초기화여도 두 스텝 뒤에는 구조 경로에 기울기가 흐른다."""
    M, N = 2, 40
    S1 = _fake_s1(N)
    m = Stage2Model(M, "tok", "c", S1, {"encoder": "ple", "d_s": 16, "bins": _bins(M)},
                    fusion_dim=32, head_hidden=16)
    x, ss, typ, rows = _inputs(M, N)
    y = torch.randn(M, N)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(2):
        opt.zero_grad()
        ((m(x, ss[rows], typ[rows], rows) - y) ** 2).mean().backward()
        g = m.struct.value_w.grad
        opt.step()
    assert g is not None and g.abs().sum() > 0


def test_invalid_tokens_are_ignored():
    """빈 칸(tok_valid=0)의 K/V 를 바꿔도 출력이 같다."""
    M, N = 2, 40
    S1 = _fake_s1(N)
    torch.manual_seed(1)
    m = Stage2Model(M, "tok", "c", S1, {"encoder": "ple", "d_s": 16, "bins": _bins(M)},
                    fusion_dim=32, head_hidden=16)
    with torch.no_grad():
        m.out.weight.normal_()                                     # 0 초기화면 차이가 안 보이므로
    x, ss, typ, rows = _inputs(M, N)
    a = m(x, ss[rows], typ[rows], rows)
    m.K[~m.tv] = 1e3
    m.V[~m.tv] = -1e3
    b = m(x, ss[rows], typ[rows], rows)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.skipif(not os.path.exists(NPZ), reason="E7 Stage 1 출력이 아직 없음")
def test_saved_tokens_reproduce_stage1_summary():
    """저장한 V 와 Stage 1 attention 가중치로 z_seq 를 다시 만들 수 있어야 한다 (칸 배치 검증)."""
    z = np.load(NPZ)
    w = np.nan_to_num(z["attn"]).astype(np.float64)
    assert np.allclose(w.sum(1), 1, atol=1e-4)
    assert (w[~z["tok_valid"]] == 0).all()
    zs = np.einsum("nt,ntd->nd", w, z["V"].astype(np.float64))
    rel = np.abs(zs - z["z_seq"]).max() / np.abs(z["z_seq"]).max()
    assert rel < 5e-3, rel                                         # V 를 fp16 으로 저장한 만큼의 오차
