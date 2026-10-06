"""StructureTokenizer(Qk) 기본 동작 검사. 실제 데이터가 없어도 돈다."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tokenizer as T  # noqa: E402


def _fake(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cont = np.column_stack([rng.uniform(20, 98, n)] + [rng.gamma(2.0, 8.0, n) for _ in range(7)])
    cont[:, 1] = rng.uniform(0, 1, n)                         # rSASA 는 0~1
    ss = rng.integers(0, 3, n)
    return cont.astype(np.float32), ss


def test_bins_plddt_fixed_and_quantiles_from_train_only():
    cont, _ = _fake()
    tr = np.arange(100)
    bins = T.fit_qk_bins(cont[tr], n_bins=4)
    assert np.array_equal(bins[0], T.PLDDT_DOMAIN_BINS)
    for j in range(1, 8):
        assert np.allclose(bins[j], np.unique(np.quantile(cont[tr, j], np.linspace(0, 1, 5))))
    other = cont.copy()
    other[100:] *= 10                                        # train 밖을 바꿔도 경계는 그대로
    assert all(np.array_equal(a, b) for a, b in zip(bins, T.fit_qk_bins(other[tr], 4)))


def test_ple_values():
    tok = T.StructureTokenizer([np.array([0.0, 50, 70, 90, 100])] + [np.array([0.0, 1, 2])] * 7, d_s=8)
    x = torch.zeros(3, 8)
    x[:, 0] = torch.tensor([60.0, -5.0, 120.0])              # 구간 안 / 아래 / 위
    e = tok.ple(x)[:, 0]
    assert torch.allclose(e[0], torch.tensor([1.0, 0.5, 0.0, 0.0]))
    assert torch.allclose(e[1], torch.zeros(4))              # 외삽 없음: 0~1 로 자름
    assert torch.allclose(e[2], torch.ones(4))
    assert (tok.ple(x)[:, 1, 2:] == 0).all()                 # 구간이 2개인 feature 의 패딩 칸은 0


def test_forward_shape_order_and_determinism():
    cont, ss = _fake()
    tok = T.StructureTokenizer.from_train(cont, d_s=16)
    out = tok(torch.as_tensor(cont), torch.as_tensor(ss))
    assert out.shape == (len(cont), 9, 16) and torch.isfinite(out).all()
    assert torch.equal(out, tok(torch.as_tensor(cont), torch.as_tensor(ss)))
    # ss 를 바꾸면 2번째 토큰(인덱스 1)만 바뀐다
    out2 = tok(torch.as_tensor(cont), torch.as_tensor((ss + 1) % 3))
    changed = (out - out2).abs().amax(dim=(0, 2)) > 0
    assert changed.tolist() == [False, True] + [False] * 7


def test_gradients_reach_all_parameters():
    cont, ss = _fake()
    tok = T.StructureTokenizer.from_train(cont, d_s=8)
    tok(torch.as_tensor(cont), torch.as_tensor(ss)).pow(2).sum().backward()
    for n, p in tok.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n


def test_save_load_roundtrip(tmp_path):
    cont, ss = _fake()
    tok = T.StructureTokenizer.from_train(cont, n_bins=3, d_s=8)
    path = tmp_path / "tok.pt"
    tok.save(str(path))
    tok2 = T.StructureTokenizer.load(str(path))
    x, s = torch.as_tensor(cont), torch.as_tensor(ss)
    assert torch.equal(tok(x, s), tok2(x, s))


def test_features_from_frame_rejects_bad_input():
    cont, ss = _fake(5)
    df = pd.DataFrame(cont, columns=T.CONT_FIELDS)
    df[T.SS_FIELD] = [T.SS_CLASSES[i] for i in ss]
    c2, s2 = T.features_from_frame(df)
    assert np.allclose(c2, cont) and (s2 == ss).all()
    with pytest.raises(ValueError):
        T.features_from_frame(df.assign(**{T.SS_FIELD: "coil"}))
    with pytest.raises(KeyError):
        T.features_from_frame(df.drop(columns=["A_rsasa"]))


DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v2_dataset.csv")


@pytest.mark.skipif(not os.path.exists(DATA), reason="python build_dataset.py 를 먼저 실행")
def test_real_data_same_position_same_tokens():
    df = pd.read_csv(DATA)
    cont, ss = T.features_from_frame(df)
    tr = (df.split == "train").to_numpy()
    tok = T.StructureTokenizer.from_train(cont[tr])
    with torch.no_grad():
        out = tok(torch.as_tensor(cont), torch.as_tensor(ss))
    for p in df.anchor_pos.unique()[:20]:
        rows = np.flatnonzero(df.anchor_pos.to_numpy() == p)
        assert torch.equal(out[rows], out[rows[:1]].expand(len(rows), -1, -1))
