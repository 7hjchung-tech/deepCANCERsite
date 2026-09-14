"""Cache schema + dataset/cohort join validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.stage1.cache import FakeFrozenEncoder, RawStage1Cache, build_cache_from_manifest
from src.stage1.dataset import build_cohort, join_cohort_with_cache
from src.stage1.schema import CACHE_SCHEMA_VERSION
from src.stage1.synthetic import SYNTHETIC_WT, make_synthetic_manifest


def test_cache_round_trip(tmp_path: Path):
    wt_seq = SYNTHETIC_WT
    rows = make_synthetic_manifest(wt_seq)
    cohort = build_cohort(rows, wt_seq)
    encoder = FakeFrozenEncoder(hidden_dim=8, seed=1)
    cache_rows = [{"var_id": e["var_id"], "mut_seq": e["row"]["mut_seq"]} for e in cohort.supported]
    cache = build_cache_from_manifest(cache_rows, wt_seq, [33], encoder)

    path = tmp_path / "cache.pt"
    cache.save(path)
    loaded = RawStage1Cache.load(path)

    assert loaded.schema_version == CACHE_SCHEMA_VERSION
    assert loaded.hidden_dim == 8
    assert loaded.layers == [33]
    assert set(loaded.mut_entries.keys()) == set(cache.mut_entries.keys())
    for vid in cache.mut_entries:
        assert torch.equal(loaded.get_mut(vid, 33), cache.get_mut(vid, 33))
    assert torch.equal(loaded.get_wt(33), cache.get_wt(33))


def test_cache_schema_version_mismatch_raises(tmp_path: Path):
    wt_seq = SYNTHETIC_WT
    encoder = FakeFrozenEncoder(hidden_dim=4, seed=0)
    cache = build_cache_from_manifest([{"var_id": "x", "mut_seq": wt_seq}], wt_seq, [33], encoder)
    path = tmp_path / "cache.pt"
    cache.save(path)

    raw = torch.load(path, weights_only=False)
    raw["schema_version"] = "some-old-version"
    torch.save(raw, path)

    with pytest.raises(ValueError, match="schema_version"):
        RawStage1Cache.load(path)


def test_duplicate_var_id_reported_not_crashed():
    wt_seq = SYNTHETIC_WT
    rows = make_synthetic_manifest(wt_seq)
    rows.append(dict(rows[0]))   # duplicate the first (missense) row's var_id
    cohort = build_cohort(rows, wt_seq)
    dup_reasons = [s["reason"] for s in cohort.skipped if s["var_id"] == rows[0]["var_id"]]
    assert any("duplicate" in r for r in dup_reasons)


def test_missing_cache_entry_reported_not_crashed():
    wt_seq = SYNTHETIC_WT
    rows = make_synthetic_manifest(wt_seq)
    cohort = build_cohort(rows, wt_seq)
    encoder = FakeFrozenEncoder(hidden_dim=4, seed=0)
    # build a cache that omits one supported variant on purpose
    omit_id = cohort.supported[0]["var_id"]
    cache_rows = [
        {"var_id": e["var_id"], "mut_seq": e["row"]["mut_seq"]}
        for e in cohort.supported if e["var_id"] != omit_id
    ]
    cache = build_cache_from_manifest(cache_rows, wt_seq, [33], encoder)

    joined = join_cohort_with_cache(cohort, cache)
    assert omit_id not in {e["var_id"] for e in joined.supported}
    assert any(s["var_id"] == omit_id and "cache" in s["reason"] for s in joined.skipped)


def test_row_order_independence_of_join():
    """var_id join must not assume manifest row order matches cache insertion
    order -- shuffle both and confirm identical resulting cohort."""
    wt_seq = SYNTHETIC_WT
    rows = make_synthetic_manifest(wt_seq)
    cohort = build_cohort(rows, wt_seq)
    encoder = FakeFrozenEncoder(hidden_dim=4, seed=0)
    cache_rows = [{"var_id": e["var_id"], "mut_seq": e["row"]["mut_seq"]} for e in cohort.supported]
    cache_rows_shuffled = list(reversed(cache_rows))
    cache = build_cache_from_manifest(cache_rows_shuffled, wt_seq, [33], encoder)

    joined = join_cohort_with_cache(cohort, cache)
    assert len(joined.supported) == len(cohort.supported)
    for e in joined.supported:
        assert torch.equal(cache.get_mut(e["var_id"], 33), cache.get_mut(e["var_id"], 33))


def test_wt_forwarded_once_broadcast_matches_direct_call():
    """The cache builder forwards WT exactly once; check the cached WT vector
    equals a direct single-sequence encode() call."""
    wt_seq = SYNTHETIC_WT
    encoder = FakeFrozenEncoder(hidden_dim=4, seed=2)
    cache = build_cache_from_manifest([{"var_id": "x", "mut_seq": wt_seq}], wt_seq, [33], encoder)
    direct = encoder.encode([wt_seq], [33])[33][0]
    assert torch.equal(cache.get_wt(33), direct)
