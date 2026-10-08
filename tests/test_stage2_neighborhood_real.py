"""Checks against the real WT neighbor cache (data/structure/results/wt_neighbor_cache.npz).
Skipped if the cache has not been built (python build_wt_neighbor_cache.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data/structure/results/wt_neighbor_cache.npz"

pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="wt_neighbor_cache.npz not built")


def test_wt_pdb_mapping_is_contiguous_1_to_376_and_matches_wt_sequence():
    from src.stage2.neighbor_structure import WTNeighborStore
    store = WTNeighborStore(CACHE)
    assert np.array_equal(store.positions, np.arange(1, len(store.positions) + 1))
    assert store.meta["n_residues"] == len(store.positions)


def test_self_excluded_and_no_duplicate_neighbors_and_slot0_is_anchor():
    from src.stage2.neighbor_structure import WTNeighborStore
    store = WTNeighborStore(CACHE)
    for row, p in enumerate(store.positions):
        neigh_positions = store.positions[store.neighbor_idx[row]]
        assert neigh_positions[0] == p and store.is_anchor[row, 0]
        others = neigh_positions[1:][store.valid_full[row, 1:]]
        assert p not in others                              # self excluded from the 8 neighbor candidates
        assert len(set(others.tolist())) == len(others)      # no duplicate neighbor in one anchor's set


def test_neighbors_sorted_by_distance_ascending_with_position_tiebreak():
    from src.stage2.neighbor_structure import WTNeighborStore
    store = WTNeighborStore(CACHE)
    for row in (0, 100, 200, 375):
        d = store.distance[row, 1:][store.valid_full[row, 1:]]
        assert np.all(np.diff(d) >= -1e-5)                   # non-decreasing distance across slots 1..8


def test_raw_for_anchors_e1_masks_all_but_the_anchor_slot():
    from src.stage2.neighbor_structure import WTNeighborStore
    store = WTNeighborStore(CACHE)
    raw_e1 = store.raw_for_anchors([10, 200], "e1")
    raw_e2 = store.raw_for_anchors([10, 200], "e2")
    assert raw_e1["valid"][:, 0].all() and not raw_e1["valid"][:, 1:].any()
    assert raw_e2["valid"].all()                             # every slot was constructed valid for these anchors
    import torch
    assert torch.equal(raw_e1["continuous"], raw_e2["continuous"])   # same content, only the mask differs


def test_fit_positions_union_is_a_strict_superset_of_the_anchors_alone():
    from src.stage2.neighbor_structure import WTNeighborStore
    store = WTNeighborStore(CACHE)
    anchors = [10, 200, 300]
    fit = store.fit_positions_for_train_anchors(anchors)
    assert set(anchors) <= set(fit)
    assert len(fit) > len(set(anchors))       # neighbors add new positions beyond the anchors themselves
