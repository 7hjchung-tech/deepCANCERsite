"""Loader, cohort binding, mask preservation and the Stage 1 input contract.

The tests that need the real 1 GB frozen cache are skipped when it is absent,
so a checkout without it still has a green suite; the contract-level tests run
on a small synthetic cache and always execute.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.embeddings import representation_cache as rc
from src.stage1 import data as sd
from src.stage1.interface import (
    FORBIDDEN_LABEL_FIELDS,
    ContractViolation,
    Stage1Batch,
    check_prediction,
)
from src.stage1.mock import MockTokenConsumer
from src.stage1.targets import TargetScaler

_ROOT = Path(__file__).resolve().parents[1]
CACHE = sd.DEFAULT_CACHE
MANIFEST = sd.DEFAULT_MANIFEST

needs_cache = pytest.mark.skipif(
    not CACHE.exists(), reason=f"{CACHE.name} not built; run `cli export` first"
)


# ==========================================================================
# synthetic cache -- contract-level tests, no 1 GB dependency
# ==========================================================================
def _synthetic(tmp_path, n=6, A=4, L=3, D=5):
    """A tiny cache + manifest that satisfy the same contract as the real ones."""
    wt_len = 12
    var_ids = [f"v{i}" for i in range(n)]
    splits = ["train"] * 4 + ["val", "test"]

    wt_pos, mut_pos, wt_pr, mut_pr, dv, tv, sk = [], [], [], [], [], [], []
    code = {k: i for i, k in enumerate(rc.SLOT_KINDS)}
    for i in range(n):
        # 3 paired slots then one pad, except row 1 which carries a wt_only gap
        if i == 1:
            kinds = ["paired", "wt_only", "paired", "pad"]
        else:
            kinds = ["paired", "paired", "paired", "pad"]
        wt_pos.append([1, 2, 3, 0])
        mut_pos.append([1, 0 if i == 1 else 2, 3, 0])
        wt_pr.append([k != "pad" for k in kinds])
        mut_pr.append([k not in ("pad", "wt_only") for k in kinds])
        dv.append([k == "paired" for k in kinds])
        tv.append([k != "pad" for k in kinds])
        sk.append([code[k] for k in kinds])

    payload = {
        "format_version": 1,
        "layers": [31, 32, 33],
        "A": A,
        "slot_kind_vocab": list(rc.SLOT_KINDS),
        "wt_full": torch.arange(L * wt_len * D, dtype=torch.float32).reshape(L, wt_len, D),
        "mut_windows": torch.arange(n * L * A * D, dtype=torch.float32).reshape(n, L, A, D) + 1.0,
        "row_to_mut_window": torch.arange(n, dtype=torch.long),
        "wt_pos": torch.tensor(wt_pos, dtype=torch.long),
        "mut_pos": torch.tensor(mut_pos, dtype=torch.long),
        "wt_present": torch.tensor(wt_pr, dtype=torch.bool),
        "mut_present": torch.tensor(mut_pr, dtype=torch.bool),
        "delta_valid": torch.tensor(dv, dtype=torch.bool),
        "token_valid": torch.tensor(tv, dtype=torch.bool),
        "slot_kind": torch.tensor(sk, dtype=torch.long),
        "var_id": var_ids,
        "split": splits,
        "variant_type": ["missense"] * 4 + ["inframe_indel", "synonymous"],
        "edit_type": ["missense"] * 4 + ["deletion", "synonymous"],
        "pp": list(range(1, n + 1)),
        "provenance": {"provenance_hash": "synthetic"},
    }
    # zero out the padded region so the contract holds
    payload["mut_windows"][:, :, 3, :] = 0.0

    manifest = pd.DataFrame({
        "var_id": var_ids,
        "split": splits,
        "slim_consequence": ["missense"] * 4 + ["codon_deletion", "synonymous"],
        "pp": list(range(1, n + 1)),
        "z_score_D4_D14": [-1.0, -2.0, 0.5, -8.0, 1.0, -3.0],
        "functional_classification": ["unchanged"] * n,
    })
    cache = rc.RepresentationCache(payload)
    return cache, manifest


@pytest.fixture
def synthetic(tmp_path):
    cache, manifest = _synthetic(tmp_path)
    data = sd.FrozenReprData(
        cache, manifest, verify_cohort=False, verify_provenance=False
    )
    return data


def test_rows_are_addressed_by_var_id_not_by_position(synthetic):
    order = [3, 0, 5]
    batch, targets = synthetic.make_batch(order)
    assert batch.var_id == ["v3", "v0", "v5"]
    assert targets.var_id == batch.var_id
    assert batch.split == ["train", "train", "test"]
    # the target that travels with each id is that id's target, not row order's
    assert targets.y_raw.tolist() == [-8.0, -1.0, -3.0]


def test_a_cache_var_id_missing_from_the_manifest_is_refused(tmp_path):
    cache, manifest = _synthetic(tmp_path)
    manifest = manifest.assign(var_id=["zzz"] + manifest.var_id.tolist()[1:])
    with pytest.raises(sd.CohortError, match="absent from the manifest"):
        sd.FrozenReprData(cache, manifest, verify_cohort=False, verify_provenance=False)


def test_a_split_disagreement_between_cache_and_manifest_is_refused(tmp_path):
    cache, manifest = _synthetic(tmp_path)
    manifest.loc[0, "split"] = "test"
    with pytest.raises(sd.CohortError, match="disagree on split"):
        sd.FrozenReprData(cache, manifest, verify_cohort=False, verify_provenance=False)


def test_duplicate_var_ids_are_refused(tmp_path):
    cache, manifest = _synthetic(tmp_path)
    cache.p["var_id"][1] = "v0"
    with pytest.raises(sd.CohortError, match="duplicate var_ids"):
        sd.FrozenReprData(cache, manifest, verify_cohort=False, verify_provenance=False)


def test_split_membership_is_exactly_the_manifest_assignment(synthetic):
    assert synthetic.indices_for_split("train") == [0, 1, 2, 3]
    assert synthetic.indices_for_split("val") == [4]
    assert synthetic.indices_for_split("test") == [5]
    assert sum(len(synthetic.indices_for_split(s)) for s in sd.SPLITS) == len(synthetic)
    with pytest.raises(KeyError):
        synthetic.indices_for_split("valid")


def test_scaler_is_fitted_on_train_rows_only(synthetic):
    s = synthetic.fit_target_scaler(split="train")
    y_train = synthetic.targets_raw(synthetic.indices_for_split("train"))
    assert s.mean == pytest.approx(y_train.mean())
    assert s.std == pytest.approx(y_train.std(ddof=0))
    assert s.n_train_rows == 4
    assert s.fit_split == "train"
    # it must differ from a scaler fitted on everything -- i.e. val/test are out
    all_y = synthetic.targets_raw(range(len(synthetic)))
    assert s.mean != pytest.approx(all_y.mean())


def test_val_batches_are_standardised_with_the_train_scaler(synthetic):
    s = synthetic.fit_target_scaler(split="train")
    val_idx = synthetic.indices_for_split("val")
    _, targets = synthetic.make_batch(val_idx, scaler=s)
    expected = (targets.y_raw.numpy() - s.mean) / s.std
    assert targets.y_std.numpy() == pytest.approx(expected)
    assert targets.scaler["fit_split"] == "train"
    assert targets.scaler["n_train_rows"] == 4


def test_targets_are_absent_from_the_feature_batch(synthetic):
    batch, targets = synthetic.make_batch([0, 1], scaler=synthetic.fit_target_scaler())
    batch.assert_no_labels()
    for field in FORBIDDEN_LABEL_FIELDS:
        assert not hasattr(batch, field)
    assert not any("z_score" in f or "classification" in f for f in vars(batch))
    # the targets exist, just not inside the features
    assert targets.y_raw.shape == (2,)


def test_gap_slots_stay_token_valid_and_padding_stays_token_invalid(synthetic):
    batch, _ = synthetic.make_batch([1])          # row 1 carries the wt_only gap
    batch.validate()
    kinds = batch.slot_kind_names(0)
    assert kinds == ["paired", "wt_only", "paired", "pad"]

    gap = batch.kind_mask("wt_only")
    pad = batch.kind_mask("pad")
    assert bool(batch.token_valid[gap].all())      # a gap is NOT padding
    assert not bool(batch.delta_valid[gap].any())  # but it is not subtractable
    assert not bool(batch.token_valid[pad].any())  # padding is padding
    assert float(batch.delta_H[:, :, gap.squeeze(0), :].abs().max()) == 0.0


def test_batch_validation_rejects_a_gap_relabelled_as_padding(synthetic):
    batch, _ = synthetic.make_batch([1])
    gap = batch.kind_mask("wt_only")
    batch.token_valid = batch.token_valid & ~gap          # pretend the gap is pad
    with pytest.raises(ContractViolation, match="token_invalid"):
        batch.validate()


def test_batch_validation_rejects_delta_on_a_gap(synthetic):
    batch, _ = synthetic.make_batch([1])
    gap = batch.kind_mask("wt_only")
    batch.delta_valid = batch.delta_valid | gap
    with pytest.raises(ContractViolation):
        batch.validate()


def test_batch_validation_rejects_a_wrong_shape(synthetic):
    batch, _ = synthetic.make_batch([0, 1])
    batch.H_MUT = batch.H_MUT[:, :, :-1, :]
    with pytest.raises(ContractViolation, match="shape"):
        batch.validate()


def test_mock_consumer_returns_one_finite_scalar_per_variant(synthetic):
    batch, _ = synthetic.make_batch([0, 1, 2])
    model = MockTokenConsumer(n_layers=batch.n_layers, embed_dim=batch.embed_dim)
    pred = model(batch)
    assert tuple(pred.shape) == (3,)
    assert torch.isfinite(pred).all()
    check_prediction(pred, batch)
    with pytest.raises(ContractViolation):
        check_prediction(pred[:2], batch)


def test_padding_cannot_influence_the_prediction(synthetic):
    """The mask contract, proved rather than asserted."""
    batch, _ = synthetic.make_batch([0, 1, 2])
    model = MockTokenConsumer(n_layers=batch.n_layers, embed_dim=batch.embed_dim)
    with torch.no_grad():
        base = model(batch)
        pad = batch.kind_mask("pad").unsqueeze(1).unsqueeze(-1)
        batch.delta_H = batch.delta_H + pad.to(batch.delta_H.dtype) * 1e6
        after = model(batch)
    assert torch.equal(base, after)


def test_a_gap_slot_does_influence_the_prediction(synthetic):
    """The other half: gaps are real tokens, so they must not be masked away."""
    batch, _ = synthetic.make_batch([1])
    model = MockTokenConsumer(n_layers=batch.n_layers, embed_dim=batch.embed_dim)
    with torch.no_grad():
        base = model(batch)
        gap = batch.kind_mask("wt_only").unsqueeze(1).unsqueeze(-1)
        batch.H_WT = batch.H_WT + gap.to(batch.H_WT.dtype) * 1e3
        batch.delta_H = batch.delta_H + gap.to(batch.delta_H.dtype) * 1e3
        after = model(batch)
    assert not torch.equal(base, after)


def test_dataloader_yields_feature_target_pairs_of_the_right_split(synthetic):
    s = synthetic.fit_target_scaler()
    loader = sd.make_loader(synthetic, "train", batch_size=2, shuffle=False, scaler=s)
    seen = []
    for batch, targets in loader:
        batch.validate()
        assert set(batch.split) == {"train"}
        assert targets.var_id == batch.var_id
        seen += batch.var_id
    assert seen == ["v0", "v1", "v2", "v3"]


# ==========================================================================
# provenance
# ==========================================================================
def _real_provenance():
    p = CACHE.with_suffix(".provenance.json")
    return json.loads(p.read_text())


@needs_cache
def test_the_shipped_cache_passes_provenance_verification():
    prov = sd.verify_cache_provenance(_real_provenance(), manifest_path=MANIFEST)
    assert prov["contains_targets"] is False
    assert prov["adapter"] == "none"
    assert prov["repr_layers"] == [31, 32, 33]


@needs_cache
@pytest.mark.parametrize(
    "field,bad",
    [
        ("manifest_hash", "deadbeef"),
        ("split_schema_hash", "deadbeef"),
        ("variant_map_hash", "deadbeef"),
        ("window_rule_version", "window_v1"),
        ("alignment_version", "taskB_canonical_v1"),
        ("split_rule_version", "split_rule_v1"),
        ("repr_layers", [30, 31, 32]),
        ("cache_precision", "torch.float16"),
        ("adapter", "lora"),
        ("adapter_state", "trainable"),
        ("model_name", "esm2_t30_150M_UR50D"),
        ("contains_targets", True),
    ],
)
def test_a_provenance_mismatch_is_rejected_and_named(field, bad):
    prov = dict(_real_provenance())
    prov[field] = bad
    with pytest.raises(sd.ProvenanceError) as e:
        sd.verify_cache_provenance(prov, manifest_path=MANIFEST)
    assert field in str(e.value)


@needs_cache
def test_a_tampered_provenance_hash_is_rejected():
    prov = dict(_real_provenance(), provenance_hash="0" * 64)
    with pytest.raises(sd.ProvenanceError, match="provenance_hash"):
        sd.verify_cache_provenance(prov, manifest_path=MANIFEST)


@needs_cache
def test_a_row_count_mismatch_is_rejected():
    with pytest.raises(sd.ProvenanceError, match="n_rows"):
        sd.verify_cache_provenance(_real_provenance(), manifest_path=MANIFEST, n_rows=1)


# ==========================================================================
# the real cache
# ==========================================================================
@pytest.fixture(scope="module")
def real_data():
    if not CACHE.exists():
        pytest.skip("frozen cache not built")
    return sd.load_frozen_data(verify_cohort=True)


@needs_cache
def test_real_cohort_is_exactly_the_validated_in_eval_scope_rows(real_data):
    rep = real_data.report()
    assert rep.cohort_verified is True
    assert rep.n_cache_rows == 5885
    assert rep.n_manifest_rows == 5887
    assert rep.rows_by_split == {"train": 4123, "val": 881, "test": 881}
    # the two excluded rows really are absent
    assert "chr17_58734138__TGT" not in real_data.cache_index_of
    assert "chr17_58732531_TGTTTCAAATCA_" not in real_data.cache_index_of


@needs_cache
def test_real_var_ids_align_with_the_manifest_row_by_row(real_data):
    man = real_data.manifest.set_index("var_id")
    for vid in list(real_data.var_id)[::311]:
        i = real_data.cache_index_of[vid]
        assert real_data.cache.var_id[i] == vid
        assert real_data.split_of[vid] == man.loc[vid, "split"]
        assert real_data.targets_raw([i])[0] == pytest.approx(
            man.loc[vid, "z_score_D4_D14"]
        )


@needs_cache
def test_real_scaler_is_train_only_and_matches_the_manifest(real_data):
    s = real_data.fit_target_scaler(split="train")
    man = real_data.manifest
    train_ids = {v for v in real_data.var_id if real_data.split_of[v] == "train"}
    y = man[man.var_id.isin(train_ids)].z_score_D4_D14.to_numpy()
    assert s.n_train_rows == len(y) == 4123
    assert s.mean == pytest.approx(y.mean())
    assert s.std == pytest.approx(y.std(ddof=0))
    for other in ("val", "test"):
        ids = {v for v in real_data.var_id if real_data.split_of[v] == other}
        y_o = man[man.var_id.isin(ids)].z_score_D4_D14.to_numpy()
        assert s.mean != pytest.approx(y_o.mean())


@needs_cache
def test_real_batch_shape_and_masks(real_data):
    s = real_data.fit_target_scaler()
    idx = real_data.indices_for_split("val")[:16]
    batch, targets = real_data.make_batch(idx, scaler=s)
    batch.validate()
    assert tuple(batch.H_WT.shape) == (16, 3, real_data.cache.A, 1280)
    assert batch.layers == [31, 32, 33]
    assert set(batch.split) == {"val"}
    assert targets.y_std is not None

    model = MockTokenConsumer(n_layers=3, embed_dim=1280)
    pred = model(batch)
    check_prediction(pred, batch)
    assert torch.isfinite(pred).all()


@needs_cache
def test_the_real_cache_payload_contains_no_label_column(real_data):
    keys = set(real_data.cache.p)
    assert "z_score_D4_D14" not in keys
    assert "functional_classification" not in keys
    assert real_data.cache.provenance["contains_targets"] is False
