"""Integration tests that run the REAL repo-native ESM-2 650M checkpoint.

These are the regression form of the Task C correctness fixture: the same
contract, asserted by pytest instead of by the CLI report. They are skipped
automatically when the checkpoint is not in the torch.hub cache, so a
checkout without the 2.6 GB download still has a green test suite.

Run just these with:
    .venv/bin/python -m pytest tests/test_esm_integration.py -q
"""

import pytest
import torch

from src.embeddings import representation_cache as rc
from src.embeddings.cli import DUP_LABELS, select_fixture_rows
from src.embeddings.esm_encoder import ESMEncoder
from src.embeddings.likelihood import AA20, llr_table, masked_wt_profile

pytestmark = pytest.mark.skipif(
    not rc.checkpoint_path().exists(),
    reason=f"{rc.MODEL_NAME} checkpoint not in the torch.hub cache",
)

WT = open("data/wt_sequence.txt").read().strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#: layers 31/32 are pre-final-LayerNorm and reach |h| ~ 500, so agreement is
#: asserted relative to the tensor scale, never as a raw absolute difference.
REL_TOL = 1e-5


@pytest.fixture(scope="module")
def encoder():
    enc = ESMEncoder({"device": DEVICE, "repr_layer": list(rc.REPR_LAYERS)})
    enc.model.eval()
    yield enc
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def manifest():
    import pandas as pd

    return pd.read_csv("data/split_manifest.csv")


@pytest.fixture(scope="module")
def fixture_plans(manifest):
    fx = select_fixture_rows(manifest)
    plans, skipped = rc.plan_rows(fx, WT, W=10, in_scope_only=False)
    assert not skipped, skipped
    label_of = dict(zip(fx.var_id, fx.fixture_label))
    return {label_of[p.var_id]: p for p in plans}


def test_the_fixture_covers_both_real_duplication_rows(fixture_plans):
    assert set(DUP_LABELS) <= set(fixture_plans)
    assert {int(fixture_plans[l].pp) for l in DUP_LABELS} == {186, 351}


# ==========================================================================
# forward contract
# ==========================================================================
def test_model_is_frozen_and_in_eval_mode(encoder):
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.model.parameters())
    assert next(encoder.model.parameters()).dtype is torch.float32


def test_repr_layers_31_32_33_have_the_expected_shape(encoder):
    reps = encoder.encode([WT], repr_layers=[31, 32, 33])
    assert sorted(reps) == [31, 32, 33]
    for layer in (31, 32, 33):
        assert tuple(reps[layer].shape) == (1, len(WT), rc.EMBED_DIM)
    # distinct layers must not collapse onto each other
    assert not torch.allclose(reps[31], reps[32])
    assert not torch.allclose(reps[32], reps[33])


def test_bos_and_eos_are_stripped(encoder):
    """Residue i of the output must be residue i of the input, not BOS."""
    reps = encoder.encode([WT], repr_layers=[33])[33]
    _, _, tokens = encoder.batch_converter([("x", WT)])
    assert tokens.shape[1] == len(WT) + 2          # BOS + residues + EOS
    assert reps.shape[1] == len(WT)


def test_mixed_length_batch_is_refused_not_truncated(encoder, fixture_plans):
    short = fixture_plans["deletion"].mut_seq
    assert len(short) != len(WT)
    with pytest.raises(ValueError, match="equal-length"):
        encoder.encode([WT, short], repr_layers=[33])


def test_encode_by_length_handles_mixed_lengths_without_truncating(encoder, fixture_plans):
    seqs = [WT, fixture_plans["deletion"].mut_seq, fixture_plans[DUP_LABELS[0]].mut_seq]
    assert len({len(s) for s in seqs}) == 3
    out = encoder.encode_by_length(seqs, repr_layers=[31, 32, 33], batch_size=2)
    assert len(out) == len(seqs)
    for seq, reps in zip(seqs, out):
        for layer in (31, 32, 33):
            assert tuple(reps[layer].shape) == (len(seq), rc.EMBED_DIM)


def test_batched_and_individual_forwards_agree_within_relative_tolerance(encoder):
    a = WT
    b = WT[:99] + ("A" if WT[99] != "A" else "G") + WT[100:]
    batched = encoder.encode([a, b], repr_layers=[31, 32, 33])
    for i, seq in enumerate([a, b]):
        single = encoder.encode([seq], repr_layers=[31, 32, 33])
        for layer in (31, 32, 33):
            diff = (single[layer][0] - batched[layer][i]).abs().max().item()
            scale = batched[layer][i].abs().max().item()
            assert diff / scale < REL_TOL, f"layer {layer}: {diff:.3e} / {scale:.1f}"


def test_a_fixed_batch_shape_is_bit_deterministic(encoder):
    reps = encoder.encode([WT, WT], repr_layers=[31, 32, 33])
    for layer in (31, 32, 33):
        assert torch.equal(reps[layer][0], reps[layer][1])


# ==========================================================================
# forward-cache reuse
# ==========================================================================
def test_identical_sequences_are_forwarded_once(encoder, fixture_plans):
    fc = rc.SequenceForwardCache(encoder, rc.REPR_LAYERS, batch_size=2)
    syn = fixture_plans["synonymous"]
    assert syn.mut_seq == WT

    wt_full = fc.get(WT)
    assert fc.stats.unique_sequences_encoded == 1
    # synonymous MUT protein is the WT protein: same hash, same tensor object
    assert fc.get(syn.mut_seq) is wt_full
    assert fc.stats.unique_sequences_encoded == 1
    assert fc.stats.cache_hits >= 1

    fc.ensure([WT, syn.mut_seq, fixture_plans["missense"].mut_seq])
    assert fc.stats.unique_sequences_encoded == 2   # only the missense protein was new
    assert fc.n_unique == 2


def test_dna_rows_sharing_a_protein_do_not_re_forward(encoder, manifest):
    """Distinct DNA rows can encode the same protein edit; ESM sees it once."""
    mis = manifest[manifest.slim_consequence == "missense"].head(40)
    plans, _ = rc.plan_rows(mis, WT, in_scope_only=True)
    fc = rc.SequenceForwardCache(encoder, rc.REPR_LAYERS, batch_size=8)
    fc.ensure([p.mut_seq for p in plans])
    n_unique = len({p.mut_seq_hash for p in plans})
    assert fc.stats.unique_sequences_encoded == n_unique
    assert fc.stats.unique_sequences_encoded < len(plans)


def test_sequence_count_and_forward_call_count_are_reported_separately(encoder, manifest):
    """A sequence count is not a model() count: batching makes them differ."""
    mis = manifest[manifest.slim_consequence == "missense"].head(24)
    plans, _ = rc.plan_rows(mis, WT, in_scope_only=True)
    seqs = list({p.mut_seq_hash: p.mut_seq for p in plans}.values())
    assert len(seqs) > 8, "need more than one batch for this to be meaningful"

    before = encoder.forward_calls
    fc = rc.SequenceForwardCache(encoder, rc.REPR_LAYERS, batch_size=8)
    fc.ensure(seqs)
    calls = encoder.forward_calls - before

    assert fc.stats.unique_sequences_encoded == len(seqs)
    assert fc.stats.model_forward_batch_calls == calls
    assert calls < len(seqs)                      # batching really happened
    # all these MUT proteins are the same length -> ceil(n / batch_size) batches
    assert len({len(s) for s in seqs}) == 1
    assert calls == -(-len(seqs) // 8)


def test_real_fixture_windows_have_the_exact_contracted_size(fixture_plans):
    """No model needed for the arithmetic, but these are the REAL manifest rows."""
    expect = {
        "missense": 21, "synonymous": 21, "deletion": 21,
        DUP_LABELS[0]: 21, DUP_LABELS[1]: 21, "delins": 26,
    }
    for label, n in expect.items():
        assert fixture_plans[label].n_slots == n, label
    # termini are the only place a window may be shorter
    for label in ("missense_nterm", "missense_cterm"):
        assert fixture_plans[label].n_slots < 21


# ==========================================================================
# end-to-end export / reload
# ==========================================================================
@pytest.fixture(scope="module")
def exported(encoder, fixture_plans, tmp_path_factory):
    plans = list(fixture_plans.values())
    out = tmp_path_factory.mktemp("repr") / "fixture.pt"
    res = rc.export_representations(
        plans, WT, encoder, out,
        manifest_path=rc._ROOT / "data" / "split_manifest.csv",
        W=10, batch_size=2, device=DEVICE, A=max(p.n_slots for p in plans),
    )
    return res, fixture_plans


def test_export_shapes_match_the_logical_contract(exported):
    res, plans = exported
    cache = rc.load_cache(res.path, expected_provenance=res.provenance)
    b = cache.batch(range(len(cache)))
    expect = (len(plans), 3, cache.A, rc.EMBED_DIM)
    assert tuple(b["H_WT"].shape) == expect
    assert tuple(b["H_MUT"].shape) == expect
    assert tuple(b["delta_H"].shape) == expect
    assert cache.layers == [31, 32, 33]


def test_reload_is_bit_identical(exported):
    res, _ = exported
    a = rc.load_cache(res.path, expected_provenance=res.provenance)
    b = rc.load_cache(res.path, expected_provenance=res.provenance)
    ba, bb = a.batch(range(len(a))), b.batch(range(len(b)))
    for key in ("H_WT", "H_MUT", "delta_H", "wt_pos", "mut_pos",
                "wt_present", "mut_present", "delta_valid", "token_valid"):
        assert torch.equal(ba[key], bb[key]), key
    assert a.provenance["provenance_hash"] == b.provenance["provenance_hash"]


def test_synonymous_valid_delta_is_exactly_zero(exported):
    res, plans = exported
    cache = rc.load_cache(res.path, expected_provenance=res.provenance)
    i = cache.var_id.index(plans["synonymous"].var_id)
    row = cache.row(i)
    dv = row["delta_valid"]
    assert dv.sum() > 0
    assert row["delta_H"][:, dv, :].abs().max().item() == 0.0
    assert torch.equal(row["H_WT"][:, dv, :], row["H_MUT"][:, dv, :])


def test_missense_delta_is_nonzero_at_the_substituted_position(exported):
    res, plans = exported
    cache = rc.load_cache(res.path, expected_provenance=res.provenance)
    plan = plans["missense"]
    row = cache.row(cache.var_id.index(plan.var_id))
    at = row["wt_pos"].tolist().index(int(plan.pp))
    assert row["delta_H"][:, at, :].abs().max().item() > 0.0


def test_delins_gathers_the_aligned_residues_not_the_same_index(exported, encoder):
    res, plans = exported
    cache = rc.load_cache(res.path, expected_provenance=res.provenance)
    plan = plans["delins"]
    i = cache.var_id.index(plan.var_id)
    row, kinds = cache.row(i), cache.slot_kind_names(i)
    wt_pos, mut_pos = row["wt_pos"].tolist(), row["mut_pos"].tolist()

    at = wt_pos.index(343)
    assert kinds[at] == "paired" and mut_pos[at] == 339

    fc = rc.SequenceForwardCache(encoder, rc.REPR_LAYERS, batch_size=2)
    expected = fc.get(plan.mut_seq)[:, 338, :] - fc.get(WT)[:, 342, :]
    assert torch.equal(row["delta_H"][:, at, :], expected)


def test_gaps_keep_their_real_representation_and_contribute_zero_delta(exported):
    res, plans = exported
    cache = rc.load_cache(res.path, expected_provenance=res.provenance)
    i = cache.var_id.index(plans["delins"].var_id)
    row, kinds = cache.row(i), cache.slot_kind_names(i)

    for a, kind in enumerate(kinds):
        if kind == "wt_only":
            assert row["H_WT"][:, a, :].abs().max().item() > 0.0     # kept
            assert row["H_MUT"][:, a, :].abs().max().item() == 0.0
            assert row["delta_H"][:, a, :].abs().max().item() == 0.0
            assert bool(row["token_valid"][a]) is True               # NOT padding
            assert bool(row["delta_valid"][a]) is False
        elif kind == "mut_only":
            assert row["H_MUT"][:, a, :].abs().max().item() > 0.0    # kept
            assert row["H_WT"][:, a, :].abs().max().item() == 0.0
            assert row["delta_H"][:, a, :].abs().max().item() == 0.0
            assert bool(row["token_valid"][a]) is True
        elif kind == "pad":
            assert bool(row["token_valid"][a]) is False
            assert row["H_WT"][:, a, :].abs().max().item() == 0.0
            assert row["H_MUT"][:, a, :].abs().max().item() == 0.0


def test_export_stores_no_target_or_functional_classification(exported):
    res, _ = exported
    cache = rc.load_cache(res.path, expected_provenance=res.provenance)
    assert "z_score_D4_D14" not in cache.p
    assert "functional_classification" not in cache.p
    assert res.provenance["contains_targets"] is False


def test_stale_provenance_is_rejected_on_a_real_cache(exported):
    res, _ = exported
    for field in ("base_checkpoint_hash", "repr_layers", "forward_precision",
                  "cache_precision", "window_rule_version", "window_W",
                  "alignment_version", "variant_map_hash", "manifest_hash",
                  "split_schema_hash", "adapter"):
        stale = dict(res.provenance, **{field: "STALE-VALUE"})
        with pytest.raises(rc.StaleCacheError) as e:
            rc.load_cache(res.path, expected_provenance=stale)
        assert field in str(e.value)


def test_provenance_records_the_real_precisions(exported):
    res, _ = exported
    prov = res.provenance
    assert prov["forward_precision"] == "torch.float32"
    assert prov["cache_precision"] == "torch.float32"
    assert prov["delta_precision"] == "torch.float32"
    assert prov["adapter"] == "none"
    assert prov["repr_layers"] == [31, 32, 33]
    assert prov["base_checkpoint_hash"] not in ("", "missing")


# ==========================================================================
# masked-WT LLR
# ==========================================================================
def test_masked_profile_is_a_normalised_20aa_distribution(encoder):
    prof = masked_wt_profile(encoder, WT, [50, 100, 200], batch_size=3)
    assert prof.logprobs.shape == (3, 20)
    assert prof.aa_order == AA20
    assert prof.masked_positions == 3
    assert prof.masked_inputs == 3
    assert prof.model_forward_batch_calls == 1     # one batch of 3, not 3 forwards
    # log-probs over the full vocab: the 20-AA subset must sum to <= 1
    assert 0.0 < float(torch.tensor(prof.logprobs).exp().sum(-1).min()) <= 1.0 + 1e-5
    assert float(torch.tensor(prof.logprobs).exp().sum(-1).max()) <= 1.0 + 1e-5
    assert prof.wt_aa == [WT[49], WT[99], WT[199]]


def test_masked_inputs_and_forward_calls_are_counted_separately(encoder):
    """20 masked sequences at batch_size 8 is 3 model() calls, not 20."""
    prof = masked_wt_profile(encoder, WT, list(range(50, 70)), batch_size=8)
    assert prof.masked_positions == 20
    assert prof.masked_inputs == 20
    assert prof.model_forward_batch_calls == 3


def test_masking_actually_changes_the_prediction(encoder):
    """A masked position must not simply read back its own WT residue."""
    prof = masked_wt_profile(encoder, WT, [100], batch_size=1)
    row = prof.logprobs[0]
    assert row.argmax() < 20
    assert float(row.max()) < 0.0                      # a log-prob, never > 0
    assert not (row == row[0]).all()


def test_llr_is_defined_only_for_missense(encoder, manifest):
    sub = manifest[manifest.slim_consequence.isin(
        ["missense", "synonymous", "codon_deletion"])].groupby(
        "slim_consequence").head(3)
    positions = sorted({int(p) for p in sub[sub.slim_consequence == "missense"].pp})
    prof = masked_wt_profile(encoder, WT, positions, batch_size=4)
    table = llr_table(sub, WT, prof)

    mis = table[table.slim_consequence == "missense"]
    assert mis.llr_valid.all()
    assert mis.llr.notna().all()
    assert (mis.scoring_method == "masked_marginal_wt").all()
    assert mis.ref_matches_wt.all()

    other = table[table.slim_consequence != "missense"]
    assert not other.llr_valid.any()
    assert other.llr.isna().all()                      # never a fabricated 0.0
    assert other.llr_undefined_reason.notna().all()


def test_llr_equals_the_logprob_difference_it_claims(encoder, manifest):
    sub = manifest[manifest.slim_consequence == "missense"].head(5)
    positions = sorted({int(p) for p in sub.pp})
    prof = masked_wt_profile(encoder, WT, positions, batch_size=4)
    table = llr_table(sub, WT, prof)
    for _, r in table[table.llr_valid].iterrows():
        i = list(prof.positions).index(r.pp)
        assert r.logp_wt == pytest.approx(prof.logprobs[i][AA20.index(r.wt_aa)])
        assert r.logp_mut == pytest.approx(prof.logprobs[i][AA20.index(r.mut_aa)])
        assert r.llr == pytest.approx(r.logp_mut - r.logp_wt)


def test_llr_output_carries_no_labels(encoder, manifest):
    sub = manifest[manifest.slim_consequence == "missense"].head(3)
    prof = masked_wt_profile(encoder, WT, sorted({int(p) for p in sub.pp}), batch_size=3)
    table = llr_table(sub, WT, prof)
    assert "z_score_D4_D14" not in table.columns
    assert "functional_classification" not in table.columns
