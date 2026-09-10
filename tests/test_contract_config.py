"""The YAML contract must not be able to drift from the runtime constants.

configs/esm_repr_v1.yaml restates the scientific contract (model, layers,
precisions, window rule, split rule, alignment version, LLR method) AND supplies
the CLI's runtime defaults. These tests check both directions: the shipped file
agrees with the code today, and any disagreement is a hard startup failure that
names the offending field rather than a silent divergence.
"""

import yaml
import pytest

from src.embeddings import contract as ct
from src.embeddings import representation_cache as rc
from src.embeddings.likelihood import AA20, LLR_VERSION, SCORING_METHOD
from src.embeddings.variant_map import SLOT_KINDS, SPLIT_RULE, SPLIT_RULE_VERSION


def test_shipped_config_matches_the_python_constants():
    c = ct.load_contract()
    assert c.path == ct.DEFAULT_CONFIG
    assert check_free(c.cfg)


def check_free(cfg) -> bool:
    bad = ct.check_contract(cfg)
    assert not bad, f"config drifted from the code: {bad}"
    return True


def test_every_contract_constant_is_actually_covered():
    """A constant nobody validates is a constant that can drift."""
    covered = {dotted for dotted, _ in ct._expectations()}
    for dotted in (
        "esm.model_name", "esm.repr_layers", "esm.backend", "esm.layer_convention",
        "precision.forward", "precision.cache", "precision.delta", "precision.logits",
        "window.rule", "window.version", "split.rule", "split.version",
        "alignment.version", "alignment.slot_kinds", "alignment.arrays",
        "llr.scoring_method", "llr.version", "llr.aa_order", "embedding.dim",
        "batching.equal_length_batches_only",
    ):
        assert dotted in covered, f"{dotted} is documented but never validated"


@pytest.mark.parametrize(
    "dotted,bad_value",
    [
        ("esm.model_name", "esm2_t30_150M_UR50D"),
        ("esm.repr_layers", [30, 31, 32]),
        ("precision.cache", "fp16"),
        ("window.version", "window_v1"),
        ("window.rule", "indel: core plus 10 flank residues outside it"),
        ("split.version", "split_rule_v1"),
        ("split.rule", "cross_span from wt_pos + mut_pos"),
        ("alignment.version", "taskB_canonical_v1"),
        ("llr.scoring_method", "wt_marginal"),
        ("llr.aa_order", "ACDEFGHIKLMNPQRSTVWX"),
        ("embedding.dim", 640),
    ],
)
def test_a_drifted_field_is_refused_and_named(tmp_path, dotted, bad_value):
    cfg = yaml.safe_load(ct.DEFAULT_CONFIG.read_text())
    node = cfg
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node[part]
    node[leaf] = bad_value

    p = tmp_path / "drifted.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ct.ContractMismatchError) as e:
        ct.load_contract(p)
    assert dotted in str(e.value)


def test_a_missing_contract_section_is_refused(tmp_path):
    cfg = yaml.safe_load(ct.DEFAULT_CONFIG.read_text())
    del cfg["split"]
    p = tmp_path / "nosplit.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ct.ContractMismatchError) as e:
        ct.load_contract(p)
    assert "split.rule" in str(e.value)


def test_prose_compares_by_content_not_by_line_wrapping(tmp_path):
    """YAML folded scalars re-wrap; that must not read as a contract change."""
    cfg = yaml.safe_load(ct.DEFAULT_CONFIG.read_text())
    cfg["window"]["rule"] = "   " + rc.WINDOW_RULE.replace(" ", "\n  ") + "  "
    cfg["split"]["rule"] = SPLIT_RULE.replace(" ", "\n")
    p = tmp_path / "rewrapped.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert ct.load_contract(p) is not None


def test_config_supplies_the_runtime_knobs_the_cli_uses():
    c = ct.load_contract()
    assert c.window_W == 10
    assert c.batch_size >= 1
    assert c.device in ("cuda", "cpu")
    assert c.path_of("manifest").name == "split_manifest.csv"
    assert c.path_of("wt_sequence").name == "wt_sequence.txt"
    assert c.path_of("repr_out_dir").name == "esm_repr_v1"
    assert c.path_of("llr_out_dir").name == "esm650m_llr"


def test_documented_versions_are_the_ones_that_gate_cache_reuse():
    cfg = yaml.safe_load(ct.DEFAULT_CONFIG.read_text())
    assert cfg["window"]["version"] == rc.WINDOW_RULE_VERSION
    assert cfg["split"]["version"] == SPLIT_RULE_VERSION
    assert cfg["alignment"]["version"] == rc.ALIGNMENT_VERSION
    for field in ("window_rule", "window_rule_version", "split_rule",
                  "split_rule_version", "alignment_version"):
        assert field in rc.PROVENANCE_IDENTITY_FIELDS
    assert cfg["alignment"]["slot_kinds"] == list(SLOT_KINDS)
    assert cfg["llr"]["version"] == LLR_VERSION
    assert cfg["llr"]["scoring_method"] == SCORING_METHOD
    assert cfg["llr"]["aa_order"] == "".join(AA20)


def test_expected_slot_budgets_in_the_config_match_the_window_builder():
    """The slot counts the config advertises are the ones the code produces."""
    from src.embeddings.variant_map import build_variant_record

    cfg = yaml.safe_load(ct.DEFAULT_CONFIG.read_text())
    want = cfg["window"]["expected_slots"]
    wt = open("data/wt_sequence.txt").read().strip()
    rows = {
        "missense": ({"var_id": "m", "slim_consequence": "missense", "pp": 100,
                      "ref_aa": wt[99], "alt_aa": "A", "HGVSp": "x:p.Xaa100Ala"}, len(wt)),
        "synonymous": ({"var_id": "s", "slim_consequence": "synonymous", "pp": 100,
                        "ref_aa": wt[99], "alt_aa": wt[99], "HGVSp": "x:p.Xaa100="}, len(wt)),
        "one_residue_deletion": ({"var_id": "d", "slim_consequence": "codon_deletion",
                                  "pp": 100, "HGVSp": "x:p.Ile100del"}, len(wt) - 1),
        "one_residue_duplication": ({"var_id": "u",
                                     "slim_consequence": "clinical_inframe_insertion",
                                     "pp": 186, "HGVSp": "x:p.Lys186dup"}, len(wt) + 1),
        "leu338_lys342delinsgln": ({"var_id": "x",
                                    "slim_consequence": "clinical_inframe_deletion",
                                    "pp": 338, "HGVSp": "x:p.Leu338_Lys342delinsGln"},
                                   len(wt) - 4),
    }
    for key, (row, mut_len) in rows.items():
        rec = build_variant_record(row, wt)
        w = rc.build_window_alignment(rec, len(wt), mut_len, W=cfg["window"]["W"])
        assert len(w["slot_kind"]) == want[key], key
