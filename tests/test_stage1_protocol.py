"""The fair-comparison contract, and the fact that no comparison was run."""

import pytest

from src.stage1.losses import HUBER_DELTA
from src.stage1.protocol import (
    FAIR_COMPARISON_FIELDS,
    PILOT,
    PilotProtocol,
    assert_fair,
)
from src.stage1.targets import DDOF


def test_the_pilot_protocol_records_its_own_unrun_status():
    assert "NOT RUN" in PILOT.status
    assert "external" in PILOT.status
    assert PILOT.test_labels_used is False
    assert PILOT.composite_score is False


def test_exactly_two_objectives_are_compared():
    assert PILOT.objectives == ("huber", "half_mse")
    assert PILOT.huber_delta == HUBER_DELTA == 1.0
    assert PILOT.only_intentional_difference == "the loss function"


def test_selection_is_validation_missense_spearman_with_documented_tie_breakers():
    assert PILOT.selection_metric == "validation missense Spearman"
    assert PILOT.tie_breakers == ("raw-z MAE", "raw-z RMSE")
    assert PILOT.composite_score is False


def test_scaler_conventions_are_pinned_to_the_protocol():
    assert PILOT.scaler_fit_split == "train"
    assert PILOT.scaler_std_ddof == DDOF == 0
    assert PILOT.target_column == "z_score_D4_D14"
    assert "standardised" in PILOT.optimisation_space
    assert "raw z-score" in PILOT.reporting_space


def test_reused_hyperparameters_match_the_established_base_config():
    """These came from configs/base.yaml; Task D introduces no search."""
    import yaml

    base = yaml.safe_load(open("configs/base.yaml"))
    assert PILOT.learning_rate == base["lr"]["head"]
    assert PILOT.weight_decay == base["weight_decay"]
    assert PILOT.batch_size == base["batch_size"]
    assert PILOT.precision == base["precision"]
    assert tuple(PILOT.seeds) == tuple(base["seed"])


def test_two_arms_differing_in_anything_but_the_loss_are_refused():
    assert_fair(PILOT, PilotProtocol())          # identical arms are fine
    for field, bad in [
        ("learning_rate", 3e-4),
        ("batch_size", 64),
        ("seeds", (1,)),
        ("max_epochs", 5),
        ("scaler_fit_split", "val"),
        ("selection_metric", "overall Spearman"),
    ]:
        with pytest.raises(ValueError, match="more than the loss"):
            assert_fair(PILOT, PilotProtocol(**{field: bad}))


def test_every_field_that_could_bias_the_comparison_is_guarded():
    for field in ("learning_rate", "weight_decay", "batch_size", "seeds",
                  "max_epochs", "early_stopping_patience", "optimizer",
                  "scaler_fit_split", "scaler_std_ddof", "cohort",
                  "split_source", "selection_metric", "init_policy"):
        assert field in FAIR_COMPARISON_FIELDS
    # the loss itself must NOT be in the guarded set -- it is the one difference
    assert "objectives" not in FAIR_COMPARISON_FIELDS


def test_the_mock_is_flagged_as_not_a_model():
    from src.stage1 import mock

    assert mock.MOCK_IS_NOT_A_SCIENTIFIC_MODEL is True
    assert "NOT A MODEL" in mock.__doc__
