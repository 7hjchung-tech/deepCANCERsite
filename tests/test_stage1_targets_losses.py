"""Target standardisation, the two objectives, and the metrics.

Model-free and cache-free: these are the numerical contracts that decide what
"the loss" and "the MSE" mean, so they are pinned independently of any data.
"""

import numpy as np
import pytest
import torch

from src.stage1.losses import (
    HUBER_DELTA,
    OBJECTIVES,
    conventional_mse,
    get_objective,
    half_mse_loss,
    huber_loss,
)
from src.stage1.metrics import (
    evaluate_split,
    regression_metrics,
    selection_score,
    spearman,
)
from src.stage1.targets import MIN_STD, TargetScaler

RNG = np.random.default_rng(0)


# ==========================================================================
# target scaler
# ==========================================================================
def test_scaler_uses_train_statistics_and_records_the_convention():
    y_train = RNG.normal(-3.0, 5.0, size=400)
    s = TargetScaler.fit_on_train(y_train)
    assert s.mean == pytest.approx(y_train.mean())
    assert s.std == pytest.approx(y_train.std(ddof=0))
    assert s.ddof == 0
    assert "ddof=0" in s.std_convention
    assert s.n_train_rows == 400
    assert s.fit_split == "train"
    assert s.target_column == "z_score_D4_D14"


def test_val_and_test_reuse_the_train_scaler_never_refit():
    y_train = RNG.normal(-3.0, 5.0, size=300)
    y_val = RNG.normal(+8.0, 0.5, size=90)          # deliberately different location
    s = TargetScaler.fit_on_train(y_train)

    std_val = s.transform(y_val)
    # standardising with TRAIN statistics must NOT centre val on zero
    assert abs(std_val.mean()) > 1.0
    # and must not rescale val to unit variance either
    assert not np.isclose(std_val.std(), 1.0, atol=0.1)
    # the train split, by construction, does land on 0 / 1
    std_train = s.transform(y_train)
    assert std_train.mean() == pytest.approx(0.0, abs=1e-12)
    assert std_train.std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_inverse_transform_round_trips_exactly():
    y = RNG.normal(-3.0, 5.0, size=500)
    s = TargetScaler.fit_on_train(y)
    assert np.abs(s.inverse_transform(s.transform(y)) - y).max() < 1e-9
    z = RNG.normal(0, 1, size=500)
    assert np.abs(s.transform(s.inverse_transform(z)) - z).max() < 1e-9


def test_inverse_transform_is_the_documented_affine_map():
    s = TargetScaler.fit_on_train(RNG.normal(-3.0, 5.0, size=100))
    z = np.array([-2.0, 0.0, 1.5])
    assert s.inverse_transform(z) == pytest.approx(z * s.std + s.mean)


def test_scaler_refuses_a_degenerate_or_invalid_split():
    with pytest.raises(ValueError, match="constant"):
        TargetScaler.fit_on_train(np.full(50, 3.0))
    with pytest.raises(ValueError, match="at least 2"):
        TargetScaler.fit_on_train([1.0])
    with pytest.raises(ValueError, match="non-finite"):
        TargetScaler.fit_on_train([1.0, 2.0, np.nan])
    with pytest.raises(ValueError):
        TargetScaler.fit_on_train([1.0, 1.0 + MIN_STD / 10])


def test_scaler_serialises_round_trip():
    s = TargetScaler.fit_on_train(RNG.normal(0, 2, size=50))
    assert TargetScaler.from_dict(s.to_dict()) == s


# ==========================================================================
# objectives vs metrics -- the distinction Task D turns on
# ==========================================================================
def test_huber_is_delta_one_smooth_l1_in_standardised_units():
    assert HUBER_DELTA == 1.0
    pred = torch.tensor([0.0, 0.5, 3.0, -4.0])
    targ = torch.tensor([0.0, 0.0, 0.0, 0.0])
    got = huber_loss(pred, targ)

    # closed form: 0.5*e^2 for |e|<=1, else |e|-0.5
    e = (pred - targ).abs()
    want = torch.where(e <= 1.0, 0.5 * e ** 2, e - 0.5).mean()
    assert float(got) == pytest.approx(float(want))
    # and it agrees with torch's own HuberLoss(delta=1.0)
    assert float(got) == pytest.approx(float(torch.nn.HuberLoss(delta=1.0)(pred, targ)))


def test_huber_is_quadratic_inside_delta_and_linear_outside():
    targ = torch.zeros(1)
    small = float(huber_loss(torch.tensor([0.5]), targ))
    assert small == pytest.approx(0.5 * 0.5 ** 2)          # quadratic branch
    big = float(huber_loss(torch.tensor([10.0]), targ))
    assert big == pytest.approx(10.0 - 0.5)                # linear branch
    # linearity outside delta is the whole point: doubling the error does not
    # quadruple the loss
    bigger = float(huber_loss(torch.tensor([20.0]), targ))
    assert bigger == pytest.approx(20.0 - 0.5)
    assert bigger < 4 * big


def test_half_mse_is_exactly_half_of_standardised_mse():
    pred = torch.tensor([0.3, -1.2, 2.0, 0.0])
    targ = torch.tensor([0.0, -1.0, 1.0, 0.5])
    mse_std = float(torch.mean((pred - targ) ** 2))
    assert float(half_mse_loss(pred, targ)) == pytest.approx(0.5 * mse_std)


def test_half_mse_loss_is_not_the_conventional_mse_metric():
    """The 0.5 belongs to the optimisation loss only, and the metric is raw."""
    pred_std = torch.tensor([0.3, -1.2, 2.0, 0.0])
    targ_std = torch.tensor([0.0, -1.0, 1.0, 0.5])
    s = TargetScaler.fit_on_train(RNG.normal(-3.0, 5.0, size=200))

    loss = float(half_mse_loss(pred_std, targ_std))
    pred_raw = s.inverse_transform(pred_std.numpy())
    targ_raw = s.inverse_transform(targ_std.numpy())
    metric = conventional_mse(pred_raw, targ_raw)

    mse_std = float(torch.mean((pred_std - targ_std) ** 2))
    assert loss == pytest.approx(0.5 * mse_std)            # loss: halved, standardised
    assert metric == pytest.approx(s.std ** 2 * mse_std)   # metric: raw scale, no 0.5
    assert metric != pytest.approx(loss)
    assert metric == pytest.approx(2.0 * s.std ** 2 * loss)


def test_the_two_objectives_are_the_only_ones_and_are_selectable_by_name():
    assert set(OBJECTIVES) == {"huber", "half_mse"}
    assert get_objective("huber") is huber_loss
    assert get_objective("half_mse") is half_mse_loss
    for banned in ("mae", "log_cosh", "ranking", "focal", "mse"):
        with pytest.raises(KeyError):
            get_objective(banned)


def test_huber_and_half_mse_differ_on_an_outlier():
    """Sanity: the comparison is not vacuous -- the two losses really differ."""
    pred = torch.tensor([0.0, 0.0, 0.0])
    targ = torch.tensor([0.1, 0.1, 12.0])
    assert float(huber_loss(pred, targ)) != pytest.approx(float(half_mse_loss(pred, targ)))
    assert float(huber_loss(pred, targ)) < float(half_mse_loss(pred, targ))


# ==========================================================================
# metrics
# ==========================================================================
def test_regression_metrics_are_in_raw_units():
    t = np.array([0.0, 1.0, 2.0])
    p = np.array([0.0, 2.0, 0.0])
    m = regression_metrics(t, p)
    assert m["raw_mae"] == pytest.approx((0 + 1 + 2) / 3)
    assert m["raw_mse"] == pytest.approx((0 + 1 + 4) / 3)
    assert m["raw_rmse"] == pytest.approx(np.sqrt((0 + 1 + 4) / 3))
    assert m["n"] == 3


def test_spearman_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for _ in range(5):
        a = RNG.normal(size=60)
        b = RNG.normal(size=60)
        assert float(spearman(a, b)) == pytest.approx(
            float(scipy_stats.spearmanr(a, b).statistic)
        )
    # with ties
    a = RNG.integers(0, 3, size=40).astype(float)
    b = RNG.integers(0, 4, size=40).astype(float)
    assert float(spearman(a, b)) == pytest.approx(
        float(scipy_stats.spearmanr(a, b).statistic)
    )


def test_spearman_is_invariant_to_the_target_scaler():
    s = TargetScaler.fit_on_train(RNG.normal(-3.0, 5.0, size=300))
    t_raw = RNG.normal(-3, 5, size=120)
    p_raw = t_raw + RNG.normal(0, 2, size=120)
    rho_raw = float(spearman(t_raw, p_raw))
    rho_std = float(spearman(s.transform(t_raw), s.transform(p_raw)))
    assert rho_raw == pytest.approx(rho_std, abs=1e-12)


def test_undefined_spearman_is_reported_as_undefined_not_as_nan():
    const_pred = spearman([1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0])
    assert const_pred.value is None
    assert const_pred.undefined_reason == "constant_predictions"
    assert not const_pred.is_defined
    with pytest.raises(ValueError, match="undefined"):
        float(const_pred)

    assert spearman([1.0, 2.0], [1.0, 2.0]).undefined_reason.startswith("too_few_rows")
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]).undefined_reason == "constant_targets"
    assert spearman([1.0, 2.0, 3.0], [1.0, np.nan, 3.0]).undefined_reason == (
        "non_finite_predictions"
    )


def test_an_undefined_selection_score_is_none_so_it_cannot_win_silently():
    report = evaluate_split(
        var_id=["a", "b", "c"],
        variant_type=["missense"] * 3,
        y_true_raw=[1.0, 2.0, 3.0],
        y_pred_raw=[5.0, 5.0, 5.0],          # collapsed model
        y_true_std=[0.1, 0.2, 0.3],
        y_pred_std=[0.5, 0.5, 0.5],
        optimization_loss=0.42,
        objective="huber",
    )
    assert selection_score(report) is None
    assert report["missense_spearman"]["undefined_reason"] == "constant_predictions"
    # a real score, by contrast, is a float
    good = evaluate_split(
        var_id=["a", "b", "c"], variant_type=["missense"] * 3,
        y_true_raw=[1.0, 2.0, 3.0], y_pred_raw=[1.5, 1.9, 3.3],
        y_true_std=[0.1, 0.2, 0.3], y_pred_std=[0.15, 0.19, 0.33],
    )
    assert isinstance(selection_score(good), float)


def test_evaluate_split_separates_missense_from_descriptive_groups():
    report = evaluate_split(
        var_id=list("abcdefgh"),
        variant_type=["missense"] * 5 + ["inframe_indel"] * 3,
        y_true_raw=[1, 2, 3, 4, 5, 9, 8, 7],
        y_pred_raw=[1, 2, 3, 5, 4, 7, 8, 9],
        y_true_std=[0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.8, 0.7],
        y_pred_std=[0.1, 0.2, 0.3, 0.5, 0.4, 0.7, 0.8, 0.9],
    )
    assert report["missense_spearman"]["n"] == 5
    assert report["overall_spearman"]["n"] == 8
    assert set(report["by_variant_type_descriptive"]) == {"missense", "inframe_indel"}
    # the indel numbers exist but are not the selection criterion
    assert selection_score(report) == report["missense_spearman"]["value"]
    assert report["spearman_raw_vs_standardized_absdiff"] == pytest.approx(0.0, abs=1e-12)
