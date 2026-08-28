"""Metric definitions and the canonical prediction shapes."""

from __future__ import annotations

import numpy as np
import pytest

from plmlayer.metrics import (
    HIGHER_IS_BETTER,
    TASK_METRICS,
    compute_metric,
    default_metric,
    multioutput_mcc,
)


def test_defaults_match_the_paper():
    """src/viz/constants.py: Pearson for regression, MCC for every classification."""
    assert TASK_METRICS == {
        "regression": "pearson",
        "binary": "mcc",
        "multi-class": "mcc",
        "multi-label": "mcc",
    }
    assert default_metric("regression") == "pearson"


@pytest.mark.parametrize(
    # multi-label columns must actually vary: MCC on a constant column is
    # undefined, and multioutput_mcc reports 0.0 there by design.
    "task,y",
    [
        ("regression", np.linspace(0, 1, 100)),
        ("binary", np.tile([0, 1], 50)),
        ("multi-class", np.repeat(np.arange(5), 20)),
        ("multi-label", np.c_[np.tile([0, 1], 50), np.tile([1, 1, 0, 0], 25)]),
    ],
)
def test_perfect_prediction_scores_one(task, y):
    metric = default_metric(task)
    assert compute_metric(y.astype(float), y, metric, task) == pytest.approx(1.0)


def test_multi_class_accepts_probability_matrices():
    y = np.repeat(np.arange(4), 25)
    probs = np.eye(4)[y]
    assert compute_metric(probs, y, "mcc", "multi-class") == pytest.approx(1.0)


def test_constant_prediction_gives_nan_pearson_not_a_crash():
    assert np.isnan(compute_metric(np.ones(10), np.arange(10.0), "pearson", "regression"))


def test_degenerate_column_contributes_zero_not_nan():
    y = np.zeros((50, 2), dtype=int)  # column 0 all-negative
    y[:, 1] = np.tile([0, 1], 25)
    score = multioutput_mcc(y, y.astype(float))
    assert np.isfinite(score) and score == pytest.approx(0.5)


def test_direction_table_covers_every_default_metric():
    for metric in TASK_METRICS.values():
        assert metric in HIGHER_IS_BETTER


def test_error_metrics_are_lower_is_better():
    assert HIGHER_IS_BETTER["rmse"] is False
    y = np.arange(10.0)
    assert compute_metric(y, y, "rmse", "regression") == pytest.approx(0.0)


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="not defined"):
        compute_metric(np.zeros(5), np.zeros(5), "nonsense", "regression")


def test_auroc_missing_class_returns_nan_rather_than_raising():
    y = np.zeros(20, dtype=int)
    assert np.isnan(compute_metric(np.random.rand(20), y, "auroc", "binary"))
