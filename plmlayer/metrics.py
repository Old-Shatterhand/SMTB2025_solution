"""Task metrics, computed in the pipeline rather than at plot time.

The research code dumps raw prediction pickles and recomputes every number in
``src/viz/utils.py`` at figure-drawing time, keyed on a module-level
``SPLIT_ID`` global. Here the metric is part of the result.

Prediction contract -- probes always emit exactly these shapes, which removes the
``y_hat.ndim == 3`` branching the original needed to cope with
``MultiOutputClassifier`` returning a list:

===============  =====================  ==================
task             ``y_hat``              ``y``
===============  =====================  ==================
regression       ``(N,)`` float         ``(N,)`` float
binary           ``(N,)`` P(class 1)    ``(N,)`` {0, 1}
multi-class      ``(N, C)`` probs       ``(N,)`` int
multi-label      ``(N, C)`` probs       ``(N, C)`` {0, 1}
===============  =====================  ==================
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from plmlayer.types import Task

__all__ = ["TASK_METRICS", "HIGHER_IS_BETTER", "compute_metric", "default_metric"]

# Matches src/viz/constants.py::TASK_METRICS so results stay comparable with the
# paper: Pearson's r for regression, MCC for every classification variant.
TASK_METRICS: dict[str, str] = {
    "regression": "pearson",
    "binary": "mcc",
    "multi-class": "mcc",
    "multi-label": "mcc",
}

HIGHER_IS_BETTER: dict[str, bool] = {
    "pearson": True,
    "spearman": True,
    "r2": True,
    "acc": True,
    "mcc": True,
    "auroc": True,
    "mse": False,
    "mae": False,
    "rmse": False,
}


def default_metric(task: Task) -> str:
    return TASK_METRICS[task]


def multioutput_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-column binary MCC at threshold 0.5.

    Vendored from ``src/downstream/utils.py`` (also duplicated verbatim in
    ``src/viz/utils.py``). Degenerate columns -- all-positive, all-negative, or a
    zero denominator -- contribute 0.0 rather than NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_pred) > 0.5).astype(int)
    scores = []
    for col in range(y_true.shape[1]):
        t, p = y_true[:, col], y_pred[:, col]
        if len(np.unique(t)) < 2 and len(np.unique(p)) < 2:
            scores.append(0.0)
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            score = matthews_corrcoef(t, p)
        scores.append(0.0 if np.isnan(score) else float(score))
    return float(np.mean(scores)) if scores else 0.0


def _as_hard_labels(y_hat: np.ndarray, task: Task) -> np.ndarray:
    if task == "regression":
        return y_hat
    if task == "binary":
        return (y_hat > 0.5).astype(int)
    if task == "multi-class":
        return y_hat.argmax(axis=1) if y_hat.ndim == 2 else y_hat.astype(int)
    return (y_hat > 0.5).astype(int)  # multi-label


def compute_metric(
    y_hat: np.ndarray,
    y: np.ndarray,
    metric: str,
    task: Task,
) -> float:
    """Score predictions. Returns ``nan`` only when the metric is undefined."""
    y_hat, y = np.asarray(y_hat), np.asarray(y)
    m = metric.lower()

    if task == "regression":
        if m == "pearson":
            if np.std(y_hat) == 0 or np.std(y) == 0:
                return float("nan")
            return float(np.corrcoef(y_hat, y)[0, 1])
        if m == "spearman":
            return float(spearmanr(y_hat, y)[0])
        if m == "r2":
            return float(r2_score(y, y_hat))
        if m == "mse":
            return float(mean_squared_error(y, y_hat))
        if m == "mae":
            return float(mean_absolute_error(y, y_hat))
        if m == "rmse":
            return float(np.sqrt(mean_squared_error(y, y_hat)))
        raise ValueError(f"metric {metric!r} is not defined for regression")

    if m == "mcc":
        if task == "multi-label":
            return multioutput_mcc(y, y_hat)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = matthews_corrcoef(y.astype(int), _as_hard_labels(y_hat, task))
        return 0.0 if np.isnan(score) else float(score)

    if m == "acc":
        return float(accuracy_score(y, _as_hard_labels(y_hat, task)))

    if m == "auroc":
        try:
            if task == "multi-class":
                return float(roc_auc_score(y, y_hat, multi_class="ovr"))
            return float(roc_auc_score(y, y_hat))
        except ValueError:  # a class missing from this split
            return float("nan")

    raise ValueError(f"metric {metric!r} is not defined for task {task!r}")
