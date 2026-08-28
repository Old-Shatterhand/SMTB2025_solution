"""Layer probes.

kNN is the default, matching the algorithm behind the paper's main figures
(``algo="knn"`` throughout ``src/viz/fig*.py``). It needs no training, has one
hyperparameter, covers all four task types, and -- as the paper argues -- aligns
with the premise that similar proteins have similar embeddings.

Unlike ``src/downstream/probe_layer.py``, which does ``import cuml`` at module
scope and is therefore unusable off a CUDA cluster, cuML here is optional: the
sklearn path is the default and the GPU path is used only when importable.
"""

from __future__ import annotations

import numpy as np

from plmlayer.types import Task

__all__ = ["fit_predict", "PROBES", "cuml_available"]

PROBES = ("knn", "lr")


def cuml_available() -> bool:
    try:
        import cuml  # noqa: F401
    except Exception:
        return False
    return True


def _knn(task: Task, k: int, use_gpu: bool):
    if use_gpu:
        from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor

        cls = KNeighborsRegressor if task == "regression" else KNeighborsClassifier
        return cls(n_neighbors=k, output_type="numpy")
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    cls = KNeighborsRegressor if task == "regression" else KNeighborsClassifier
    return cls(n_neighbors=k, n_jobs=-1)


def _linear(task: Task, use_gpu: bool):
    if task == "regression":
        if use_gpu:
            from cuml import LinearRegression

            return LinearRegression(output_type="numpy")
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    task: Task,
    probe: str = "knn",
    k: int = 10,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Fit on train, predict on val.

    Returns predictions in the canonical shape documented in
    :mod:`plmlayer.metrics`: ``(N,)`` for regression, binary and multi-class;
    ``(N, C)`` for multi-label.
    """
    if probe not in PROBES:
        raise ValueError(f"probe must be one of {PROBES}, got {probe!r}")
    gpu = cuml_available() if use_gpu is None else use_gpu

    if task == "multi-label":
        # One independent binary probe per column, stacked back to (N, C).
        cols = []
        for c in range(train_y.shape[1]):
            cols.append(
                fit_predict(
                    train_x, train_y[:, c], val_x, val_y[:, c],
                    task="binary", probe=probe, k=k, use_gpu=gpu,
                )
            )
        return np.stack(cols, axis=1)

    if probe == "knn":
        # A neighbourhood cannot be larger than the training set.
        model = _knn(task, max(1, min(k, len(train_x))), gpu)
    else:
        model = _linear(task, gpu)

    if task != "regression" and len(np.unique(train_y)) < 2:
        # Degenerate split: a constant predictor is the honest answer.
        return np.full(len(val_x), float(train_y[0]))

    model.fit(train_x, train_y)

    if task == "regression":
        return np.asarray(model.predict(val_x)).ravel()
    if task == "binary":
        proba = np.asarray(model.predict_proba(val_x))
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    # multi-class: hard labels avoid having to map predict_proba columns back
    # through model.classes_ when a class is missing from the training subsample.
    return np.asarray(model.predict(val_x)).ravel()
