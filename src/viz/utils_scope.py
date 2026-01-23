from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

from src.downstream.utils import multioutput_mcc
from src.viz.constants import LAYERS, MODEL_COLORS, MODELS


def compute_scope_performance(root: Path, model, dataset, layer, algo, metric, level, k=None, min_x=None):
    assert (k is not None) != (min_x is not None), "Exactly one of k and min_x must be provided."
    with open(root / "embeddings" / model / dataset / f"layer_{layer}" / f"predictions_{algo}_{level}_{k if k is not None else f'min{min_x}'}.pkl", "rb") as f:
        y_hat, y = pd.read_pickle(f)[1]
    match metric.lower():
        case "mcc":
            if y_hat.ndim == 3:  # multilabel
                return multioutput_mcc(y, np.array(y_hat)[:, :, 1].T)
            if y_hat.ndim == 2:
                return matthews_corrcoef(y, y_hat.argmax(axis=1))
            return matthews_corrcoef(y, y_hat > 0.5)
        case "auroc":
            if y_hat.ndim == 3:  # multilabel
                return roc_auc_score(y, np.array(y_hat)[:, :, 1].T, multi_class="ovr")
            return roc_auc_score(y, y_hat, multi_class="ovr")
        case "acc":
            return accuracy_score(y, y_hat)
        case _:
            raise ValueError(f"Unknown metric: {metric}")


def plot_scope_top4_metric(ax, root, algorithm, metric, level, relative):
    for model in MODELS[:-1]:
        if model.startswith("esmc"):
            continue
        perfs = []
        for layer in range(LAYERS[model] + 1):
            result = compute_scope_performance(root, model, "scope_40_208", layer, algorithm, metric, level, k=4)
            perfs.append(result)
        if relative:
            ax.plot(np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])), perfs, label=model, color=MODEL_COLORS.get(model, None))
        else:
            ax.plot(perfs, label=model, color=MODEL_COLORS.get(model, None))

    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} of {algorithm.upper()} heads ({level.capitalize()} Top-4)")


def plot_scope_topX_metric(ax, root, model, algorithm, metric, level, relative, correlation: bool = False):
    for k in [4, 6, 8, 10, 15, 20]:
        perfs = []
        for layer in range(LAYERS[model] + 1):
            if correlation:
                spearman, pearson = read_correlations(root, model, layer, level, k)
                result = spearman if metric == "spearman" else pearson
            else:
                result = compute_scope_performance(root, model, "scope_40_208", layer, algorithm, metric, level, k)
            perfs.append(result)
        if relative:
            ax.plot(np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])), perfs, label=f"Top {k}")  # , c=MODEL_COLORS.get(model, None))
        else:
            ax.plot(perfs, label=model)  # , c=MODEL_COLORS.get(model, None))

    model_map = {
        "prostt5": "ProstT5",
        "prott5": "ProtT5",
        "ankh-large": "Ankh-large",
    }
    metric_map = {
        "mcc": "MCC",
        "spearman": "Spearman",
        "pearson": "Pearson",
    }
    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(metric_map[metric])
    ax.set_title(f"{metric_map[metric]} of {algorithm.upper()} heads ({model_map[model]})")


def read_correlations(root: Path, model, layer, level, k):
    with open(root / model / "scope_40_208" / f"layer_{layer}" / f"correlations_{level}_{k}.csv", "rb") as f:
        corrs = pd.read_csv(f)
    return corrs["spearman"].values[0], corrs["pearson"].values[0]


def read_metric(root: Path, model, layer, metric, filename):
    name_map = {"ids": "twonn_id", "density": "density", "noverlap": "neighbor_overlap", "noverlap_50": "neighbor_overlap"}
    filepath = root / "embeddings" / model / "scope_40_208" / f"layer_{layer}" / filename
    if not filepath.exists():
        return 0
    df = pd.read_csv(filepath)
    return df[name_map[metric]].values[0]


def plot_scope_minx_performance(ax, root, algorithm, metric, level, model_prefix, relative: bool = True, colored: bool | str = True, title: str | None = None):
    for model in MODELS[:-1]:
        perfs = []
        for layer in range(LAYERS[model] + 1):
            result = compute_scope_performance(root, model_prefix + model, "scope_40_208", layer, algorithm, metric, level, min_x=10)
            perfs.append(result)
        if relative:
            ax.plot(np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])), perfs, label=model_prefix + model, color=MODEL_COLORS.get(model, None) if colored == True else colored)
        else:
            ax.plot(perfs, label=model_prefix + model, color=MODEL_COLORS.get(model, None) if colored == True else colored)
    
    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} of {algorithm.upper()} heads")


def plot_scope_minx_metric(ax, root, algorithm, metric, level, model_prefix, relative):
    title_map = {"ids": "Intrinsic Dimensions", "density": "Density", "noverlap": "Neighbor Overlap", "noverlap_50": "Neighbor Overlap (50)"}
    for model in MODELS[:-1]:
        perfs = []
        for layer in range(LAYERS[model] + 1):
            if metric.startswith("noverlap") and layer == LAYERS[model]:
                continue
            if metric == "ids":
                result = read_metric(root, model_prefix + model, layer, metric, f"{metric}_{level}_min10.csv")
            else:
                result = read_metric(root, model_prefix + model, layer, metric, f"{metric}_{level}_min10_10.csv")
            perfs.append(result)
        if relative:
            x_ticks = np.arange(0, 1 + 1e-5, 1 / (LAYERS[model]))
            if metric.startswith("noverlap"):
                x_ticks = x_ticks[:-1]
                x_ticks += 1 / (2 * LAYERS[model])
            ax.plot(x_ticks, perfs, label=model_prefix + model, color=MODEL_COLORS.get(model, None))
        else:
            ax.plot(perfs, label=model_prefix + model, color=MODEL_COLORS.get(model, None))
    
    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(title_map[metric])
    ax.set_title(f"{title_map[metric]} of {algorithm.upper()} heads")
