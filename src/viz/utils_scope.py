from pathlib import Path

import numpy as np
import pandas as pd

from src.viz.constants import LAYERS, MODEL_COLORS, MODEL_MARKERS, MODELS
from src.viz.utils_general import compute_metric, read_pca_metric


def compute_scope_performance(
        root: Path, 
        model, 
        dataset, 
        layer, 
        algo, 
        metric, 
        level, 
        k=None, 
        min_x=None
    ):
    assert (k is not None) != (min_x is not None), "Exactly one of k and min_x must be provided."
    with open(root / "embeddings" / model / dataset / f"layer_{layer}" / f"predictions_{algo}_{level}_{k if k is not None else f'min{min_x}'}.pkl", "rb") as f:
        y_hat, y = pd.read_pickle(f)[1]
    return compute_metric(np.array(y_hat), np.array(y), metric, task="multi-class")


# Unused
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


# Unused
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
        "ankh_large": "Ankh-large",
    }
    metric_map = {
        "mcc": "MCC",
        "spearman": "Spearman",
        "pearson": "Pearson",
    }
    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(metric_map[metric])
    ax.set_title(f"{metric_map[metric]} of {algorithm.upper()} heads ({model_map[model]})")


# Unused
def read_correlations(root: Path, model, layer, level, k):
    with open(root / model / "scope_40_208" / f"layer_{layer}" / f"correlations_{level}_{k}.csv", "rb") as f:
        corrs = pd.read_csv(f)
    return corrs["spearman"].values[0], corrs["pearson"].values[0]


def read_metric(root: Path, model, layer, metric, filename):
    if metric in {"zero", "pc@95", "var@10", "5dvol"}:
        return read_pca_metric(root, model, "scope_40_208", layer, metric, filename)
    
    name_map = {"ids": "twonn_id", "density": "density", "noverlap": "neighbor_overlap", "noverlap_50": "neighbor_overlap"}
    filepath = root / "embeddings" / model / "scope_40_208" / f"layer_{layer}" / filename
    if not filepath.exists():
        return 0
    df = pd.read_csv(filepath)
    return df[name_map[metric]].values[0]


def plot_scope_minx_performance(ax, root, algorithm, metric, level, model_prefix, relative: bool = True, colored: bool | str = True, title: str | None = None):
    for model in MODELS[:-1]:
        perfs = [compute_scope_performance(root, model_prefix + model, "scope_40_208", layer, algorithm, metric, level, min_x=10) for layer in range(LAYERS[model] + 1)]
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        ax.plot(
            np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])) if relative else np.arange(LAYERS[model] + 1), 
            perfs, 
            label=model_prefix + model, 
            color=MODEL_COLORS.get(model, None) if colored == True else colored, 
            marker=MODEL_MARKERS.get(model, None) if colored == True else None
        )

    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} of {algorithm.upper()} heads")
    ax.set_ylim(bottom=-0.05, top=1.05)


def plot_scope_minx_metric(ax, root, metric, level, model_prefix, relative):
    title_map = {"ids": "Intrinsic Dimensions", "density": "Density", "noverlap": "Neighbor Overlap", "noverlap_50": "Neighbor Overlap (50)"}
    for model in MODELS[:-1]:
        perfs = []
        for layer in range(LAYERS[model] + 1):
            if metric.startswith("noverlap") and layer == LAYERS[model]:
                continue
            if metric == "ids":
                result = read_metric(root, model_prefix + model, layer, metric, f"{metric}_{level}_min10.csv")
            elif metric in {"zero", "pc@95", "var@10", "5dvol"}:
                result = read_metric(root, model_prefix + model, layer, metric, f"pca_10_{level}_min10.pkl")
            else:
                result = read_metric(root, model_prefix + model, layer, metric, f"{metric}_{level}_min10_10.csv")
            perfs.append(result)
        if relative:
            x_ticks = np.arange(0, 1 + 1e-5, 1 / (LAYERS[model]))
            if metric.startswith("noverlap"):
                x_ticks = x_ticks[:-1]
                x_ticks += 1 / (2 * LAYERS[model])
            ax.plot(x_ticks, perfs, label=model_prefix + model, color=MODEL_COLORS.get(model, None), marker=MODEL_MARKERS.get(model, None))
        else:
            ax.plot(perfs, label=model_prefix + model, color=MODEL_COLORS.get(model, None), marker=MODEL_MARKERS.get(model, None))
    
    if metric == "5dvol":
        ax.set_yscale("log")
    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(title_map.get(metric, metric))
    ax.set_title(title_map.get(metric, metric))
    if metric not in {"ids", "density", "5dvol"}:
        ax.set_ylim(bottom=-0.05, top=1.05)
