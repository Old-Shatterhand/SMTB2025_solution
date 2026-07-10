from pathlib import Path

import numpy as np
import pandas as pd

from src.viz.constants import LAYERS, MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, MODELS
from src.viz.utils_general import compute_metric, read_pca_metric


# Unused
def plot_scope_top4_metric(ax, root, algorithm, metric, level, relative):
    for model in MODELS:
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



