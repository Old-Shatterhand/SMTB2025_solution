from pathlib import Path
from typing import Literal

from matplotlib import pyplot as plt, transforms
import numpy as np

from src.viz.utils import compute_performance, compute_scope_performance, read_metric, read_pca_metric, read_scope_metric
from src.viz.constants import LAYERS, MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, MODELS


def plot_performance(
        ax, 
        root: Path, 
        dataset: str, 
        algo: str, 
        metric: str, 
        task: Literal["regression", "binary", "multi-label", "multi-class"] = "regression",
        aa: bool = False, 
        n_classes: int = 42,
        model_prefix: Literal["", "empty_"] = "", 
        relative: bool = False, 
        models: list[str] = MODELS,
        colored: bool | str = True, 
    ) -> dict[str, list[float]]:
    """
    Plot performance metrics for different models on a given axis.

    Args:
        ax: Matplotlib axis to plot on.
        root: Root directory containing the embeddings and results.
        dataset: Name of the dataset.
        algo: Algorithm used (e.g., "lr", "knn").
        metric: Performance metric to plot (e.g., "pearson", "mcc").
        relative: Whether to plot relative layer positions.
        model_prefix: Prefix to add to model names. Either "" or "empty_" to indicate using normal or untrained models.
        legend: Whether to display the legend.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
        task: Type of task ("regression", "binary", "multi-label", "multi-class").
        models: List of model names to include in the plot.
        colored: Color setting for the plot lines.
    
    Returns:
        A dictionary mapping model names to their performance metrics across layers.
    """
    performances = {}
    for model in models:
        perfs = [compute_performance(root, model_prefix + model, dataset, layer, algo=algo, metric=metric, aa=aa, n_classes=n_classes, task=task) for layer in range(LAYERS[model] + 1)]
        performances[model] = perfs
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        ax.plot(
            np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])) if relative else np.arange(len(perfs)),
            perfs, 
            label=model_prefix + MODEL_NAMES.get(model, model), 
            c=MODEL_COLORS.get(model, None) if colored == True else colored,
            marker=MODEL_MARKERS.get(model, None) if colored == True else None,
        )
    return performances


def plot_metric(
        ax, 
        root: Path | None, 
        dataset: str | None, 
        relative: bool, 
        metric: Literal["ids", "density", "noverlap", "noverlap_50", "zero", "pc@95", "var@10", "5dvol"] = "ids", 
        model_prefix: Literal["", "empty_"] = "", 
        legend: bool = False, 
        aa: bool = False, 
        n_classes: int = 42,
        models: list[str] = MODELS,
        colored: bool | str = True,
        title: str | bool | None = None,
    ) -> None:
    """
    Plot a specific metric for different models on a given axis.

    Args:
        ax: Matplotlib axis to plot on.
        root: Root directory containing the embeddings and results.
        dataset: Name of the dataset.
        relative: Whether to plot relative layer positions.
        metric: Metric to plot ("ids", "density", "noverlap", "noverlap_50", "zero", "pc@95", "var@10", "5dvol").
        model_prefix: Prefix to add to model names. Either "" or "empty_" to indicate using normal or untrained models.
        legend: Whether to display the legend.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
        models: List of model names to include in the plot.
        colored: Color setting for the plot lines.
        title: Optional title for the plot. Can be a string, boolean, or None.
    """
    title_map = {"ids": "Intrinsic Dimensions", "density": "Density", "noverlap": "Neighbor Overlap", "noverlap_50": "Neighbor Overlap (50)"}
    for model in models:
        # if metric == "5dvol" and model.startswith("ankh"):
        #     continue  # ankh is crazy in this metric
        perfs = []
        for layer in range(LAYERS[model] + 1):
            if metric.startswith("noverlap") and layer == LAYERS[model]:
                continue
            if metric in {"zero", "pc@95", "var@10", "5dvol"}:
                result = read_pca_metric(root, model_prefix + model, dataset, layer, metric=metric, aa=aa)
            else:
                result = read_metric(root, model_prefix + model, dataset, layer, metric=metric, aa=aa)
            perfs.append(result)
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        if relative:
            x_ticks = np.arange(0, 1 + 1e-5, 1 / LAYERS[model])
            if metric.startswith("noverlap"):
                x_ticks = x_ticks[:-1]
                x_ticks += 1 / (2 * LAYERS[model])
            ax.plot(
                x_ticks, 
                perfs, 
                label=model_prefix + MODEL_NAMES.get(model, model), 
                c=MODEL_COLORS.get(model, None) if colored == True else colored, 
                marker=MODEL_MARKERS.get(model, None)
            )
        else:
            ax.plot(
                perfs, 
                label=model_prefix + MODEL_NAMES.get(model, model), 
                c=MODEL_COLORS.get(model, None) if colored == True else colored, 
                marker=MODEL_MARKERS.get(model, None)
            )


def plot_scope_minx_performance(
        ax, 
        root: Path, 
        algorithm: str, 
        metric: str, 
        level: Literal["fold", "superfamily"],
        model_prefix: str = "", 
        relative: bool = True, 
        colored: bool | str = True, 
        models: list[str] = MODELS
    ) -> None:
    """
    Plot performance metrics for different models on a given axis for the scope_minx task.

    Args:
        ax: Matplotlib axis to plot on.
        root: Root directory containing the embeddings and results.
        algorithm: Algorithm used (e.g., "lr", "knn").
        metric: Performance metric to plot (e.g., "pearson", "mcc").
        level: Level of the task.
        model_prefix: Prefix to add to model names. Either "" or "empty_" to indicate using normal or untrained models.
        relative: Whether to plot relative layer positions.
        colored: Color setting for the plot lines.
        models: List of model names to include in the plot.
    """
    for model in models:
        perfs = [compute_scope_performance(root, model_prefix + model, "scope_40_208", layer, algorithm, metric, level, min_x=10) for layer in range(LAYERS[model] + 1)]
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        ax.plot(
            np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])) if relative else np.arange(LAYERS[model] + 1), 
            perfs, 
            label=model_prefix + MODEL_NAMES.get(model, model), 
            color=MODEL_COLORS.get(model, None) if colored == True else colored, 
            marker=MODEL_MARKERS.get(model, None) if colored == True else None
        )


def plot_scope_minx_metric(
        ax, 
        root: Path, 
        metric: str, 
        level: str, 
        model_prefix: str = "", 
        relative: bool = True, 
        models: list[str] = MODELS,
    ):
    for model in models:
        perfs = []
        for layer in range(LAYERS[model] + 1):
            if metric.startswith("noverlap") and layer == LAYERS[model]:
                continue
            if metric in {"zero", "pc@95", "var@10", "5dvol"}:
                result = read_pca_metric(root, model_prefix + model, "scope_40_208", layer, metric, f"pca_{level}_min10.pkl")
            else:
                result = read_scope_metric(root, model_prefix + model, layer, metric, f"{metric}_{level}_min10.csv")
            perfs.append(result)
        if relative:
            x_ticks = np.arange(0, 1 + 1e-5, 1 / (LAYERS[model]))
            if metric.startswith("noverlap"):
                x_ticks = x_ticks[:-1]
                x_ticks += 1 / (2 * LAYERS[model])
            ax.plot(x_ticks, perfs, label=model_prefix + MODEL_NAMES.get(model, model), color=MODEL_COLORS.get(model, None), marker=MODEL_MARKERS.get(model, None))
        else:
            ax.plot(perfs, label=model_prefix + MODEL_NAMES.get(model, model), color=MODEL_COLORS.get(model, None), marker=MODEL_MARKERS.get(model, None))
    
    if metric == "5dvol":
        ax.set_yscale("log")


def set_subplot_label(ax: plt.Axes, fig: plt.Figure, label: str) -> None:
    """
    Set the label for a subplot.
    Args:
        ax: The subplot
        fig: The figure
        label: The label to set
    """
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + transforms.ScaledTranslation(
            -25 / 72,
            10 / 72,
            fig.dpi_scale_trans
        ),
        fontsize="x-large",
        va="bottom",
        fontfamily="serif",
    )