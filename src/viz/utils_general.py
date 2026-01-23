from pathlib import Path
from typing import Literal, Optional

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef, roc_auc_score, accuracy_score

from src.viz.constants import MODEL_COLORS, MODELS, SPLIT_ID, LAYERS
from src.downstream.utils import multioutput_mcc


def compute_performance(
        root: Path | None = None, 
        model: str | None = None, 
        dataset: str | None = None, 
        layer: int | None = None, 
        algo: str | None = None, 
        metric: str | None = None, 
        filepath: Path | None = None, 
        aa: bool = False, 
        n_classes: int = 42
    ) -> float:
    """
    Compute a performance metric for the given model, dataset, layer, and algorithm.

    Args:
        root: Root directory containing the embeddings and results. [path parameter]
        model: Name of the model. [path parameter]
        dataset: Name of the dataset. [path parameter]
        layer: Layer number. [path parameter]
        algo: Algorithm used (e.g., "lr", "knn").
        metric: Performance metric to compute (e.g., "pearson", "mcc").
        filepath: Optional path to the results file. If None, constructs the path. This overwrites the other path parameters.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
    
    Returns:
        The computed performance metric.
    """
    embed_dir = "aa_embeddings" if aa else "embeddings"
    if filepath is None:
        filepath = root / embed_dir / model / dataset / f"layer_{layer}" / f"predictions_{algo}_{n_classes}.pkl"
    if not filepath.exists():
        return 0
    with open(filepath, "rb") as f:
        y_hat, y = pd.read_pickle(f)[SPLIT_ID]
        y_hat = np.array(y_hat).squeeze()
        y = np.array(y)
    
    match metric.lower():
        case "pearson":
            return np.corrcoef(y_hat, y)[0, 1]
        case "spearman":
            return spearmanr(y_hat, y)[0]
        case "r2":
            return r2_score(y, y_hat)
        case "mse":
            return mean_squared_error(y, y_hat)
        case "mae":
            return mean_absolute_error(y, y_hat)
        case "rmse":
            return np.sqrt(mean_squared_error(y, y_hat))
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
        case _:
            raise ValueError(f"Unknown metric: {metric}")


def plot_performance(
        ax, 
        root: Path, 
        dataset: str, 
        algo: str, 
        metric: str, 
        relative: bool = False, 
        model_prefix: Literal["", "empty_"] = "", 
        legend: bool = False, 
        colored: bool | str = True, 
        title: str | None = None, 
        aa: bool = False, 
        n_classes: int = 42
    ) -> None:
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
        colored: Color setting for the plot lines.
        title: Optional title for the plot.
    """
    for model in MODELS[:-1]:
        perfs = []
        for layer in range(LAYERS[model] + 1):
            result = compute_performance(root, model_prefix + model, dataset, layer, algo, metric, aa=aa, n_classes=n_classes)
            perfs.append(result)
        if sum(perfs) == 0:
            continue
        if relative:
            ax.plot(np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])), perfs, label=model_prefix + model, c=MODEL_COLORS.get(model, None) if colored == True else colored)
        else:
            ax.plot(perfs, label=model_prefix + model, c=MODEL_COLORS.get(model, None) if colored == True else colored)

    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} of {algo.upper()} heads")
    if legend:
        ax.legend()


def read_metric(
        root: Path | None, 
        model: str | None, 
        dataset: str | None, 
        layer: int | None, 
        metric: Literal["ids", "density", "noverlap", "noverlap_50"], 
        filepath: Path | None = None, 
        aa: bool = False, 
        n_classes: int = 42
    ) -> float:
    """
    Read a specific metric from a CSV file for the given model, dataset, and layer.

    Args:
        root: Root directory containing the embeddings and results. [path parameter]
        model: Name of the model. [path parameter]
        dataset: Name of the dataset. [path parameter]
        layer: Layer number. [path parameter]
        metric: Metric to read ("ids", "density", "noverlap", "noverlap_50").
        filepath: Optional path to the CSV file. If None, constructs the path. This overwrites the other path parameters.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
    
    Returns:
        The value of the specified metric.
    """
    name_map = {"ids": "twonn_id", "density": "density", "noverlap": "neighbor_overlap", "noverlap_50": "neighbor_overlap"}
    if dataset == "deeploc2_bin":  # The metrics for the binary deeploc version and the multi-class version are the same together
        dataset = "deeploc2"
    if filepath is None:
        if aa:
            filepath = root / "aa_embeddings" / model / dataset / f"layer_{layer}" / f"{'noverlap_10' if metric == 'noverlap' else metric}_{n_classes}.csv"
        else:
            filepath = root / "embeddings" / model / dataset / f"layer_{layer}" / f"{'noverlap_10' if metric == 'noverlap' else metric}.csv"
    if not filepath.exists():
        return 0
    df = pd.read_csv(filepath)
    return df[name_map[metric]].values[0]


def plot_metric(
        ax, 
        root: Path | None, 
        dataset: str | None, 
        relative: bool, 
        legend: bool = False, 
        metric: Literal["ids", "density", "noverlap", "noverlap_50"] = "ids", 
        model_prefix: Literal["", "empty_"] = "", 
        aa: bool = False, 
        n_classes: int = 42
    ) -> None:
    """
    Plot a specific metric for different models on a given axis.

    Args:
        ax: Matplotlib axis to plot on.
        root: Root directory containing the embeddings and results.
        dataset: Name of the dataset.
        relative: Whether to plot relative layer positions.
        legend: Whether to display the legend.
        metric: Metric to plot ("ids", "density", "noverlap", "noverlap_50").
        model_prefix: Prefix to add to model names. Either "" or "empty_" to indicate using normal or untrained models.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
    """
    title_map = {"ids": "Intrinsic Dimensions", "density": "Density", "noverlap": "Neighbor Overlap", "noverlap_50": "Neighbor Overlap (50)"}
    for model in MODELS[:-1]:
        perfs = []
        for layer in range(LAYERS[model] + 1):
            if metric.startswith("noverlap") and layer == LAYERS[model]:
                continue
            result = read_metric(root, model_prefix + model, dataset, layer, metric, aa=aa, n_classes=n_classes)
            perfs.append(result)
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        if relative:
            x_ticks = np.arange(0, 1 + 1e-5, 1 / (LAYERS[model]))
            if metric.startswith("noverlap"):
                x_ticks = x_ticks[:-1]
                x_ticks += 1 / (2 * LAYERS[model])
            ax.plot(x_ticks, perfs, label=model_prefix + model, c=MODEL_COLORS.get(model, None))
        else:
            ax.plot(perfs, label=model_prefix + model, c=MODEL_COLORS.get(model, None))

    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(title_map[metric])
    ax.set_title(f"{title_map[metric]}")
    if legend:
        ax.legend(loc="upper right")
