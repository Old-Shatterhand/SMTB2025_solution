import pickle
from pathlib import Path
from typing import Literal

import scipy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef, roc_auc_score

from src.viz.constants import MODEL_COLORS, MODEL_MARKERS, MODELS, SPLIT_ID, LAYERS, DATASET2TASK
from src.downstream.utils import multioutput_mcc


FINETUNE_LAYERS = [0, 10, 15, 20, 22, 24, 26, 28, 30]


def compute_metric(y_hat: np.ndarray, y: np.ndarray, metric: str, task: Literal["regression", "binary", "multi-label", "multi-class"]) -> float:
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
            if task == "multi-label":
                if y_hat.ndim == 3:
                    return multioutput_mcc(y.astype(int), np.array(y_hat)[:, :, 1].T)
                else:
                    return multioutput_mcc(y.astype(int), y_hat)
            if task == "multi-class":
                if y_hat.ndim == 3:
                    return matthews_corrcoef(y.astype(int), y_hat[0].argmax(axis=1))
                if y_hat.ndim == 1:
                    return matthews_corrcoef(y.astype(int), y_hat)
                return matthews_corrcoef(y.astype(int), y_hat.argmax(axis=1))
            if y_hat.ndim == 2:
                return matthews_corrcoef(y.astype(int), y_hat[:, 1] > 0.5)
            return matthews_corrcoef(y.astype(int), y_hat > 0.5)
        case "auroc":
            if task == "multi-label":
                if y_hat.ndim == 3:
                    return roc_auc_score(y.astype(int), np.array(y_hat)[:, :, 1].T, multi_class="ovr")
                return roc_auc_score(y.astype(int), y_hat)
            if task == "binary":
                if y_hat.ndim == 2:
                    return roc_auc_score(y.astype(int), y_hat[:, 1])
                return roc_auc_score(y.astype(int), y_hat)
            return roc_auc_score(y.astype(int), y_hat, multi_class="ovr")
        case _:
            raise ValueError(f"Unknown metric: {metric}")


def compute_performance(
        root: Path | None = None, 
        model: str | None = None, 
        dataset: str | None = None, 
        layer: int | None = None, 
        algo: str | None = None, 
        metric: str | None = None, 
        filepath: Path | None = None, 
        aa: bool = False, 
        n_classes: int = 42,
        task: Literal["regression", "binary", "multi-label", "multi-class"] = "regression"
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
        task: Type of task ("regression", "binary", "multi-label", "multi-class").
    
    Returns:
        The computed performance metric.
    """
    # try:
    embed_dir, filename = "embeddings", f"predictions_{algo}.pkl"
    if aa:
        embed_dir = "aa_embeddings"
        if dataset == "scope_40_208":
            filename = f"predictions_{algo}_{n_classes}.pkl"
    if filepath is None:
        filepath = root / embed_dir / model / dataset / f"layer_{layer}" / filename
    
    if not filepath.exists():
        return 0
    
    with open(filepath, "rb") as f:
        y_hat, y = pd.read_pickle(f)[SPLIT_ID]
        y_hat = np.array(y_hat)
        y = np.array(y)
    
    return compute_metric(y_hat, y, metric, task)
    # except Exception as e:
    #     print(f"Error computing performance for {model} layer {layer} on {dataset} with {algo}: {e}")
    #     return 0


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
        n_classes: int = 42,
        task: Literal["regression", "binary", "multi-label", "multi-class"] = "regression"
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
        perfs = [compute_performance(root, model_prefix + model, dataset, layer, algo, metric, aa=aa, n_classes=n_classes, task=task) for layer in range(LAYERS[model] + 1)]
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        ax.plot(
            np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])) if relative else np.arange(len(perfs)),
            perfs, 
            label=model_prefix + model, 
            c=MODEL_COLORS.get(model, None) if colored == True else colored,
            marker=MODEL_MARKERS.get(model, None) if colored == True else None,
        )

    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} of {algo.upper()} heads")
    ax.set_ylim(bottom=-0.05, top=1.05)
    if legend:
        ax.legend()

def comp_finetune_performance(base, dataset, metric):
    perfs = []
    for layer in FINETUNE_LAYERS:
        with open(base / "semifrozen_esm" / dataset / f"esm_t30" / f"unfreeze_{layer}" / "lr_1e-4" / f"predictions_unfrozen_esm_t30_{layer}_end_0.0001.pkl", "rb") as f:
            y_hat, y = pickle.load(f)[1]
        y = np.array(y).squeeze()
        y_hat = np.array(y_hat).squeeze()
        if DATASET2TASK[dataset] == "multi-label":
            y_hat = scipy.special.expit(y_hat)
        perfs.append(compute_metric(y_hat, y, metric, DATASET2TASK[dataset]))
    return perfs


def plot_dataset_finetune_comparison(ax, root, dataset, metric, n_classes, task):
    layer_perfs = [compute_performance(root, "esm_t30", dataset, layer, "lr", metric, aa=False, n_classes=n_classes, task=task) for layer in range(LAYERS["esm_t30"] + 1)]
    ax.plot(
        np.arange(len(layer_perfs)),
        layer_perfs, 
        label="layer-trained", 
        c=MODEL_COLORS.get("esm_t30", None),
        marker=MODEL_MARKERS.get("esm_t30", None),
    )

    ft_perfs = comp_finetune_performance(root, dataset, metric)
    ax.plot(
        FINETUNE_LAYERS,
        ft_perfs, 
        label="finetuned", 
        c="orange",
        marker=MODEL_MARKERS.get("esm_t30", None),
        linestyle="--",
    )
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_title(f"{dataset.capitalize()}")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.upper())


def read_metric(
        root: Path | None, 
        model: str | None, 
        dataset: str | None, 
        layer: int | None, 
        metric: Literal["ids", "density", "noverlap", "noverlap_50"], 
        filepath: Path | None = None, 
        aa: bool = False
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
    if dataset == "fluorescence_classification":
        dataset = "fluorescence"
    
    embed_dir = "aa_embeddings" if aa else "embeddings"

    if filepath is None:
        filepath = root / embed_dir / model / dataset / f"layer_{layer}" / (metric + ".csv")
    
    if not filepath.exists(): # type: ignore
        return 0
    
    df = pd.read_csv(filepath)
    return df[name_map[metric]].values[0]


def read_pca_metric(
        root: Path | None, 
        model: str | None, 
        dataset: str | None, 
        layer: int | None, 
        metric: Literal["zero", "pc@95", "var@10", "5dvol"], 
        filename: str = "pca.pkl",
        filepath: Path | None = None,
        k: int = 5,
        aa: bool = False, 
    ) -> float:
    """
    Read a specific metric from a CSV file for the given model, dataset, and layer.

    Args:
        root: Root directory containing the embeddings and results. [path parameter]
        model: Name of the model. [path parameter]
        dataset: Name of the dataset. [path parameter]
        layer: Layer number. [path parameter]
        metric: Metric to read ("zero", "pc@95", "var@10", "5dvol").
        filepath: Optional path to the CSV file. If None, constructs the path. This overwrites the other path parameters.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
    
    Returns:
        The value of the specified metric.
    """
    if filepath is None:
        embed_dir = "aa_embeddings" if aa else "embeddings"
        filepath = root / embed_dir / model / dataset / f"layer_{layer}" / filename
    if not filepath.exists():
        return 0
    
    with open(filepath, "rb") as f:
        exp_var = pickle.load(f)
    if any(np.isclose(exp_var, 0)):
        zero_index = next(i for i, v in enumerate(exp_var) if np.isclose(v, 0))
    else:
        zero_index = len(exp_var)
    
    if metric == "zero":
        return zero_index
    
    exp_var = exp_var[:zero_index]
    if metric == "pc@95":
        return next(i for i, v in enumerate(np.cumsum(exp_var) / sum(exp_var)) if v >= 0.95) / len(exp_var)
    elif metric == "var@10":
        return (np.cumsum(exp_var) / sum(exp_var))[10]
    elif metric == "5dvol":
        return (np.pi ** (k / 2)) / scipy.special.gamma((k / 2) + 1) * np.prod(exp_var[:k])
    else:
        raise ValueError(f"Unknown metric: {metric}")


def plot_metric(
        ax, 
        root: Path | None, 
        dataset: str | None, 
        relative: bool, 
        legend: bool = False, 
        metric: Literal["ids", "density", "noverlap", "noverlap_50", "zero", "pc@95", "var@10", "5dvol"] = "ids", 
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
        metric: Metric to plot ("ids", "density", "noverlap", "noverlap_50", "zero", "pc@95", "var@10", "5dvol").
        model_prefix: Prefix to add to model names. Either "" or "empty_" to indicate using normal or untrained models.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
    """
    title_map = {"ids": "Intrinsic Dimensions", "density": "Density", "noverlap": "Neighbor Overlap", "noverlap_50": "Neighbor Overlap (50)"}
    for model in MODELS[:-1]:
        # if metric == "5dvol" and model.startswith("ankh"):
        #     continue  # ankh is crazy in this metric
        perfs = []
        for layer in range(LAYERS[model] + 1):
            if metric.startswith("noverlap") and layer == LAYERS[model]:
                continue
            if metric in {"zero", "pc@95", "var@10", "5dvol"}:
                result = read_pca_metric(root, model_prefix + model, dataset, layer, metric, aa=aa)
            else:
                result = read_metric(root, model_prefix + model, dataset, layer, metric, aa=aa)
            perfs.append(result)
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        if relative:
            x_ticks = np.arange(0, 1 + 1e-5, 1 / (LAYERS[model]))
            if metric.startswith("noverlap"):
                x_ticks = x_ticks[:-1]
                x_ticks += 1 / (2 * LAYERS[model])
            ax.plot(x_ticks, perfs, label=model_prefix + model, c=MODEL_COLORS.get(model, None), marker=MODEL_MARKERS.get(model, None))
        else:
            ax.plot(perfs, label=model_prefix + model, c=MODEL_COLORS.get(model, None), marker=MODEL_MARKERS.get(model, None))

    ax.set_xlabel(("Relative" if relative else "Absolute") + " Layer")
    ax.set_ylabel(title_map.get(metric, metric))
    ax.set_title(f"{title_map.get(metric, metric)}")
    if metric not in {"ids", "density", "5dvol"}:
        ax.set_ylim(bottom=-0.05, top=1.05)
    if metric == "5dvol":
        ax.set_yscale("log")
    if legend:
        ax.legend(loc="upper right")


if __name__ == "__main__":
    print("Hello")
    compute_performance(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB", "ankh_base", "deeploc2", 10, "lr", "mcc", aa=False, n_classes=10, task="multi-label")
