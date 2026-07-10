from pathlib import Path
import pickle
from typing import Literal

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef, roc_auc_score
import torch

from src.viz.constants import SPLIT_ID


def multioutput_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the average Matthews Correlation Coefficient (MCC) for a multi-output task.

    Args:
        y_true: np.ndarray of shape (n_samples, n_outputs)
        y_pred: np.ndarray of shape (n_samples, n_outputs)

    Returns:
        float: average MCC across outputs
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    
    mccs = []
    for i in range(y_true.shape[1]):
        try:
            if sum(y_true[:, i]) == 0:
                mccs.append(0.0)
            else:
                mccs.append(matthews_corrcoef(y_true[:, i], y_pred[:, i] > 0.5))
        except ValueError:
            # Handle cases where MCC is undefined (e.g., only one class present)
            mccs.append(0.0)
    
    return float(np.mean(mccs))


def compute_metric(y_hat: np.ndarray, y: np.ndarray, metric: str, task: Literal["regression", "binary", "multi-label", "multi-class"]) -> float:
    """
    Compute a performance metric between predictions and true labels.

    Args:
        y_hat: Predicted values.
        y: True values.
        metric: Performance metric to compute (e.g., "pearson", "mcc").
        task: Type of task ("regression", "binary", "multi-label", "multi-class").
    
    Returns:
        The computed performance metric.
    """
    match metric.lower():
        case "pearson":
            return np.corrcoef(y_hat, y)[0, 1]
        case "spearman":
            return spearmanr(y_hat, y)[0] # type: ignore
        case "r2":
            return r2_score(y, y_hat)
        case "mse":
            return mean_squared_error(y, y_hat)
        case "mae":
            return mean_absolute_error(y, y_hat)
        case "rmse":
            return np.sqrt(mean_squared_error(y, y_hat))
        case "acc":
            if task == "multi-label":
                return accuracy_score(y, y_hat > 0.5) # type: ignore
            return accuracy_score(y, y_hat) # type: ignore
        case "mlm":
            # return torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits.view(-1, logits.size(-1)), y.view(-1))
            if y_hat.ndim == 3:
                y_hat = y_hat[0]
            return torch.nn.CrossEntropyLoss(label_smoothing=0.1)(torch.tensor(y_hat), torch.tensor(y)).item()
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
                    return roc_auc_score(y.astype(int), np.array(y_hat)[:, :, 1].T, multi_class="ovr") # type: ignore
                return roc_auc_score(y.astype(int), y_hat) # type: ignore
            if task == "binary":
                if y_hat.ndim == 2:
                    return roc_auc_score(y.astype(int), y_hat[:, 1]) # type: ignore
                return roc_auc_score(y.astype(int), y_hat) # type: ignore
            return roc_auc_score(y.astype(int), y_hat, multi_class="ovr") # type: ignore
        case _:
            raise ValueError(f"Unknown metric: {metric}")


def compute_performance(
        root: Path | None = None, 
        model: str | None = None, 
        dataset: str | None = None, 
        layer: int | None = None, 
        filepath: Path | None = None, 
        algo: str | None = None, 
        metric: str | None = None, 
        task: Literal["regression", "binary", "multi-label", "multi-class"] = "regression",
        aa: bool = False, 
        n_classes: int = 42,
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
    try:
        embed_dir, filename = "embeddings", f"predictions_{algo}.pkl"
        if aa:
            embed_dir = "aa_embeddings"
            if dataset == "scope_40_208":
                filename = f"predictions_{algo}_{n_classes}.pkl"
        if filepath is None:
            filepath = root / embed_dir / model / dataset / f"layer_{layer}" / filename
        
        if not filepath.exists():
            return np.nan
        
        with open(filepath, "rb") as f:
            y_hat, y = pd.read_pickle(f)[SPLIT_ID]
            y_hat = np.array(y_hat)
            y = np.array(y)
        
        return compute_metric(y_hat, y, metric, task)
    except Exception as e:
        print(f"Error computing performance for {model} layer {layer} on {dataset} with {algo}: {e}")
        return np.nan


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
    fpath = root / "embeddings" / model / dataset / f"layer_{layer}" / f"predictions_{algo}_{level}_{k if k is not None else f'min{min_x}'}.pkl"
    if not fpath.exists():
        return 0
    with open(fpath, "rb") as f:
        try:
            y_hat, y = pd.read_pickle(f)[1]
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            return 0
    return compute_metric(np.array(y_hat), np.array(y), metric, task="multi-class")


def read_metric(
        root: Path | None, 
        model: str | None, 
        dataset: str | None, 
        layer: int | None, 
        filepath: Path | None = None, 
        metric: Literal["ids", "density", "noverlap", "noverlap_50"] | None = None, 
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


def read_scope_metric(root: Path, model, layer, metric, filename):
    name_map = {"ids": "twonn_id", "density": "density", "noverlap": "neighbor_overlap", "noverlap_50": "neighbor_overlap"}
    filepath = root / "embeddings" / model / "scope_40_208" / f"layer_{layer}" / filename
    if not filepath.exists():
        return 0
    df = pd.read_csv(filepath)
    return df[name_map[metric]].values[0]