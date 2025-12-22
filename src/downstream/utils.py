from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import matthews_corrcoef
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBRegressor


def build_dataloader(df: pd.DataFrame, embed_path: Path, labels: str | list[str] | None = "label", shuffle: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build a DataLoader for the given DataFrame and embedding path.

    Args:
        df: DataFrame containing the data.
        embed_path: Path to the directory containing the embeddings.
        labels: Column name(s) for the target labels.

    Returns:
        Tuple of (inputs, targets) where inputs are the embeddings and targets are the labels.
    """
    embed_path = Path(embed_path)
    embeddings = []
    valid_ids = set()
    for idx in df["ID"].values:
        try:
            with open(embed_path / f"{idx}.pkl", "rb") as f:
                tmp = pickle.load(f)
            if not isinstance(tmp, np.ndarray):
                tmp = tmp.cpu().numpy()
            embeddings.append(tmp)
            valid_ids.add(idx)
        except Exception:
            pass
    
    inputs = np.stack(embeddings)
    if shuffle:
        permut = np.random.permutation(inputs.shape[0])
        inputs = inputs[permut]

    if labels is not None:
        targets = np.array(df[df["ID"].isin(valid_ids)][labels].values).astype(np.float32)
        if shuffle:
            targets = targets[permut]
        return inputs, targets
    else:
        return inputs, None


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
    
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    
    mccs = []
    for i in range(y_true.shape[1]):
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i] > 0.5)
        except ValueError:
            # Handle cases where MCC is undefined (e.g., only one class present)
            mcc = 0.0
        mccs.append(mcc)
    
    return float(np.mean(mccs))


def fit_model(task: str, algo: str, trainX: np.ndarray, trainY: np.ndarray, binary: bool = False) -> sklearn.base.BaseEstimator:
    """
    Fit a machine learning model based on the specified task and algorithm.

    Args:
        task: "regression" or "classification"
        algo: Algorithm to use ("lr", "knn")
        trainX: Training features
        trainY: Training labels
        binary: Indicator for binary classification (only relevant if task is "classification")
    """
    if task == "regression":
        if algo == "lr":
            return LinearRegression(n_jobs=1).fit(trainX, trainY)
        elif algo == "knn":
            return KNeighborsRegressor(n_neighbors=5, weights="distance", algorithm="brute", metric="cosine", n_jobs=1).fit(trainX, trainY)
    else:
        if algo == "lr":
            if binary:
                return LogisticRegression(n_jobs=1).fit(trainX, trainY)
            else:
                return MultiOutputClassifier(LogisticRegression(n_jobs=1), n_jobs=1).fit(trainX, trainY)
        elif algo == "knn":
            return KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute", metric="cosine", n_jobs=1).fit(trainX, trainY)
    raise ValueError(f"Unknown task: {task} or algorithm: {algo}")
