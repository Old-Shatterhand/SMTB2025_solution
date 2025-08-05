from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import matthews_corrcoef
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def build_dataloader(df: pd.DataFrame, embed_path: Path, labels: str | list[str] = "label"):
    """
    Build a DataLoader for the given DataFrame and embedding path.

    :param df: DataFrame containing the data.
    :param embed_path: Path to the directory containing the embeddings.
    :param dataloader_kwargs: Additional arguments for DataLoader.

    :return: DataLoader for the embeddings and targets.
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
    targets = np.array(df[df["ID"].isin(valid_ids)][labels].values).astype(np.float32)

    # Shuffle the inputs and targets
    permut = np.random.permutation(inputs.shape[0])
    inputs = inputs[permut]
    targets = targets[permut]
    
    return inputs, targets


def multioutput_mcc(y_true, y_pred):
    """
    Compute the average Matthews Correlation Coefficient (MCC) for a multi-output task.

    Parameters:
    - y_true: np.ndarray of shape (n_samples, n_outputs)
    - y_pred: np.ndarray of shape (n_samples, n_outputs)

    Returns:
    - float: average MCC across outputs
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    
    mccs = []
    for i in range(y_true.shape[1]):
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
        except ValueError:
            # Handle cases where MCC is undefined (e.g., only one class present)
            mcc = 0.0
        mccs.append(mcc)
    
    return np.mean(mccs)


def fit_model(task, algo, trainX, trainY, binary: bool = False) -> sklearn.base.BaseEstimator:
    if task == "regression":
        if algo == "lr":
            return LinearRegression().fit(trainX, trainY)
        elif algo == "xgb":
            return XGBRegressor(
                tree_method="hist",
                n_estimators=50,
                max_depth=20,
                random_state=42,
                device="cpu",
            ).fit(trainX, trainY)
        elif algo == "knn":
            return KNeighborsRegressor(n_neighbors=5, weights="distance", algorithm="brute", metric="cosine").fit(trainX, trainY)
    else:
        if algo == "lr":
            if binary:
                return LogisticRegression().fit(trainX, trainY)
            else:
                return MultiOutputClassifier(LogisticRegression()).fit(trainX, trainY)
        elif algo == "knn":
            return KNeighborsRegressor(n_neighbors=5, weights="distance", algorithm="brute", metric="cosine").fit(trainX, trainY)
    raise ValueError(f"Unknown task: {task} or algorithm: {algo}")
