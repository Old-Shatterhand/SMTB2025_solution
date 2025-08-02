import random
import pickle
import argparse
from time import time
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
import torch
from sklearn.metrics import matthews_corrcoef, mean_squared_error, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import spearmanr


def build_dataloader(df: pd.DataFrame, embed_path: Path):
    """
    Build a DataLoader for the given DataFrame and embedding path.

    :param df: DataFrame containing the data.
    :param embed_path: Path to the directory containing the embeddings.
    :param dataloader_kwargs: Additional arguments for DataLoader.

    :return: DataLoader for the embeddings and targets.
    """
    embed_path = Path(embed_path)
    embeddings = []
    for idx in df["ID"].values:
        with open(embed_path / f"{idx}.pkl", "rb") as f:
            tmp = pickle.load(f)
            if not isinstance(tmp, np.ndarray):
                tmp = tmp.cpu().numpy()
            embeddings.append(tmp)
    inputs = np.stack(embeddings)
    targets = np.array(df['label'].values).astype(np.float32)
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


start = time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE + " is available")

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--function', type=str, required=True, choices=["lr", "xgb"])
parser.add_argument('--task', type=str, default="regression", choices=["regression", "classification"])
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument("--binary", action='store_true', default=False, help="Indicator for binary classification")
args = parser.parse_args()

layer = int(args.embed_path.name.split("_")[-1])
dataset = args.data_path.stem
model = args.embed_path.parent.parent.name
result_folder = args.embed_path.parent / dataset

if (result_folder / f"metrics_{args.function}_{args.seed}.csv").exists():
    exit(0)

df = pd.read_csv(args.data_path)

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("Loading embeddings from", args.embed_path)
train_X, train_Y = build_dataloader(df[df["split"] == "train"], args.embed_path)
valid_X, valid_Y = build_dataloader(df[df["split"] == "valid"], args.embed_path)
test_X, test_Y = build_dataloader(df[df["split"] == "test"], args.embed_path)

print("Training", args.function, "model on", dataset)
if args.task == "regression":
    if args.function == "lr":
        model = LinearRegression().fit(train_X, train_Y)
    elif args.function == "xgb":
        model = XGBRegressor(
            tree_method="hist",
            n_estimators=50,
            max_depth=20,
            random_state=42,
            device="cpu"
        ).fit(train_X, train_Y)
else:
    if args.function == "lr":
        if args.binary:
            model = LogisticRegression().fit(train_X, train_Y)
        else:
            model = MultiOutputClassifier(LogisticRegression()).fit(train_X, train_Y)

print("Evaluating model")
train_prediction = model.predict(train_X)
valid_prediction = model.predict(valid_X)
test_prediction = model.predict(test_X)

if args.task == "regression":
    train_m1 = spearmanr(train_prediction, train_Y)[0]
    valid_m1 = spearmanr(valid_prediction, valid_Y)[0]
    test_m1 = spearmanr(test_prediction, test_Y)[0]

    train_m2 = mean_squared_error(train_Y, train_prediction)
    valid_m2 = mean_squared_error(valid_Y, valid_prediction)
    test_m2 = mean_squared_error(test_Y, test_prediction)
else:
    if args.binary:
        train_m1 = matthews_corrcoef(train_Y, train_prediction)
        valid_m1 = matthews_corrcoef(valid_Y, valid_prediction)
        test_m1 = matthews_corrcoef(test_Y, test_prediction)

        train_m2 = roc_auc_score(train_Y, train_prediction)
        valid_m2 = roc_auc_score(valid_Y, valid_prediction)
        test_m2 = roc_auc_score(test_Y, test_prediction)
    else:
        train_m1 = multioutput_mcc(train_Y, train_prediction)
        valid_m1 = multioutput_mcc(valid_Y, valid_prediction)
        test_m1 = multioutput_mcc(test_Y, test_prediction)

        train_m2 = roc_auc_score(train_Y, train_prediction, average='weighted', multi_class='ovr')
        valid_m2 = roc_auc_score(valid_Y, valid_prediction, average='weighted', multi_class='ovr')
        test_m2 = roc_auc_score(test_Y, test_prediction, average='weighted', multi_class='ovr')

pd.DataFrame({
    "train_spearman": [train_m1],
    "valid_spearman": [valid_m1],
    "test_spearman": [test_m1],
    "train_mse": [train_m2],
    "valid_mse": [valid_m2],
    "test_mse": [test_m2],
}).to_csv(result_folder / f"metrics_{args.function}_{args.seed}.csv", index=False)

with open(result_folder / f"predictions_{args.function}_{args.seed}.pkl", "wb") as f:
    pickle.dump(((train_prediction, train_Y), (valid_prediction, valid_Y), (test_prediction, test_Y)), f)

if not (res := (args.data_path.parent / "results.csv")).exists():
    with open(res, "w") as f:
        f.write("Embedding Model,Downstream Model,#layers,Dataset,Seed,Spearman,MSE\n")

with open(res, "a") as f:
    f.write(f"{model},{args.function},{layer},{dataset},{args.seed},{test_m1},{test_m2}\n")

print(f"Script finished in {time() - start:.2f} seconds")
