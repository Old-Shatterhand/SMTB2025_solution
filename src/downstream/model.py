import random
import pickle
import argparse
from time import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
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
            embeddings.append(pickle.load(f))
    inputs = np.stack(embeddings)
    targets = np.array(df['label'].values).astype(np.float32)
    return inputs, targets


start = time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device + " is available")

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--function', type=str, required=True, choices=["lr", "xgb"])
parser.add_argument('--task', type=str, default="regression", choices=["regression", "classification"])
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

layer = int(args.embed_path.name.split("_")[-1])
dataset = args.embed_path.parent.name
model = args.embed_path.parent.parent.name

if (args.embed_path / f"metrics_{args.function}_{args.seed}.csv").exists():
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
if args.function == "lr":
    reg = LinearRegression().fit(train_X, train_Y)
elif args.function == "xgb":
    reg = XGBRegressor(
        tree_method="gpu_hist" if device == "cuda" else "hist",
        n_estimators=100, 
        max_depth=20, 
        random_state=42, 
        device=device
    ).fit(train_X, train_Y)

print("Evaluating model")
train_prediction = reg.predict(train_X)
test_prediction = reg.predict(test_X)
valid_prediction = reg.predict(valid_X)

train_spearman = spearmanr(train_prediction, train_Y)[0]
valid_spearman = spearmanr(valid_prediction, valid_Y)[0]
test_spearman = spearmanr(test_prediction, test_Y)[0]

train_mse = mean_squared_error(train_prediction, train_Y)
valid_mse = mean_squared_error(valid_prediction, valid_Y)
test_mse = mean_squared_error(test_prediction, test_Y)

pd.DataFrame({
    "train_spearman": [train_spearman],
    "valid_spearman": [valid_spearman],
    "test_spearman": [test_spearman],
    "train_mse": [train_mse],
    "valid_mse": [valid_mse],
    "test_mse": [test_mse],
}).to_csv(args.embed_path / f"metrics_{args.function}_{args.seed}.csv", index=False)

with open(args.embed_path / f"predictions_{args.function}_{args.seed}.pkl", "wb") as f:
    pickle.dump(((train_prediction, train_Y), (valid_prediction, valid_Y), (test_prediction, test_Y)), f)

if not (res := (args.data_path.parent / "results.csv")).exists():
    with open(res, "w") as f:
        f.write("Embedding Model,Downstream Model,#layers,Dataset,Seed,Spearman,MSE\n")

with open(res, "a") as f:
    f.write(f"{model},{args.function},{layer},{dataset},{args.seed},{test_spearman},{test_mse}\n")

print(f"Script finished in {time() - start:.2f} seconds")
