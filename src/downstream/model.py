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
from xgboost import XGBRegressor
from scipy.stats import spearmanr

from src.downstream.utils import build_dataloader, fit_model, multioutput_mcc

start = time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE + " is available")

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--function', type=str, required=True, choices=["lr", "xgb", "knn"])
parser.add_argument('--task', type=str, default="regression", choices=["regression", "classification"])
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument("--binary", action='store_true', default=False, help="Indicator for binary classification")
args = parser.parse_args()

layer = int(args.embed_path.name.split("_")[-1])
dataset = args.data_path.stem
model = args.embed_path.parent.parent.name
result_folder = args.embed_path.parent.parent / dataset / f"layer_{layer}"

if (result_folder / f"metrics_{args.function}_{args.seed}.csv").exists():
    exit(0)
if not result_folder.exists():
    result_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.data_path)

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("Loading embeddings from", args.embed_path)
labels = "label"
if dataset == "deeploc2":
    labels = ["Membrane", "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]
train_X, train_Y = build_dataloader(df[df["split"] == "train"], args.embed_path, labels)
valid_X, valid_Y = build_dataloader(df[df["split"] == "valid"], args.embed_path, labels)
test_X, test_Y = build_dataloader(df[df["split"] == "test"], args.embed_path, labels)

print("Fitting", args.function, "model on", dataset)
model = fit_model(args.task, args.function, train_X, train_Y, binary=args.binary)

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
        train_m1 = matthews_corrcoef(train_Y.astype(int), train_prediction.astype(int))
        valid_m1 = matthews_corrcoef(valid_Y.astype(int), valid_prediction.astype(int))
        test_m1 = matthews_corrcoef(test_Y.astype(int), test_prediction.astype(int))

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
