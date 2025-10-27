import random
import pickle
import argparse
from time import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from src.downstream.utils import build_dataloader, fit_model, twonn_dimension


parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--function', type=str, required=True, choices=["lr", "xgb", "knn", "2nn", "lpca"])
parser.add_argument('--task', type=str, default="regression", choices=["regression", "classification"])
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument("--binary", action='store_true', default=False, help="Indicator for binary classification")
args = parser.parse_args()

MODE_ID = args.function.upper() in {"2NN", "LPCA"}

layer = int(args.embed_path.name.split("_")[-1])
dataset = args.data_path.stem
model_name = args.embed_path.parent.parent.name
result_folder = args.embed_path.parent.parent / dataset / f"layer_{layer}"
if not MODE_ID:
    result_file = result_folder / f"predictions_{args.function}_{args.seed}.pkl"
else:
    result_file = result_folder / f"ids.csv"

if result_file.exists():
    print("result already exists.")
    exit(0)
result_folder.mkdir(parents=True, exist_ok=True)

start = time()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE + " is available")

df = pd.read_csv(args.data_path)

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("Loading embeddings from", args.embed_path)
labels = "label"
if dataset == "deeploc2":
    labels = ["Membrane", "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

if "train" in df.columns:
    train_X, train_Y = build_dataloader(df[df["split"] == "train"], args.embed_path, labels)
else:
    train_X, train_Y = build_dataloader(df, args.embed_path, labels)

if not MODE_ID:
    valid_X, valid_Y = build_dataloader(df[df["split"] == "valid"], args.embed_path, labels)
    test_X, test_Y = build_dataloader(df[df["split"] == "test"], args.embed_path, labels)

    print("Fitting", args.function, "model on", dataset)
    model = fit_model(args.task, args.function, train_X, train_Y, binary=args.binary)

    print("Evaluating model")
    train_prediction = model.predict(train_X)
    valid_prediction = model.predict(valid_X)
    test_prediction = model.predict(test_X)

    with open(result_file, "wb") as f:
        pickle.dump(((train_prediction, train_Y), (valid_prediction, valid_Y), (test_prediction, test_Y)), f)
else:
    twonn_id = twonn_dimension(train_X)
    
    # print("Training lPCA")
    # lpca_estimator = lPCA()
    # lpca_estimator.fit_pw(train_X)
    # lpca_id = lpca_estimator.dimension_

    pd.DataFrame({
        "twonn_id": [twonn_id],
        # "lpca_id": [lpca_id]
    }).to_csv(result_folder / f"ids.csv", index=False)

print(f"Script finished in {time() - start:.2f} seconds")