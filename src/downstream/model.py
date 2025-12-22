import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import pickle
import argparse
from time import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from dadapy import data

from src.downstream.utils import build_dataloader, fit_model


parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--function', type=str, required=True, choices=["lr", "knn", "2nn", "lpca", "no", "dense"])
parser.add_argument('--no-num', type=int, default=10, help='Number of neighbors for noverlap computation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--task', type=str, default="regression", choices=["regression", "classification"])
parser.add_argument("--binary", action='store_true', default=False, help="Indicator for binary classification")
parser.add_argument("--force", action='store_true', default=False, help="Force recomputation even if results exist")
args = parser.parse_args()

layer = int(args.embed_path.name.split("_")[-1])
dataset = args.data_path.stem
model_name = args.embed_path.parent.parent.name
result_folder = args.embed_path.parent.parent / dataset / f"layer_{layer}"

# Determine result file path based on the function to be computed
match args.function:
    case "lr" | "xgb" | "knn":
        result_file = result_folder / f"predictions_{args.function}_{args.seed}.pkl"
    case "2nn" | "lpca":
        result_file = result_folder / f"ids.csv"
    case "no":
        embed_path_next = args.embed_path.parent / f"layer_{layer + 1}"
        if not embed_path_next.exists():
            exit(0)
        result_file = result_folder / f"noverlap_{args.no_num}.csv"
    case "dense":
        result_file = result_folder / f"density.csv"
    case _:
        raise ValueError(f"Unknown function: {args.function}")

# Check if result file already exists and skip eventually
if not args.force and result_file.exists():
    print("result already exists.")
    exit(0)
result_folder.mkdir(parents=True, exist_ok=True)

start = time()

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("Loading embeddings from", args.embed_path)
if args.function not in {"lr", "knn", "xgb"}:
    labels = None
else:
    labels = "label"
    if dataset == "deeploc2":
        labels = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

# Load the data
df = pd.read_csv(args.data_path)
train_X, train_Y = build_dataloader(df[df["split"] == "train"], args.embed_path, labels, shuffle=args.function != "no")

match args.function:
    case "lr" | "knn":
        valid_X, valid_Y = build_dataloader(df[df["split"] == "valid"], args.embed_path, labels)
        test_X, test_Y = build_dataloader(df[df["split"] == "test"], args.embed_path, labels)

        print("Fitting", args.function, "model on", dataset)
        model = fit_model(args.task, args.function, train_X, train_Y, binary=args.binary) # type: ignore

        print("Evaluating model")
        if args.task == "classification":
            train_prediction = model.predict_proba(train_X)
            valid_prediction = model.predict_proba(valid_X)
            test_prediction = model.predict_proba(test_X)
            if args.binary:
                train_prediction = train_prediction[:, 1]
                valid_prediction = valid_prediction[:, 1]
                test_prediction = test_prediction[:, 1]
        else:
            train_prediction = model.predict(train_X)
            valid_prediction = model.predict(valid_X)
            test_prediction = model.predict(test_X)

        with open(result_file, "wb") as f:
            pickle.dump(((train_prediction, train_Y), (valid_prediction, valid_Y), (test_prediction, test_Y)), f)
    case "2nn" | "lpca":
        twonn_id = data.Data(train_X).compute_id_2NN()[0]

        pd.DataFrame({
            "twonn_id": [twonn_id],
        }).to_csv(result_file, index=False)
    case "no":
        train_X_next, _ = build_dataloader(df[df["split"] == "train"], embed_path_next, labels, shuffle=False)
        noverlap = data.Data(train_X).return_data_overlap(train_X_next, k=args.no_num)
        
        pd.DataFrame({
            "neighbor_overlap": [noverlap],
        }).to_csv(result_file, index=False)
    case "dense":
        density = data.Data(train_X).compute_density_kNN(k=10)[0]
        mean_density = np.log(np.mean(np.exp(density)))

        pd.DataFrame({
            "density": [mean_density]
        }).to_csv(result_file, index=False)
    case _:
        raise ValueError(f"Unknown function: {args.function}")

print("Results saved to", result_file)
print(f"Script finished in {time() - start:.2f} seconds")
