import os
from pathlib import Path
import argparse
import pickle
import random
from time import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import torch
from dadapy import data

from src.downstream.utils import build_dataloader


parser = argparse.ArgumentParser(description="Embeddings.\nEither --top-k or --min-x must be specified.")
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--top-k', type=int, default=None, help='Number of top labels to consider')
parser.add_argument("--min-x", type=int, default=None, help="Minimum number of samples for a label to be included")
parser.add_argument('--level', required=True, choices=["class", "fold", "family", "superfamily"])
parser.add_argument('--function', type=str, required=True, choices=["knn", "lr", "2nn", "no"])
parser.add_argument('--no-num', type=int, default=10, help='Number of neighbors for noverlap computation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--force', action='store_true', help='Force recomputation even if results exist')
args = parser.parse_args()

start = time()

layer = int(args.embed_path.name.split("_")[-1])
model_name = args.embed_path.parent.parent.name
result_folder = args.embed_path.parent.parent / "scope_40_208" / f"layer_{layer}"
if args.top_k is None and args.min_x is None:
    raise ValueError("Either --top-k or --min-x must be specified.")
suffix = args.level + "_" + (args.top_k if args.top_k is not None else f"min{args.min_x}")

# Determine result file path based on the function to be computed
match args.function:
    case "lr" | "knn":
        result_file = result_folder / f"predictions_{args.function}_{suffix}.pkl"
    case "2nn":
        result_file = result_folder / f"ids_{suffix}.csv"
    case "no":
        embed_path_next = args.embed_path.parent / f"layer_{layer + 1}"
        if not embed_path_next.exists():
            exit(0)
        result_file = result_folder / f"noverlap_{suffix}_{args.no_num}.csv"
    case _:
        raise ValueError(f"Unknown function: {args.function}")

if not args.force and result_file.exists():
    print(f"Results already exist.")
    exit(0)
result_folder.mkdir(parents=True, exist_ok=True)

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load SCOPe data and reduce to the top-k labels
df = pd.read_csv(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB" / "datasets" / "scope_40_208.csv")
if args.top_k is not None:
    top_k_labels = set(x[0] for x in list(sorted(dict(df[args.level].value_counts()).items(), key=lambda x: x[1], reverse=True)[:args.top_k]))
    df = df[df[args.level].isin(top_k_labels)].reset_index(drop=True)
if args.min_x is not None:
    label_counts = df[args.level].value_counts()
    valid_labels = label_counts[label_counts >= args.min_x].index
    df = df[df[args.level].isin(valid_labels)].reset_index(drop=True)
train_X, train_Y = build_dataloader(df[df["split"] == "train"], args.embed_path, args.level, shuffle=args.function != "no")

match args.function:
    case "lr" | "knn":
        val_X, val_Y = build_dataloader(df[df["split"] == "val"], args.embed_path, args.level)
        test_X, test_Y = build_dataloader(df[df["split"] == "test"], args.embed_path, args.level)

        # Train the model and make predictions
        print("Fitting", args.function, "model on SCOPe40 2.08")
        if args.function == "lr":
            model = LogisticRegression(random_state=args.seed)
        elif args.function == "knn":
            model = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute", metric="cosine")
        model.fit(train_X, train_Y)

        print("Evaluating model")
        train_prediction = model.predict_proba(train_X)
        val_prediction = model.predict_proba(val_X)
        test_prediction = model.predict_proba(test_X)
        with open(result_file, "wb") as f:
            pickle.dump(((train_prediction, train_Y), (val_prediction, val_Y), (test_prediction, test_Y)), f)

    case "2nn" | "lpca":
        twonn_id = data.Data(train_X).compute_id_2NN()[0]

        pd.DataFrame({
            "twonn_id": [twonn_id],
        }).to_csv(result_file, index=False)
    case "no":
        train_X_next, _ = build_dataloader(df[df["split"] == "train"], embed_path_next, args.level, shuffle=False)
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
