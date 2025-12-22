from pathlib import Path
import argparse
import pickle
import random
from time import time
import numpy as np
import pandas as pd
import torch
from dadapy import data

from src.downstream.utils import fit_model


MAP = {
    2: ["X", "M"],
    3: ["H", "E", "C"],
    8: ["G", "H", "I", "B", "E", "T", "S", "-"],
}

def build_aa_dataloader(df, embed_path, n_classes: int, shuffle: bool = True):
    if len(df) == 0:
        return np.array([]), np.array([])
    CLASS_MAPPING = {c: n for n, c in enumerate(MAP[n_classes])}
    if n_classes == 2:
        labels = "labels"
    elif n_classes == 3:
        labels = "secstr_3c"
    elif n_classes == 8:
        labels = "secstr_8c"
    
    embeddings = []
    aa_labels = []
    
    for _, row in df.iterrows():
        try:
            tmp_labels = row[labels]
            if not hasattr(tmp_labels, "__len__") or len(tmp_labels) != len(row["sequence"]):
                continue
            with open(embed_path / f"{row['ID']}.pkl", "rb") as f:
                tmp = pickle.load(f)
            embeddings.append(tmp)
            aa_labels += [CLASS_MAPPING[c] for c in tmp_labels]
        except Exception:
            pass
    embeddings = np.concatenate(embeddings, axis=0)
    if shuffle:
        permut = np.random.permutation(embeddings.shape[0])
        embeddings = embeddings[permut]
        aa_labels = np.array(aa_labels)[permut]
    return embeddings, aa_labels


parser = argparse.ArgumentParser(description="Train on amino-acid embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--function', type=str, required=True, choices=["knn", "lr"])
parser.add_argument('--n-classes', type=int, required=True, help='Number of classes to consider')  # all tasks are either binary or multi-class classification
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--force', action='store_true', help='Force recomputation even if results exist')
args = parser.parse_args()

layer = int(args.embed_path.name.split("_")[-1])
dataset = args.data_path.stem
model_name = args.embed_path.parent.parent.name
result_folder = args.embed_path.parent.parent / dataset / f"layer_{layer}"
result_file = result_folder / f"predictions_{args.function}_{args.seed}.pkl"


if not args.force and result_file.exists():
    print(f"Results already exist.")
    exit(0)
result_folder.mkdir(parents=True, exist_ok=True)

start = time()

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("Loading embeddings from", args.embed_path)

df = pd.read_csv(args.data_path)
train_X, train_Y = build_aa_dataloader(df[df["split"] == "train"], args.embed_path, args.n_classes, shuffle=True)
valid_X, valid_Y = build_aa_dataloader(df[df["split"] == "valid"], args.embed_path, args.n_classes, shuffle=False)
test_X, test_Y = build_aa_dataloader(df[df["split"] == "test"], args.embed_path, args.n_classes, shuffle=False)

print("Fitting", args.function, "model on", dataset)
model = fit_model("classification", args.function, train_X, train_Y, binary=(args.n_classes == 2)) # type: ignore

print("Evaluating model")
train_prediction = model.predict_proba(train_X)
valid_prediction = model.predict_proba(valid_X)
test_prediction = model.predict_proba(test_X)
if args.n_classes == 2:
    train_prediction = train_prediction[:, 1]
    valid_prediction = valid_prediction[:, 1]
    test_prediction = test_prediction[:, 1]

with open(result_file, "wb") as f:
    pickle.dump(((train_prediction, train_Y), (valid_prediction, valid_Y), (test_prediction, test_Y)), f)
