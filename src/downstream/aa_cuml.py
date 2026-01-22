import os
os.environ['CUPY_NVCC_GENERATE_CODE'] = 'current'

import sys
import copy
import random
import pickle
import argparse
from time import time
from pathlib import Path
from typing import Literal
from datetime import datetime

import cuml
from sklearn.multioutput import MultiOutputClassifier
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from cuml.manifold import UMAP as cuUMAP
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml.random_projection import GaussianRandomProjection as cuGRP
from sklearn.neighbors import KNeighborsClassifier as skKNN

from src.downstream.dada import compute_id_2NN, return_data_overlap
from src.unfrozen_esm import ProteinDataset, predict, train_loop


N_ROWS = 1000
N_DIMS = 75

MAP = {
    2: ["X", "M"],
    3: ["H", "E", "C"],
    8: ["G", "H", "I", "B", "E", "T", "S", "-"],
}


def fit_model(task: str, algo: str, trainX: np.ndarray, trainY: np.ndarray, binary: bool = False):
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
            return cuml.LinearRegression(output_type="numpy").fit(trainX, trainY)
        elif algo == "knn":
            return cuKNN(n_neighbors=5, weights="distance", algorithm="brute", metric="cosine", n_jobs=1).fit(trainX, trainY)
    else:
        if algo == "lr":
            if binary:
                return cuml.LogisticRegression(output_type="numpy").fit(trainX, trainY)
            else:
                return MultiOutputClassifier(cuml.LogisticRegression(output_type="numpy"), n_jobs=1).fit(trainX, trainY)
        elif algo == "knn":
            return cuKNN(n_neighbors=5, weights="distance", algorithm="brute", metric="cosine", n_jobs=1).fit(trainX, trainY)
    raise ValueError(f"Unknown task: {task} or algorithm: {algo}")


class LRHead(torch.nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.head = torch.nn.Linear(embed_dim, n_classes if n_classes > 2 else 1)
    
    def forward(self, x):
        out = self.head(x)
        return torch.sigmoid(out) if self.n_classes == 2 else torch.softmax(out, dim=1)


def train_lr_head(out_folder, train_X, train_y, valid_X, valid_y, test_X, test_y, binary: bool = False):
    # loss_file = out_folder / f"loss_unfrozen_lr_42.csv"
    result_file = out_folder / f"predictions_lr_42.pkl"

    # ESM-t6 818s für 2000 samples mit Backprop, 162s mit sklearn LR, 80s mit cuML LR
    # Ankh L 152s fuer 200 samples mit cuML LR, 3862s for 2000, 1023s for 1000
    #n_classes = 3
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train_dataset = ProteinDataset(torch.tensor(train_X, device=DEVICE), torch.tensor(train_y, device=DEVICE))
    #train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    #valid_dataset = ProteinDataset(torch.tensor(valid_X, device=DEVICE), torch.tensor(valid_y, device=DEVICE))
    #valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    #test_dataset = ProteinDataset(torch.tensor(test_X, device=DEVICE), torch.tensor(test_y, device=DEVICE))
    #test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #model = LRHead(embed_dim=320, n_classes=n_classes).to(DEVICE)
    #optimizer = torch.optim.AdamW(model.parameters())
    #loss_fn = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()
    #torch.manual_seed(42)

    #best_model_state, losses = train_loop(model, train_dataloader, valid_dataloader, loss_fn, optimizer, epochs=100, device=DEVICE)
    #model.load_state_dict(best_model_state)
    #train_predictions, train_labels = predict(model, train_dataloader)
    #valid_predictions, valid_labels = predict(model, valid_dataloader)
    #test_predictions, test_labels = predict(model, test_dataloader)
    if binary:
        model = cuml.LogisticRegression(output_type="numpy").fit(train_X, train_y)
    else:
        model = MultiOutputClassifier(cuml.LogisticRegression(output_type="numpy"), n_jobs=1).fit(train_X, train_y.reshape(-1, 1))

    print("Evaluating model")
    train_predictions = model.predict_proba(train_X)
    valid_predictions = model.predict_proba(valid_X)
    test_predictions = model.predict_proba(test_X)
    train_labels = train_y
    valid_labels = valid_y
    test_labels = test_y

    with open(result_file, "wb") as f:
        pickle.dump(((train_predictions, train_labels), (valid_predictions, valid_labels), (test_predictions, test_labels)), f)
    #pd.DataFrame(losses).to_csv(loss_file, index=False)


def build_aa_dataloader(df, n_classes: int, embed_path):
    CLASS_MAPPING = {c: n for n, c in enumerate(MAP[n_classes])}
    if n_classes == 2:
        labels = "label"
    elif n_classes == 3:
        labels = "secstr_3c"
    elif n_classes == 8:
        labels = "secstr_8c"
    
    embeddings = []
    aa_labels = []
    
    for _, row in df.iterrows():
        try:
            # Need to trim labels because ESM embeddings max length is 1022
            tmp_labels = row[labels][:1022]
            if not hasattr(tmp_labels, "__len__") or len(tmp_labels) != len(row["sequence"]):
                continue
            with open(embed_path / f"{row['ID']}.pkl", "rb") as f:
                tmp = pickle.load(f)
            embeddings.append(tmp)
            aa_labels += [CLASS_MAPPING[c] for c in tmp_labels]
        except Exception as e:
            print(e)
            pass
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, np.array(aa_labels)


def knn(X, y, val_X, test_X, perm: np.ndarray, mode: Literal["sklearn", "cuml"] = "sklearn"):
    if mode == "sklearn":
        knn = skKNN(n_neighbors=10).fit(X[perm], y[perm])
    else:
        knn = cuKNN(n_neighbors=10).fit(X[perm], y[perm])
    tr_pred = knn.predict(X)
    val_pred = knn.predict(val_X)
    test_pred = knn.predict(test_X)
    return tr_pred, val_pred, test_pred, knn.kneighbors(X)


def dim_red(X, mode: Literal["umap", "random", "none"] = "umap"):
    if mode == "none":
        return lambda x: x
    elif mode == "umap":
        reducer = cuUMAP(
            n_components=N_DIMS,
            n_neighbors=15,
            output_type="numpy",
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
            verbose=True,
            n_epochs=200,              # Reduce if too slow (default: 500 for large data)
            init='spectral',           # Initialization method
            target_weight=0.5,
        ).fit(X)
    elif mode == "random":
        reducer = cuGRP(
            n_components=N_DIMS,
            output_type="numpy",
            random_state=42,
            verbose=True,
        ).fit(X)
    else:
        raise ValueError(f"Unknown dim reduction mode: {mode}")
    return lambda x: reducer.transform(x)

parser = argparse.ArgumentParser(description="Train on amino-acid embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-base", type=Path, required=True)
parser.add_argument('--n-classes', type=int, required=True, choices=[2, 3, 8], help='Number of classes to consider')
parser.add_argument('--max-layer', type=int, required=True)
parser.add_argument('--lib', type=str, default="cuml", choices=["sklearn", "cuml"])
parser.add_argument('--dim-red', type=str, default="none", choices=["umap", "random", "none"])
parser.add_argument('--no-num', type=int, default=10, help='Number of neighbors for noverlap computation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--force', action='store_true', help='Force recomputation even if results exist')
print(sys.argv)
args = parser.parse_args()

start = time()
print(f"[{datetime.now()}] Starting AA model rolling computation...")

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = args.data_path.stem
model_name = args.embed_base.parent.name
df = pd.read_csv(args.data_path, nrows=N_ROWS)
if "sampled" in df.columns:
    df = df[df["sampled"] == True]
result_fix = "_".join([str(N_ROWS), args.lib, args.dim_red, str(args.seed)])

print(f"[{time() - start:.2f}s] Loading layer 0 embeddings...")
curr_train_X, curr_train_y = build_aa_dataloader(df[df["split"] == "train"], args.n_classes, args.embed_base / "layer_0")
curr_val_X, curr_val_y = build_aa_dataloader(df[df["split"] == "val"], args.n_classes, args.embed_base / "layer_0")
curr_test_X, curr_test_y = build_aa_dataloader(df[df["split"] == "test"], args.n_classes, args.embed_base / "layer_0")

permutation = np.random.permutation(curr_train_X.shape[0])

print(f"[{time() - start:.2f}s] Reducing dimensionality of layer 0 ...")
reducer = dim_red(curr_train_X, mode=args.dim_red)
curr_train_X_red = reducer(curr_train_X)
curr_val_X_red = reducer(curr_val_X)
curr_test_X_red = reducer(curr_test_X)

print(f"[{time() - start:.2f}s] Fitting kNN on layer 0 ...")
curr_train_pred, curr_val_pred, curr_test_pred, (curr_distances, curr_dist_indices) = knn(curr_train_X_red, curr_train_y, curr_val_X_red, curr_test_X_red, perm=permutation, mode=args.lib)

with open(args.embed_base / "layer_0" / f"predictions_{result_fix}_knn.pkl", "wb") as f:
    pickle.dump(((curr_train_pred, curr_train_y.squeeze()), (curr_val_pred, curr_val_y.squeeze()), (curr_test_pred, curr_test_y.squeeze())), f)
print(f"[{time() - start:.2f}s] Training LR on layer 0 ...")
train_lr_head(args.embed_base / "layer_0", curr_train_X_red, curr_train_y, curr_val_X_red, curr_val_y, curr_test_X_red, curr_test_y)

for layer in range(0, args.max_layer + 1):
    print(f"[{time() - start:.2f}s] Processing layer {layer} ...")
    result_folder = args.embed_base / f"layer_{layer}"
    
    twonn_id = compute_id_2NN(curr_distances)
    print(f"[{time() - start:.2f}s] Layer {layer}: 2NN ID = {twonn_id}")
    pd.DataFrame({
        "twonn_id": [twonn_id],
    }).to_csv(result_folder / f"ids_{result_fix}.csv", index=False)

    if layer == args.max_layer:
        break

    print(f"[{time() - start:.2f}s] Loading layer {layer + 1} embeddings...")
    next_train_X, next_train_y = build_aa_dataloader(df[df["split"] == "train"], args.n_classes, args.embed_base / f"layer_{layer + 1}")
    next_val_X, next_val_y = build_aa_dataloader(df[df["split"] == "val"], args.n_classes, args.embed_base / f"layer_{layer + 1}")
    next_test_X, next_test_y = build_aa_dataloader(df[df["split"] == "test"], args.n_classes, args.embed_base / f"layer_{layer + 1}")

    print(f"[{time() - start:.2f}s] Reducing dimensionality of layer {layer + 1} ...")
    reducer = dim_red(next_train_X, mode=args.dim_red)
    permutation = np.random.permutation(next_train_X.shape[0])
    next_train_X_red = reducer(next_train_X)
    next_val_X_red = reducer(next_val_X)
    next_test_X_red = reducer(next_test_X)

    print(f"[{time() - start:.2f}s] Fitting kNN on layer {layer + 1} ...")
    next_train_pred, next_val_pred, next_test_pred, (next_distances, next_dist_indices) = knn(next_train_X_red, next_train_y, next_val_X_red, next_test_X_red, perm=permutation, mode=args.lib)

    with open(args.embed_base / f"layer_{layer + 1}" / f"predictions_{result_fix}_knn.pkl", "wb") as f:
        pickle.dump(((next_train_pred, next_train_y.squeeze()), (next_val_pred, next_val_y.squeeze()), (next_test_pred, next_test_y.squeeze())), f)
    print(f"[{time() - start:.2f}s] Training LR on layer {layer + 1} ...")
    train_lr_head(args.embed_base / f"layer_{layer + 1}", next_train_X_red, next_train_y, next_val_X_red, next_val_y, next_test_X_red, next_test_y)

    noverlap = return_data_overlap(curr_dist_indices, next_dist_indices, k=args.no_num)
    print(f"[{time() - start:.2f}s] Layer {layer}: Neighbor Overlap = {noverlap}")
    pd.DataFrame({
        "neighbor_overlap": [noverlap],
    }).to_csv(result_folder / f"noverlap_{args.no_num}_{result_fix}.csv", index=False)

    curr_dist_indices = copy.deepcopy(next_dist_indices)
