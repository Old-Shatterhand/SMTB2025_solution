import os
os.environ['CUPY_NVCC_GENERATE_CODE'] = 'current'

import copy
import random
import pickle
import argparse
from time import time
from pathlib import Path
from datetime import datetime

import cuml
from sklearn.multioutput import MultiOutputClassifier
import torch
import numpy as np
import pandas as pd
from cuml import PCA
from cuml.neighbors import KNeighborsClassifier as kNN

from src.downstream.dada import compute_id_2NN, return_data_overlap


N_ROWS = 300

MAP = {
    2: ["X", "M"],
    3: ["H", "E", "C"],
    8: ["G", "H", "I", "B", "E", "T", "S", "-"],
}


def build_aa_dataloader(df: pd.DataFrame, n_classes: int, embed_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Build amino-acid level dataloader from precomputed embeddings.
    
    Args:
        df (pd.DataFrame): DataFrame containing sequences and labels.
        n_classes (int): Number of classes for classification (2, 3, or 8).
        embed_path (Path): Path to the directory containing precomputed embeddings.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of embeddings and corresponding labels.
    """
    CLASS_MAPPING = {c: n for n, c in enumerate(MAP[n_classes])}
    if n_classes == 2:
        labels = "label"  # ligand-binding dataset
    elif n_classes == 3:
        labels = "secstr_3c"  # 3-class SSP
    elif n_classes == 8:
        labels = "secstr_8c"  # 8-class SSP
    
    embeddings = []
    aa_labels = []
    
    for _, row in df.iterrows():
        try:
            # fetch labels, if they are nan or otherwise broken, skip
            tmp_labels = row[labels]
            if not hasattr(tmp_labels, "__len__") or len(tmp_labels) != len(row["sequence"]):
                continue
            
            # Need to trim labels because ESM embeddings max length is 1022
            tmp_labels = tmp_labels[:1022]

            # load embeddings
            with open(embed_path / f"{row['ID']}.pkl", "rb") as f:
                tmp = pickle.load(f)
            
            embeddings.append(tmp)
            aa_labels += [CLASS_MAPPING[c] for c in tmp_labels]
        except Exception as e:
            print(e)
            pass
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, np.array(aa_labels)


def knn(
        out_folder: Path, 
        train_X: np.ndarray, 
        train_y: np.ndarray, 
        val_X: np.ndarray, 
        val_y: np.ndarray, 
        test_X: np.ndarray, 
        test_y: np.ndarray, 
        perm: np.ndarray, 
        n_classes: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Train and evaluate a kNN classifier using cuML.

    Args:
        out_folder (Path): Output folder to save predictions.
        train_X (np.ndarray): Training features.
        train_y (np.ndarray): Training labels.
        val_X (np.ndarray): Validation features.
        val_y (np.ndarray): Validation labels.
        test_X (np.ndarray): Test features.
        test_y (np.ndarray): Test labels.
        perm (np.ndarray): Permutation of training indices for shuffling.
        n_classes (int): Number of classes.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Distances and indices of k-nearest neighbors for ID and NOverlap computation.
    """
    knn = kNN(n_neighbors=10).fit(train_X[perm], train_y[perm])

    print("Evaluating kNN model")
    train_preds = knn.predict(train_X)
    val_preds = knn.predict(val_X)
    test_preds = knn.predict(test_X)

    with open(out_folder / f"predictions_knn_{n_classes}.pkl", "wb") as f:
        pickle.dump(((train_preds, train_y), (val_preds, val_y), (test_preds, test_y)), f)
    
    return knn.kneighbors(train_X)


def train_lr_head(
        out_folder: Path, 
        train_X: np.ndarray, 
        train_y: np.ndarray, 
        val_X: np.ndarray, 
        val_y: np.ndarray, 
        test_X: np.ndarray, 
        test_y: np.ndarray, 
        perm: np.ndarray, 
        n_classes: int = 2, 
        binary: bool = False
    ) -> None:
    """
    Train and evaluate a Logistic Regression classifier using cuML.

    Args:
        out_folder (Path): Output folder to save predictions.
        train_X (np.ndarray): Training features.
        train_y (np.ndarray): Training labels.
        val_X (np.ndarray): Validation features.
        val_y (np.ndarray): Validation labels.
        test_X (np.ndarray): Test features.
        test_y (np.ndarray): Test labels.
        perm (np.ndarray): Permutation of training indices for shuffling.
        n_classes (int): Number of classes.
        binary (bool): Whether the classification is binary.
    """
    if binary:
        model = cuml.LogisticRegression(output_type="numpy").fit(train_X[perm], train_y[perm])
    else:
        model = MultiOutputClassifier(cuml.LogisticRegression(output_type="numpy"), n_jobs=1).fit(train_X[perm], train_y[perm].reshape(-1, 1))

    print("Evaluating LR model")
    train_preds = model.predict_proba(train_X)
    val_preds = model.predict_proba(val_X)
    test_preds = model.predict_proba(test_X)

    with open(out_folder / f"predictions_lr_{n_classes}.pkl", "wb") as f:
        pickle.dump(((train_preds, train_y), (val_preds, val_y), (test_preds, test_y)), f)


def main(args):
    start = time()
    print(f"[{datetime.now()}] Starting AA model rolling computation...")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the dataset and reduce it to the sampled sequences only (if applicable)
    df = pd.read_csv(args.data_path)  # , nrows=N_ROWS)
    if "sampled" in df.columns:
        df = df[df["sampled"] == True]
    calcs = set(args.calcs)

    # Load the first layer embeddings
    print(f"[{time() - start:.2f}s] Loading layer 0 embeddings...")
    curr_train_X, curr_train_y = build_aa_dataloader(df[df["split"] == "train"], args.n_classes, args.embed_base / "layer_0")
    curr_val_X, curr_val_y = build_aa_dataloader(df[df["split"] == "val"], args.n_classes, args.embed_base / "layer_0")
    curr_test_X, curr_test_y = build_aa_dataloader(df[df["split"] == "test"], args.n_classes, args.embed_base / "layer_0")
    permutation = np.random.permutation(curr_train_X.shape[0])

    if {'knn', 'id', 'no'}.intersection(calcs):
        print(f"[{time() - start:.2f}s] Fitting kNN on layer 0 ...")
        curr_distances, curr_dist_indices = knn(args.embed_base / "layer_0", curr_train_X, curr_train_y, curr_val_X, curr_val_y, curr_test_X, curr_test_y, perm=permutation, n_classes=args.n_classes)

    if 'lr' in calcs:
        print(f"[{time() - start:.2f}s] Training LR on layer 0 ...")
        train_lr_head(args.embed_base / "layer_0", curr_train_X, curr_train_y, curr_val_X, curr_val_y, curr_test_X, curr_test_y, perm=permutation, n_classes=args.n_classes)

    for layer in range(args.max_layer + 1):
        print(f"[{time() - start:.2f}s] Processing layer {layer} ...")
        result_folder = args.embed_base / f"layer_{layer}"
        
        # Compute 2NN ID
        if 'id' in calcs:
            twonn_id = compute_id_2NN(curr_distances)
            print(f"[{time() - start:.2f}s] Layer {layer}: 2NN ID = {twonn_id}")
            pd.DataFrame({
                "twonn_id": [twonn_id],
            }).to_csv(result_folder / f"ids_{args.k}_{args.n_classes}.csv", index=False)

        if 'pca' in calcs:
            # calculate PCA and PCA-induced volume
            pca = PCA(svd_solver='auto').fit(curr_train_X)
            with open(result_folder / f"pca_{args.n_classes}.pkl", "wb") as f:
                pickle.dump(pca.explained_variance_, f)
        
        # if 'var' in calcs:
        #     pd.DataFrame({
        #         "var_k": [pca.explained_variance_[:args.k].sum() / pca.explained_variance_.sum()],
        #     }).to_csv(result_folder / f"var_k_{args.k}_{args.n_classes}.csv", index=False)

        #     pd.DataFrame({
        #         "var_95": next(i for i, v in enumerate(np.cumsum(pca.explained_variance_) / sum(pca.explained_variance_)) if v >= 0.95)
        #     }).to_csv(result_folder / f"var_95_{args.k}_{args.n_classes}.csv", index=False)
        
        # if 'vol' in calcs:
        #     pd.DataFrame({
        #         "vol": [(np.pi ** (args.k / 2)) / cuml.special.gamma((args.k / 2) + 1) * np.prod(pca.explained_variance_[:args.k])],  # k-dimensional hyperellipsoid volume
        #     }).to_csv(result_folder / f"vol_{args.k}_{args.n_classes}.csv", index=False)

        if layer == args.max_layer:
            break

        # Load the next layer embeddings
        print(f"[{time() - start:.2f}s] Loading layer {layer + 1} embeddings...")
        next_train_X, next_train_y = build_aa_dataloader(df[df["split"] == "train"], args.n_classes, args.embed_base / f"layer_{layer + 1}")
        next_val_X, next_val_y = build_aa_dataloader(df[df["split"] == "val"], args.n_classes, args.embed_base / f"layer_{layer + 1}")
        next_test_X, next_test_y = build_aa_dataloader(df[df["split"] == "test"], args.n_classes, args.embed_base / f"layer_{layer + 1}")
        permutation = np.random.permutation(next_train_X.shape[0])

        if {'knn', 'id', 'no'}.intersection(calcs):
            print(f"[{time() - start:.2f}s] Fitting kNN on layer {layer + 1} ...")
            next_distances, next_dist_indices = knn(args.embed_base / f"layer_{layer + 1}", next_train_X, next_train_y, next_val_X, next_val_y, next_test_X, next_test_y, perm=permutation, n_classes=args.n_classes)

        if 'lr' in calcs:
            print(f"[{time() - start:.2f}s] Training LR on layer {layer + 1} ...")
            train_lr_head(args.embed_base / f"layer_{layer + 1}", next_train_X, next_train_y, next_val_X, next_val_y, next_test_X, next_test_y, perm=permutation, n_classes=args.n_classes)

        if 'no' in calcs:
            # Compute Neighborhood Overlap between this layer (next) and the previous one (curr)
            noverlap = return_data_overlap(curr_dist_indices, next_dist_indices, k=args.k)
            print(f"[{time() - start:.2f}s] Layer {layer}: Neighbor Overlap = {noverlap}")
            pd.DataFrame({
                "neighbor_overlap": [noverlap],
            }).to_csv(result_folder / f"noverlap_{args.k}_{args.n_classes}.csv", index=False)

        if {'knn', 'id', 'no'}.intersection(calcs):
            # save the distance indices for next Neighborhood Overlap computation
            curr_dist_indices = copy.deepcopy(next_dist_indices)
            curr_distances = copy.deepcopy(next_distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on amino-acid embeddings.")
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument("--embed-base", type=Path, required=True)
    parser.add_argument('--n-classes', type=int, required=True, choices=[2, 3, 8], help='Number of classes to consider')
    parser.add_argument('--max-layer', type=int, required=True)
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors for all neighbor-based computations')
    parser.add_argument('--calcs', nargs='+', default=['lr', 'knn', 'id', 'no', 'pca'], choices=['lr', 'knn', 'id', 'no', 'pca'], help='Calculations to perform. Choices are '
                        '[lr]: Logistic Regression, '
                        '[knn]: k-Nearest Neighbors, '
                        '[id]: Intrinsic Dimension, '
                        '[no]: Neighborhood Overlap, '
                        '[pca]: Compute eigenvectors from PCA.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if results exist')
    args = parser.parse_args()
    main(args)
