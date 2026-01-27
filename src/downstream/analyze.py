import os
os.environ['CUPY_NVCC_GENERATE_CODE'] = 'current'

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
from cuml import PCA
from cuml.neighbors import KNeighborsClassifier as kNN_class, KNeighborsRegressor as kNN_reg

from src.downstream.utils import compute_id_2NN, return_data_overlap


N_ROWS = 300

MAP = {
    2: ["X", "M"],
    3: ["H", "E", "C"],
    8: ["G", "H", "I", "B", "E", "T", "S", "-"],
}


def build_wp_dataloader(df: pd.DataFrame, embed_path: Path, labels: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a DataLoader for the given DataFrame and embedding path.

    Args:
        df: DataFrame containing the data.
        embed_path: Path to the directory containing the embeddings.
        labels: Column name(s) for the target labels.

    Returns:
        Tuple of (inputs, targets) where inputs are the embeddings and targets are the labels.
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
        except Exception as e:
            print(e)
            pass
    
    inputs = np.stack(embeddings)
    targets = np.array(df[df["ID"].isin(valid_ids)][labels].values).astype(np.float32)
    return inputs, targets


def build_aa_dataloader(df: pd.DataFrame, embed_path: Path, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build amino-acid level dataloader from precomputed embeddings.
    
    Args:
        df (pd.DataFrame): DataFrame containing sequences and labels.
        embed_path (Path): Path to the directory containing precomputed embeddings.
        n_classes (int): Number of classes for classification (2, 3, or 8).
    
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


def build_dataloader(df: pd.DataFrame, embed_path: Path, labels: str | list[str] | int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a DataLoader for the given DataFrame and embedding path.

    Args:
        df: DataFrame containing the data.
        embed_path: Path to the directory containing the embeddings.
        labels: Column name(s) for the target labels or number of classes for amino-acid data.
    Returns:
        Tuple of (inputs, targets) where inputs are the embeddings and targets are the labels.

    """
    if isinstance(labels, int):
        return build_aa_dataloader(df, embed_path, labels)
    else:
        return build_wp_dataloader(df, embed_path, labels)


def knn(
        out_folder: Path, 
        train_X: np.ndarray, 
        train_y: np.ndarray, 
        val_X: np.ndarray, 
        val_y: np.ndarray, 
        test_X: np.ndarray, 
        test_y: np.ndarray, 
        perm: np.ndarray,
        n_neighbors: int = 10,
        task: Literal["regression", "binary", "class"] = "binary",
        suffix: str = "",
        force: bool = False,
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
        n_neighbors (int): Number of neighbors for kNN.
        task: Type of task: "regression", "binary", or "class".
        suffix (str): Suffix for output files.
        force (bool): Whether to force retraining even if predictions exist.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Distances and indices of k-nearest neighbors for ID and NOverlap computation.
    """
    if task == "regression":
        knn = kNN_reg(output_type="numpy", n_neighbors=n_neighbors).fit(train_X[perm], train_y[perm])
    else:
        knn = kNN_class(output_type="numpy", n_neighbors=n_neighbors).fit(train_X[perm], train_y[perm])

    if not (r_file := (out_folder / f"predictions_knn_{suffix}.pkl")).exists() or force:
        print("Evaluating kNN model")
        if task == "regression":
            train_preds = knn.predict(train_X)
            val_preds = knn.predict(val_X)
            test_preds = knn.predict(test_X)
        else:
            train_preds = knn.predict(train_X)
            val_preds = knn.predict(val_X)
            test_preds = knn.predict(test_X)

        with open(r_file, "wb") as f:
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
        task: Literal["regression", "binary", "class"] = "binary",
        suffix: str = "",
        force: bool = False,
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
        task: Type of task: "regression", "binary", or "class".
        suffix (str): Suffix for output files.
        force (bool): Whether to force retraining even if predictions exist.
    """
    if (r_file := (out_folder / f"predictions_lr_{suffix}.pkl")).exists() and not force:
        print("LR predictions already exist.")
        return
    
    if task == "regression":
        model = cuml.LinearRegression(output_type="numpy").fit(train_X[perm], train_y[perm])
    elif task == "binary":
        model = cuml.LogisticRegression(output_type="numpy").fit(train_X[perm], train_y[perm])
    else:
        model = MultiOutputClassifier(cuml.LogisticRegression(output_type="numpy"), n_jobs=1).fit(train_X[perm], train_y[perm].reshape(-1, 1))

    print("Evaluating LR model")
    if task == "regression":
        train_preds = model.predict(train_X)
        val_preds = model.predict(val_X)
        test_preds = model.predict(test_X)
    else:
        train_preds = model.predict_proba(train_X)
        val_preds = model.predict_proba(val_X)
        test_preds = model.predict_proba(test_X)

    with open(r_file, "wb") as f:
        pickle.dump(((train_preds, train_y), (val_preds, val_y), (test_preds, test_y)), f)


def main(args):
    start = time()
    print(f"[{datetime.now()}] Starting AA model rolling computation...")
    dataset = args.data_path.stem
    base_result_folder = args.embed_base.parent / dataset 

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the dataset and reduce it to the sampled sequences only (if applicable)
    df = pd.read_csv(args.data_path)  # , nrows=N_ROWS)
    val_name = "val" if "val" in df["split"].unique() else "valid"
    if "sampled" in df.columns:
        df = df[df["sampled"] == True]
    labels = "label"
    model_suffix, space_suffix = [args.k], [args.k]
    if args.n_classes is not None:  # amino-acid level prediction
        labels = args.n_classes
        model_suffix.append(args.n_classes)
    else:  # whole-protein level prediction
        if args.level is not None:
            model_suffix.append(args.level)
            space_suffix.append(args.level)
            # Reduce the datasets for SCOPe superfamiliy/fold prediction to only hold samples of the the most frequent k classes or those classes with at least min-x samples
            if args.top is not None == args.min is not None:
                raise ValueError("Exactly one of --top and --min must be specified for SCOPe superfamiliy/fold predictions.")
            if args.min is not None:
                model_suffix.append(f"min{args.min}")
                space_suffix.append(f"min{args.min}")
                label_counts = df[args.level].value_counts()
                valid_labels = label_counts[label_counts >= args.min].index
                df = df[df[args.level].isin(valid_labels)].reset_index(drop=True)
            if args.top is not None:
                model_suffix.append(f"top{args.top}")
                space_suffix.append(f"top{args.top}")
                top_k_labels = set(x[0] for x in list(sorted(dict(df[args.level].value_counts()).items(), key=lambda x: x[1], reverse=True)[:args.top]))
                df = df[df[args.level].isin(top_k_labels)].reset_index(drop=True)
            labels = args.level
        if dataset == "deeploc2":
            labels = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]
    calcs = set(args.calcs)
    model_suffix = "_".join(map(str, model_suffix))
    space_suffix = "_".join(map(str, space_suffix))

    # Load the first layer embeddings
    print(f"[{time() - start:.2f}s] Loading layer 0 embeddings...")
    curr_train_X, curr_train_y = build_dataloader(df[df["split"] == "train"], args.embed_base / "layer_0", labels)
    if {'knn', 'id', 'no', 'lr'}.intersection(calcs):
        curr_val_X, curr_val_y = build_dataloader(df[df["split"] == val_name], args.embed_base / "layer_0", labels)
        curr_test_X, curr_test_y = build_dataloader(df[df["split"] == "test"], args.embed_base / "layer_0", labels)
    
    # The permutations have to be the same for all layers otherwise, the neighborhood overlap cannot be computed properly
    permutation = np.random.permutation(curr_train_X.shape[0])

    if {'knn', 'id', 'no'}.intersection(calcs):
        print(f"[{time() - start:.2f}s] Fitting kNN on layer 0 ...")
        curr_distances, curr_dist_indices = knn(
            out_folder=base_result_folder / "layer_0", 
            train_X=curr_train_X, 
            train_y=curr_train_y, 
            val_X=curr_val_X, 
            val_y=curr_val_y, 
            test_X=curr_test_X, 
            test_y=curr_test_y, 
            perm=permutation, 
            n_neighbors=args.k, 
            task=args.task, 
            suffix=model_suffix, 
            force=args.force,
        )            

    if 'lr' in calcs:
        print(f"[{time() - start:.2f}s] Training LR on layer 0 ...")
        train_lr_head(
            out_folder=base_result_folder / "layer_0", 
            train_X=curr_train_X, 
            train_y=curr_train_y, 
            val_X=curr_val_X, 
            val_y=curr_val_y, 
            test_X=curr_test_X, 
            test_y=curr_test_y, 
            perm=permutation, 
            task=args.task, 
            suffix=model_suffix, 
            force=args.force
        )
    
    for layer in range(args.max_layer + 1):
        print(f"[{time() - start:.2f}s] Processing layer {layer} ...")
        result_folder = base_result_folder / f"layer_{layer}"
        print(f"[{time() - start:.2f}s] Result folder: {result_folder}")
        
        # Compute 2NN ID
        # if 'id' in calcs and (not (r_file := (result_folder / f"ids_{space_suffix}.csv")).exists() or args.force):
        #     twonn_id = compute_id_2NN(curr_distances)
        #     print(f"[{time() - start:.2f}s] Layer {layer}: 2NN ID = {twonn_id}")
        #     pd.DataFrame({
        #         "twonn_id": [twonn_id],
        #     }).to_csv(r_file, index=False)

        # calculate PCA and PCA-induced volume
        if 'pca' in calcs and (not (r_file := (result_folder / f"pca_{space_suffix}.pkl")).exists() or args.force):
            pca = PCA(svd_solver='auto').fit(curr_train_X)
            print(f"[{time() - start:.2f}s] Layer {layer}: PCA computed.")
            with open(r_file, "wb") as f:
                pickle.dump(pca.explained_variance_, f)
        
        if layer == args.max_layer:
            break

        # Load the next layer embeddings
        print(f"[{time() - start:.2f}s] Loading layer {layer + 1} embeddings...")
        next_train_X, next_train_y = build_dataloader(df[df["split"] == "train"], args.embed_base / f"layer_{layer + 1}", labels)
        if {'knn', 'id', 'no', 'lr'}.intersection(calcs):
            next_val_X, next_val_y = build_dataloader(df[df["split"] == val_name], args.embed_base / f"layer_{layer + 1}", labels)
            next_test_X, next_test_y = build_dataloader(df[df["split"] == "test"], args.embed_base / f"layer_{layer + 1}", labels)

        if {'knn', 'id', 'no'}.intersection(calcs):
            print(f"[{time() - start:.2f}s] Fitting kNN on layer {layer + 1} ...")
            next_distances, next_dist_indices = knn(
                out_folder=base_result_folder / f"layer_{layer + 1}", 
                train_X=next_train_X, 
                train_y=next_train_y, 
                val_X=next_val_X, 
                val_y=next_val_y, 
                test_X=next_test_X, 
                test_y=next_test_y, 
                perm=permutation, 
                n_neighbors=args.k, 
                task=args.task, 
                suffix=model_suffix, 
                force=args.force
            )

        if 'lr' in calcs:
            print(f"[{time() - start:.2f}s] Training LR on layer {layer + 1} ...")
            train_lr_head(
                out_folder=base_result_folder / f"layer_{layer + 1}", 
                train_X=next_train_X, 
                train_y=next_train_y, 
                val_X=next_val_X, 
                val_y=next_val_y, 
                test_X=next_test_X, 
                test_y=next_test_y, 
                perm=permutation, 
                task=args.task, 
                suffix=model_suffix, 
                force=args.force
            )
        
        # Compute Neighborhood Overlap between this layer (next) and the previous one (curr)
        if 'no' in calcs and (not (r_file := (result_folder / f"noverlap_{space_suffix}.csv")).exists() or args.force):
            noverlap = return_data_overlap(curr_dist_indices, next_dist_indices, k=args.k)
            print(f"[{time() - start:.2f}s] Layer {layer}: Neighbor Overlap = {noverlap}")
            pd.DataFrame({
                "neighbor_overlap": [noverlap],
            }).to_csv(r_file, index=False)

        # save the distance indices for next Neighborhood Overlap computation
        if {'knn', 'id', 'no'}.intersection(calcs):
            curr_dist_indices = copy.deepcopy(next_dist_indices)
            curr_distances = copy.deepcopy(next_distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on amino-acid embeddings.")
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument("--embed-base", type=Path, required=True)
    parser.add_argument('--max-layer', type=int, required=True)
    parser.add_argument('--k', type=int, default=10, 
                        help='Number of neighbors for all neighbor-based computations')
    parser.add_argument('--task', type=str, default='classification', choices=['regression', 'binary', 'class'], 
                        help='Type of prediction task.') 
    parser.add_argument('--level', type=str, default=None, choices=["superfamily", "fold"], 
                        help='Level of SCOPe hierarchy to consider. ' \
                        'Only used for SCOPe fold/superfamily prediction.')
    parser.add_argument('--top', type=int, default=None, 
                        help='Number of top labels to consider. ' \
                        'Only used for SCOPe fold/superfamily prediction, mutually exclusive with --min.')
    parser.add_argument('--min', type=int, default=None, 
                        help="Minimum number of samples for a label to be included. " \
                        "Only used for SCOPe fold/superfamily prediction, mutually exclusive with --top.")
    parser.add_argument('--n-classes', type=int, default=None, choices=[2, 3, 8], 
                        help='Number of classes to consider. Only for amino-acid level prediction tasks.')
    parser.add_argument('--calcs', nargs='+', default=['lr', 'knn', 'id', 'no', 'pca'], 
                        choices=['lr', 'knn', 'id', 'no', 'pca'], help='Calculations to perform. Choices are '
                        '[lr]: Logistic Regression, '
                        '[knn]: k-Nearest Neighbors, '
                        '[id]: Intrinsic Dimension, '
                        '[no]: Neighborhood Overlap, '
                        '[pca]: Compute eigenvectors from PCA.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if results exist')
    args = parser.parse_args()
    main(args)
