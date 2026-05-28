from datetime import datetime
import random
from time import time

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import argparse

from src.downstream.analyze import prepare_dataset


def main(args):
    print(f"[{datetime.now()}] Starting AA model rolling computation...")
    dataset = args.data_path.stem

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"[{datetime.now()}] Loading dataset...")
    df, labels, val_name, model_suffix, space_suffix = prepare_dataset(
        dataset, args.data_path, args.n_classes, args.level, args.top, args.min
    )
    if (args.data_path.parent / f"results_seq_knn_{dataset}{model_suffix}.pkl").exists() and not args.force:
        print("Results already exist, skipping computation. Use --force to recompute.")
        return

    id_map = {id: i for i, id in enumerate(df["ID"])}
    matrix = np.zeros((len(id_map), len(id_map)), dtype=np.float32)

    print(f"[{datetime.now()}] Computing distance matrix...")
    with open(args.sim_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            
            if i % 1_000_000 == 0:
                print(f"\r{i // 1_000_000}M", end="")
            
            a, b, sim = line.strip().split("\t")
            if a not in id_map or b not in id_map or float(sim) != float(sim):
                continue
            
            matrix[id_map[a], id_map[b]] = 1 - float(sim)
            matrix[id_map[b], id_map[a]] = 1 - float(sim)

    print()
    print("NaNs in distance matrix:", np.isnan(matrix).sum())
    print("Mean sequence distance:", np.nanmean(matrix[matrix > 0]))
    print("Median sequence distance:", np.nanmedian(matrix[matrix > 0]))

    train_idx = [id_map[idx] for idx in df[df["split"] == "train"]["ID"]]
    train_y = np.array(df[df["split"] == "train"][labels].values).astype(np.float32)
    valid_idx = [id_map[idx] for idx in df[df["split"] == val_name]["ID"]]
    valid_y = np.array(df[df["split"] == val_name][labels].values).astype(np.float32)
    test_idx = [id_map[idx] for idx in df[df["split"] == "test"]["ID"]]
    test_y = np.array(df[df["split"] == "test"][labels].values).astype(np.float32)

    print(f"[{datetime.now()}] Training kNN classifier...")
    knn = KNeighborsClassifier(
        n_neighbors=10,
        metric='precomputed',   # tells sklearn the matrix IS the distances
        algorithm='brute',      # required for precomputed distances
        weights='distance',     # to weight by inverse distance
    ).fit(matrix[train_idx][:, train_idx], train_y)

    print(f"[{datetime.now()}] Computing predictions...")
    train_preds = knn.predict_proba(matrix[train_idx][:, train_idx])
    val_preds = knn.predict_proba(matrix[valid_idx][:, train_idx])
    test_preds = knn.predict_proba(matrix[test_idx][:, train_idx])

    # Save results
    print(f"[{datetime.now()}] Saving results...")
    with open(args.data_path.parent.parent / "seq_knns" / f"results_{dataset}{model_suffix}.pkl", "wb") as f:
        pickle.dump(((train_preds, train_y), (val_preds, valid_y), (test_preds, test_y)), f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on amino-acid embeddings.")
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument("--sim-path", type=Path, required=True)
    parser.add_argument('--k', type=int, default=10, 
                        help='Number of neighbors for all neighbor-based computations')
    parser.add_argument('--task', type=str, default='multi-class', choices=['regression', 'binary', 'multi-class', 'multi-label'], 
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
    parser.add_argument('--n-classes', type=int, default=None, 
                        help='Number of classes to consider. Only for amino-acid level prediction tasks.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if results exist')
    args = parser.parse_args()
    main(args)