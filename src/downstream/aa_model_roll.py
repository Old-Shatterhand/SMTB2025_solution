from datetime import datetime
import random
import pickle
import argparse
from time import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from dadapy import data
from tqdm import tqdm


def build_aa_dataloader(df, embed_path):
    label = "secstr_3c" if "secstr_3c" in df.columns else "labels"
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            tmp_labels = row[label]
            if not hasattr(tmp_labels, "__len__") or len(tmp_labels) != len(row["sequence"]):
                continue
            with open(embed_path / f"{row['ID']}.pkl", "rb") as f:
                tmp = pickle.load(f)
            embeddings.append(tmp)
        except Exception:
            pass
    return np.concatenate(embeddings, axis=0)


parser = argparse.ArgumentParser(description="Train on amino-acid embeddings.")
parser.add_argument('--data-path', type=Path, required=True)
parser.add_argument("--embed-base", type=Path, required=True)
parser.add_argument('--max-layer', type=int, required=True)
parser.add_argument('--no-num', type=int, default=10, help='Number of neighbors for noverlap computation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--force', action='store_true', help='Force recomputation even if results exist')
args = parser.parse_args()

start = time()
print(f"[{datetime.now()}] Starting AA model rolling computation...")

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = args.data_path.stem
model_name = args.embed_base.parent.name
df = pd.read_csv(args.data_path)  # , nrows=500)

# rolling variables
print(f"[{time() - start:.2f}s] Loading initial layer embeddings...")
curr_train_X = build_aa_dataloader(df[df["split"] == "train"], args.embed_base / "layer_0")
print(curr_train_X.shape)
curr_X = data.Data(curr_train_X)
curr_X.compute_distances(10, n_jobs=8)

for layer in range(0, args.max_layer + 1):
    print(f"[{time() - start:.2f}s] Processing layer {layer}...")
    result_folder = args.embed_base / f"layer_{layer}"
    
    twonn_id = curr_X.compute_id_2NN()[0]
    print(f"[{time() - start:.2f}s] Layer {layer}: 2NN ID = {twonn_id}")
    pd.DataFrame({
        "twonn_id": [twonn_id],
    }).to_csv(result_folder / "ids.csv", index=False)

    if layer == args.max_layer:
        break

    print(f"[{time() - start:.2f}s] Loading next layer embeddings...")
    next_train_X = build_aa_dataloader(df[df["split"] == "train"], args.embed_base / f"layer_{layer + 1}")
    next_X = data.Data(next_train_X)
    next_X.compute_distances(10, n_jobs=8)

    noverlap = curr_X.return_data_overlap(coordinates=next_train_X, distances=next_X.distances, dist_indices=next_X.dist_indices, k=args.no_num)
    print(f"[{time() - start:.2f}s] Layer {layer}: Neighbor Overlap = {noverlap}")
    pd.DataFrame({
        "neighbor_overlap": [noverlap],
    }).to_csv(result_folder / "noverlap_10.csv", index=False)

    curr_train_X = next_train_X
    curr_X = next_X
