from pathlib import Path
import argparse
import pickle
import random
from time import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch

from src.downstream.utils import build_dataloader


parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--top-k', type=int, default=4, help='Number of top labels to consider')
parser.add_argument('--level', required=True, choices=["class", "fold", "family", "superfamily"])
parser.add_argument('--function', type=str, required=True, choices=["rf", "knn", "lr"])
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

start = time()

layer = int(args.embed_path.name.split("_")[-1])
model_name = args.embed_path.parent.parent.name
result_folder = args.embed_path.parent.parent / "scope_40_208" / f"layer_{layer}"
result_file = result_folder / f"predictions_{args.function}_{args.level}_{args.top_k}.pkl"
# if result_file.exists():
#     print(f"Results already exist.")
#     exit(0)
result_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB" / "datasets" / "scope_40_208.csv")
top_k_labels = set(x[0] for x in list(sorted(dict(df[args.level].value_counts()).items(), key=lambda x: x[1], reverse=True)[:args.top_k]))
df = df[df[args.level].isin(top_k_labels)].reset_index(drop=True)
train_X, train_Y = build_dataloader(df, args.embed_path, args.level)

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.function == "rf":
    model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
elif args.function == "lr":
    model = LogisticRegression(random_state=args.seed)
elif args.function == "knn":
    model = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute", metric="cosine")
model.fit(train_X, train_Y)
train_prediction = model.predict(train_X)
with open(result_file, "wb") as f:
    pickle.dump(((train_prediction, train_Y),), f)

print(f"Done in {time() - start:.2f} seconds")
