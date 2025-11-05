from pathlib import Path
import argparse
import pickle
from time import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument("--embed-path", type=Path, required=True)
parser.add_argument('--top-k', type=int, default=4, help='Number of top labels to consider')
parser.add_argument('--level', required=True, choices=["class", "fold", "family", "superfamily"])
args = parser.parse_args()

start = time()

layer = int(args.embed_path.name.split("_")[-1])
model_name = args.embed_path.parent.parent.name
result_folder = args.embed_path.parent.parent / "scope_40_208" / f"layer_{layer}"
result_file = result_folder / f"correlations_{args.level}_{args.top_k}.csv"
if result_file.exists():
    print(f"Results already exist.")
    exit(0)
result_folder.mkdir(parents=True, exist_ok=True)

# Load SCOPE data and reduce to the top-k labels
df = pd.read_csv(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB" / "datasets" / "scope_40_208.csv")
top_k_labels = set(x[0] for x in list(sorted(dict(df[args.level].value_counts()).items(), key=lambda x: x[1], reverse=True)[:args.top_k]))
df = df[df[args.level].isin(top_k_labels)].reset_index(drop=True)

# Load the similarity martix from foldseek and reduce it to the current set of proteins
with open("/scratch/SCRATCH_SAS/roman/SMTB/datasets/scope_foldseek.pkl", "rb") as f:
    M, id_map = pickle.load(f)
mask = [id_map[idx] for idx in df["scope_id"]]
M = M[np.ix_(mask, mask)]

# Load embeddings and compute their norm
embeds = {}
norms = {}
for idx in df["ID"]:
    with open(args.embed_path / f"{idx}.pkl", "rb") as f:
        embeds[idx] = pickle.load(f)
    norms[idx] = np.linalg.norm(embeds[idx])

# Compute the embedding cosine similarity matrix
E = np.zeros((len(df), len(df)))
ids = list(df["ID"].values)
for a, idx_a in enumerate(ids):
    E[a, a] = 1
    for b, idx_b in enumerate(ids[a + 1:], start=a + 1):
        E[a, b] = np.dot(embeds[idx_a], embeds[idx_b]) / (norms[idx_a] * norms[idx_b])
        E[b, a] = E[a, b]

# Store the spearman and pearson correlations between Foldseek (M) and embedding (E) similarity matrices
pd.DataFrame({
    "pearson": [np.corrcoef(M.flatten(), E.flatten())[0, 1]],
    "spearman": [spearmanr(M.flatten(), E.flatten()).statistic],
}).to_csv(result_file, index=False)

print(f"Script finished in {time() - start:.2f} seconds")
