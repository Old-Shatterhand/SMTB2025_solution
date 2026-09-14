import argparse
import pickle
from time import time
from datetime import datetime
from cuml import GaussianRandomProjection
import cupy
from pathlib import Path
from filelock import FileLock

from src.viz.constants import LAYERS, MODELS
from src.downstream.probe_layer import prepare_dataset, build_dataloader

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def grp(X):
    return GaussianRandomProjection(n_components=128, random_state=42, output_type="cupy").fit_transform(X)

def load_lysozymes(model, layer, artificial: bool = False):
    dataset = "lysosomes" if artificial else "lysosomes_natural"
    embed_base = BASE / "embeddings" / model / dataset / f"layer_{layer}"

    df, labels, _, _, _ = prepare_dataset(
        dataset, BASE / "datasets" / f"{dataset}.csv", None, None, None, None, max_rows=None, seed=42
    )
    return df, embed_base, labels


def load_stability(model, layer, artificial: bool = True):
    embed_base = BASE / "embeddings" / model / "stability" / f"layer_{layer}"
    
    df, labels, _, _, _ = prepare_dataset(
        "stability_random", BASE / "datasets" / "stability_random.csv", None, None, None, None, max_rows=None, seed=42
    )
    df = df[df["artificial"] == artificial]
    return df, embed_base, labels


def load_scope(model, layer):
    embed_base = BASE / "embeddings" / model / "scope_40_208" / f"layer_{layer}"
    
    df, labels, _, _, _ = prepare_dataset(
        "scope_40_208", BASE / "datasets" / "scope_40_208.csv", None, "fold", None, 10, max_rows=None, seed=42
    )
    return df, embed_base, labels


def load(model, layer, dataset):
    if dataset in ["art_lys", "nat_lys"]:
        artificial = dataset == "art_lys"
        return load_lysozymes(model, layer, artificial)
    elif dataset == "stability":
        return load_stability(model, layer, artificial=True)
    elif dataset == "scope":
        return load_scope(model, layer)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def comp_mean_cov(df, embed_base, labels):
    X, _, _ = build_dataloader(df[df["split"] == "train"], embed_base, labels)
    X = grp(cupy.array(X))  # Convert to CuPy array for GPU computation

    mu_art = cupy.mean(X, axis=0)
    cov_art = cupy.cov(X, rowvar=False)

    return mu_art, cov_art


def mahalanobis_head(mu1, cov1, mu2, cov2, reg_eps=1e-6):
    cov_pooled = ((cov2.shape[0] - 1) * cov2 + (cov1.shape[0] - 1) * cov1) / (cov2.shape[0] + cov1.shape[0] - 2)
    cov_pooled += cupy.eye(cov_pooled.shape[0]) * reg_eps
    delta = mu1 - mu2
    return float(cupy.sqrt(cupy.dot(cupy.dot(delta, cupy.linalg.inv(cov_pooled)), delta.T)))


def routine(args):
    start = time()
    print(f"[{datetime.now()}] Starting Mahalanobis distance computation...")
    args.force |= args.fforce  # If fforce is set, also set force
    
    out_pkl = BASE / "mahalanobis.pkl"
    print(f"[{time() - start:.2f}s] Results file: {out_pkl}")
    if out_pkl.exists():
        print(f"[{time() - start:.2f}s] Loading existing results cache...")
        with open(out_pkl, "rb") as f:
            results = pickle.load(f)
    else:
        print(f"[{time() - start:.2f}s] No existing results cache found. Initializing fresh results.")
        results = {}

    # Sort datasets to ensure consistent ordering and list model if not already present in results
    ds1, ds2 = list(sorted(args.datasets))
    if args.model not in results:
        results[args.model] = {}

    # Check if results already exist for the given model and datasets
    if (ds1, ds2) in results[args.model] and not args.force:
        print(f"[{time() - start:.2f}s] Results for {args.model} and datasets {ds1}, {ds2} already exist. Skipping...")
        return
    
    results = []

    for layer in range(LAYERS[args.model] + 1):
        print(f"[{time() - start:.2f}s] Processing layer {layer}...")
        df1, base1, labels1 = load(args.model, layer, ds1)
        df2, base2, labels2 = load(args.model, layer, ds2)

        # Compute mean and covariance for the first dataset, using cached values if available and not forcing recalculation
        print(f"[{time() - start:.2f}s] Layer {layer}: computing moments...")
        if (base1 / "sampled_mahalanobis.pkl").exists() and not args.fforce:
            with open(base1 / "sampled_mahalanobis.pkl", "rb") as f:
                mu1, cov1 = pickle.load(f)
        else:
            mu1, cov1 = comp_mean_cov(df1, base1, labels1)

            # Save the computed mean and covariance for the first dataset to a pickle file
            with open(base1 / "sampled_mahalanobis.pkl", "wb") as f:
                pickle.dump((mu1, cov1), f)

        # Compute mean and covariance for the second dataset, using cached values if available and not forcing recalculation
        if (base2 / "sampled_mahalanobis.pkl").exists() and not args.fforce:
            print(f"[{time() - start:.2f}s] Layer {layer}: loading cached moments for {ds2}.")
            with open(base2 / "sampled_mahalanobis.pkl", "rb") as f:
                mu2, cov2 = pickle.load(f)
        else:
            mu2, cov2 = comp_mean_cov(df2, base2, labels2)

            # Save the computed mean and covariance for the second dataset to a pickle file
            with open(base2 / "sampled_mahalanobis.pkl", "wb") as f:
                pickle.dump((mu2, cov2), f)

        # Calculate the Mahalanobis distance between the two datasets using their means and covariances
        dist = mahalanobis_head(mu1, cov1, mu2, cov2)
        results.append(dist)
        print(f"[{time() - start:.2f}s] Layer {layer}: Mahalanobis distance = {dist:.6f}")

    # Save the results to the output pickle file, ensuring thread safety with a file lock
    with FileLock(str(out_pkl) + ".lock"):
        # Load existing results if the output pickle file exists, otherwise initialize an empty dictionary
        if out_pkl.exists():
            with open(out_pkl, "rb") as f:
                results_live = pickle.load(f)
        else:
            results_live = {}

        # Update the results for the current model and datasets
        if args.model not in results_live:
            results_live[args.model] = {}
        results_live[args.model][(ds1, ds2)] = results

        # Save the updated results to the output pickle file
        with open(out_pkl, "wb") as f:
            pickle.dump(results_live, f)
    print(f"[{time() - start:.2f}s] Saved results to {out_pkl}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Calculate Mahalanobis distance between Natural and Artificial protein datasets.")
    argparser.add_argument("--model", type=str, required=True, choices=MODELS, help="Model name")
    argparser.add_argument("--datasets", type=str, nargs=2, required=True, choices=["art_lys", "nat_lys", "stability", "scope"], help="Two datasets to compare")
    argparser.add_argument("--force", action="store_true", help="Force recalculation of mahalanobis distance")
    argparser.add_argument("--fforce", action="store_true", help="Force recalculation of means and covariances, triggers --force")
    args = argparser.parse_args()
    routine(args)
