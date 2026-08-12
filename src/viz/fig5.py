import pickle
from pathlib import Path

import matplotlib
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import gridspec, pyplot as plt
from sklearn.metrics import auc
import umap
from scipy.stats import mannwhitneyu
from seaborn import stripplot

from src.viz.plot_utils import set_subplot_label
from src.viz.constants import DATASET_COLORS, DATASET_NAMES, LAYERS, METRIC_TITLES
from src.viz.utils import XP, interpolate_data, minmax_normalize_list

matplotlib.rc('font', **{'size': 11})

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

ds_to_col = {
    "fluorescence": "royalblue",
    "gb1_sampled": "red",
    "gb1": "red",
    "meltome_atlas": "purple",
    "stability": "darkorange",
    "solubility": "green",
    "deeploc2": "red",
    "scope_40_208": "purple",
    "scope_40_208_fold": "purple",
    "scope_40_208_3ssp": "violet",
    "binding": "thistle",
    "lysosomes_natural": "black",
    "lysosomes": "black",
    "tsuboyama": "lightgreen",
}

ds_to_marker = {
    "fluorescence": "o",
    "meltome_atlas": "o",
    "stability": "o",
    "solubility": "x",
    "deeploc2": "x",
    "scope_40_208": "x",
    "scope_40_208_fold": "x",
    "scope_40_208_3ssp": "x",
    "binding": "x",
    "gb1_sampled": "o",
    "gb1": "o",
    "lysosomes_natural": "x",
    "lysosomes": "^",
    "tsuboyama": "o",
}

sampling_ratios = {
    "deeploc2": 0.07,
    "stability": 0.03,
    "meltome_atlas": 0.675,
    "fluorescence": 0.035,
    "scope_40_208": 0.135,
    "solubility": 0.03,
    "gb1_sampled": 0.25,
    "lysosomes_natural": 1,
    "lysosomes": 1,
    "tsuboyama": 1,
}

def plot_fig5_space():
    plt.figure(figsize=(18, 6))

    ds_names = []
    embeddings = []
    marker = []

    stab = pd.read_csv(BASE / "datasets" / "stability_random.csv")
    art_map = dict(stab[["ID", "artificial"]].values)

    for dataset in ["deeploc2", "stability", "meltome_atlas", "fluorescence", "scope_40_208", "solubility", "gb1_sampled", "tsuboyama", "lysosomes_natural", "lysosomes"]:
        ds_count = 0
        print(f"Loading embeddings for {dataset}...")
        for filepath in (BASE / "embeddings" / "esmc_600m" / dataset / "layer_36").glob("P*.pkl"):
            if np.random.rand() > sampling_ratios[dataset]:
                continue
            ds_count += 1
            print(f"\rLoading {filepath}...", end="")
            with(open(filepath, "rb")) as f:
                embeddings.append(pickle.load(f))
                ds_names.append(dataset)
            if dataset == "stability" and art_map[filepath.stem] == True:
                marker.append("^")
            else:
                marker.append(ds_to_marker[dataset])
        print(f"\rLoaded {filepath}...")
        print("Loaded", ds_count, "embeddings for", dataset)

    embs = np.stack(embeddings)
    shuffle = np.random.permutation(len(embs))
    embs = embs[shuffle]
    ds = np.array(ds_names)[shuffle]
    marker = np.array(marker)[shuffle]

    compressor = PCA(n_components=50, random_state=42)
    X_tsne_full = compressor.fit_transform(embs)

    with open("paper_figures/fig_5_embed_big.pkl", "wb") as f:
        pickle.dump((X_tsne_full, ds, [DATASET_COLORS[d] for d in ds], marker), f)

    scatter_mixed(X_tsne_full[:, 0], X_tsne_full[:, 1], colors=[DATASET_COLORS[d] for d in ds], markers=marker, s=8, alpha=0.5)
    # mask = [x == "meltome_atlas" for x in ds]
    # mscatter(X_tsne_full[:, 0], X_tsne_full[:, 1], None, color=[DATASET_COLORS[d] for d in ds], s=8, alpha=0.5, m=marker)
    # for i, d in enumerate(ds):
    #     print("\rPlotting", i, "/", len(ds), end="")
    #     plt.plot(X_tsne_full[:, 0], X_tsne_full[:, 1], color=DATASET_COLORS[d], marker=marker[i], markersize=5, alpha=0.5, linestyle="None")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.tick_params(axis='x', labelbottom=False, bottom=False)
    plt.tick_params(axis='y', labelleft=False, left=False)
    
    plt.tight_layout()
    plt.savefig(f"paper_figures/fig_5_embed.pdf", bbox_inches="tight")  # , background="transparent")


def scatter_mixed(x, y, colors, markers, ax=None, size=20, **kwargs):
    """
    Scatter plot with per-point colors and markers, kept in original order
    (no group is drawn entirely on top of another) while producing a
    compact output (one collection per distinct marker, not per point).

    Parameters
    ----------
    x, y : array-like
        Point coordinates.
    colors : list
        Per-point colors (any matplotlib-accepted color spec).
    markers : list
        Per-point marker strings (e.g. 'o', 's', '^').
    ax : matplotlib Axes, optional
        Axes to draw on. A new one is created if omitted.
    size : float or array-like
        Marker size(s).
    **kwargs :
        Passed through to ax.scatter (e.g. alpha, edgecolors, linewidths).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    colors = np.asarray(colors, dtype=object)
    markers = np.asarray(markers, dtype=object)
    sizes = np.broadcast_to(np.asarray(size), x.shape)

    if ax is None:
        _, ax = plt.subplots()

    idx = np.arange(len(x))

    for m in dict.fromkeys(markers):          # preserve first-seen marker order
        sel = markers == m
        # zorder = mean original index, so marker groups interleave in
        # roughly their original ordering instead of one hiding another.
        z = idx[sel].mean()
        ax.scatter(
            x[sel], y[sel],
            c=list(colors[sel]),
            marker=m,
            zorder=z,
            **kwargs,
        )
    return ax