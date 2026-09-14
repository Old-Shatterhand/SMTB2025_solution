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
from src.viz.constants import DATASET_COLORS, DATATASK_NAMES, LAYERS, METRIC_TITLES
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
    "stability": "x",
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

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


import matplotlib.pyplot as plt
import numpy as np

def scatter_with_markers(x, y, colors, markers, ax=None, s=50, **kwargs):
    """
    Scatter plot with a distinct color and marker per point.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the points.
    colors : list
        One color per point (any matplotlib-recognized color format).
    markers : list
        One marker per point (e.g. 'o', 's', '^', ...).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    s : scalar or array-like
        Marker size(s), passed through to scatter.
    **kwargs :
        Any other keyword args passed to ax.scatter (alpha, edgecolors, etc.)

    Returns
    -------
    list of PathCollection objects (one per unique marker group)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    colors = np.asarray(colors, dtype=object)
    markers = np.asarray(markers, dtype=object)

    if not (len(x) == len(y) == len(colors) == len(markers)):
        raise ValueError("x, y, colors, and markers must all have the same length")

    if ax is None:
        ax = plt.gca()

    # size can be scalar or per-point array
    s_arr = np.full(len(x), s) if np.isscalar(s) else np.asarray(s)

    collections = []
    # Group points by marker since scatter() only takes one marker at a time
    for marker in np.unique(markers):
        mask = markers == marker
        pc = ax.scatter(
            x[mask], y[mask],
            c=colors[mask],
            marker=marker,
            s=s_arr[mask],
            **kwargs
        )
        collections.append(pc)

    return collections

def plot_fig5(models, algo, class_metric, reg_metric):
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    axs = [
        fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ]

    ds_names = []
    embeddings = []
    marker = []

    stab = pd.read_csv(BASE / "datasets" / "stability_random.csv")
    art_map = dict(stab[["ID", "artificial"]].values)

    for dataset in ["deeploc2", "stability", "meltome_atlas", "fluorescence", "scope_40_208", "solubility", "gb1_sampled", "tsuboyama", "lysosomes_natural", "lysosomes"]:
        print(f"Loading embeddings for {dataset}...")
        for filepath in (BASE / "embeddings" / "esmc_600m" / dataset / "layer_36").glob("P*.pkl"):
            if np.random.rand() > sampling_ratios[dataset] * 0.3:
                continue
            print(f"\rLoading {filepath}...", end="")
            with(open(filepath, "rb")) as f:
                embeddings.append(pickle.load(f))
                ds_names.append(dataset)
            if dataset == "stability" and art_map[filepath.stem] == True:
                marker.append("^")
            else:
                marker.append(ds_to_marker[dataset])
        print(f"\rLoaded {filepath}...")

    embs = np.stack(embeddings)
    shuffle = np.random.permutation(len(embs))
    embs = embs[shuffle]
    ds = np.array(ds_names)[shuffle]
    marker = np.array(marker)[shuffle]

    compressor = PCA(n_components=50, random_state=42)
    X_tsne_full = compressor.fit_transform(embs)

    mscatter(X_tsne_full[:, 0], X_tsne_full[:, 1], axs[0], c=[DATASET_COLORS[d] for d in ds], s=5, alpha=0.3, m=marker)
    # axs[0].scatter(X_tsne_full[:, 0], X_tsne_full[:, 1], color=[DATASET_COLORS[d] for d in ds], s=5, alpha=0.3)
    # scatter_with_markers(X_tsne_full[:, 0], X_tsne_full[:, 1], colors=[DATASET_COLORS[d] for d in ds], markers=[ds_to_marker[d] for d in ds], ax=axs[0], s=5, alpha=0.3)
    axs[0].set_xlabel("principal component 1")
    axs[0].set_ylabel("principal component 2")
    axs[0].tick_params(axis='x', labelbottom=False, bottom=False)
    axs[0].tick_params(axis='y', labelleft=False, left=False)
    set_subplot_label(axs[0], fig, "A")

    with open(f"data_{algo}_{class_metric}_{reg_metric}.pkl", "rb") as f:
        metrics = pickle.load(f)

    datasets = ["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding", "fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]

    data = [[metrics[model][dataset] for model in models] for dataset in datasets]

    aucs = {}
    for d, dataset in enumerate(datasets):
        aucs[dataset] = []
        for m, model in enumerate(models):
            tmp = auc(np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])), data[d][m])
            if tmp == tmp:
                aucs[dataset].append(tmp)
    aucs = [np.array(aucs[ds]) for ds in datasets]

    stripplot(aucs, ax=axs[1], palette=[DATASET_COLORS[dataset] for dataset in datasets])  # , showmeans=True)
    axs[1].set_xticks(np.arange(0, 10))
    axs[1].set_xticklabels(["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding", "fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"], rotation=45, ha="right")
    axs[1].set_ylabel("AUC")
    set_subplot_label(axs[1], fig, "B")

    # print("Mann-Whitney U test results:")
    # wp = [np.mean([np.argmax(x) for x in ds]) for ds in data[:5]]
    # dms = [np.mean([np.argmax(x) for x in ds]) for ds in data[5:]]
    # print(mannwhitneyu(wp, dms))
    # print(mannwhitneyu(wp, dms, alternative="less"))
    # wp = [np.argmax(x) for ds in data[:5] for x in ds]
    # dms = [np.argmax(x) for ds in data[5:] for x in ds]
    # print(mannwhitneyu(wp, dms))
    # print(mannwhitneyu(wp, dms, alternative="less"))
    # wp = [np.mean([np.argmax(x) for x in ds]) / np.mean([len(x) for x in ds]) for ds in data[:5]]
    # dms = [np.mean([np.argmax(x) for x in ds]) / np.mean([len(x) for x in ds]) for ds in data[5:]]
    # print(mannwhitneyu(wp, dms))
    # print(mannwhitneyu(wp, dms, alternative="less"))
    # wp = [np.argmax(x) / len(x) for ds in data[:5] for x in ds]
    # dms = [np.argmax(x) / len(x) for ds in data[5:] for x in ds]
    # print(mannwhitneyu(wp, dms))
    # print(mannwhitneyu(wp, dms, alternative="less"))

    plt.tight_layout()
    plt.savefig(f"paper_figures/fig_5_{algo}_{class_metric}_{reg_metric}.pdf", bbox_inches="tight")


def plot_fig5_space():
    plt.figure(figsize=(15, 8))

    ds_names = []
    embeddings = []
    marker = []

    stab = pd.read_csv(BASE / "datasets" / "stability_random.csv")
    art_map = dict(stab[["ID", "artificial"]].values)

    for dataset in ["deeploc2", "stability", "meltome_atlas", "fluorescence", "scope_40_208", "solubility", "gb1_sampled", "tsuboyama", "lysosomes_natural", "lysosomes"]:
        ds_count = 0
        print(f"Loading embeddings for {dataset}...")
        for filepath in (BASE / "embeddings" / "esmc_600m" / dataset / "layer_36").glob("P*.pkl"):
            if np.random.rand() > sampling_ratios[dataset] * 0.15:
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