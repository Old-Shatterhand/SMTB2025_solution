import pickle
from pathlib import Path

import matplotlib
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import gridspec, pyplot as plt
import umap

from src.viz.plot_utils import set_subplot_label
from src.viz.constants import DATASET_NAMES, LAYERS, METRIC_TITLES

ds_to_col = {
    "fluorescence": "darkred",
    "meltome_atlas": "darkorange",
    "stability": "royalblue",
    "solubility": "gold",
    "deeploc2": "green",
    "scope_40_208": "violet",
    "scope_40_208_fold": "violet",
    "scope_40_208_3ssp": "purple",
    "binding": "thistle",
    "gb1_sampled": "red",
    "gb1": "red",
}

matplotlib.rc('font', **{'size': 11})

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

def scale_data(data):
    scaled = {}
    for dataset in data["esm_t6"]:
        scaled[dataset] = {}
        for model in data:
            if dataset not in data[model] or any(np.isnan(data[model][dataset])):
                continue
            x = np.arange(0, 1 + 1e-5, 1 / (LAYERS[model]))
            y = data[model][dataset]
            xp = np.linspace(0, 1, 1000)
            yp = np.interp(xp, x, y)
            scaled[dataset][model] = yp
        mean = np.mean(list(scaled[dataset].values()), axis=0)
        scaled[dataset][model] = mean
        # maxi, mini = np.max(mean), np.min(mean)
        # for model in scaled[dataset]:
        #     scaled[dataset][model] = [(y - mini) / (maxi - mini) for y in scaled[dataset][model]]
    return scaled


def find_best_models(sc_data):
    best_local, best_global, best_total = 0, 0, 0
    best_local_model, best_global_model, best_total_model = -1, -1, 1

    for model in sc_data["fluorescence"].keys():
        model_perf_local = np.array(sc_data["fluorescence"][model]) + np.array(sc_data["stability"][model])
        model_perf_global = np.array(sc_data["deeploc2"][model]) + np.array(sc_data["solubility"][model]) + np.array(sc_data["scope_40_208_fold"][model])
        model_perf = model_perf_local + model_perf_global + np.array(sc_data["meltome_atlas"][model])

        local_max = model_perf_local.argmax()
        global_max = model_perf_global.argmax()
        total_max = model_perf.argmax()

        if model_perf_local[local_max] > best_local:
            best_local = model_perf_local[local_max]
            best_local_model = model, local_max
        
        if model_perf_global[global_max] > best_global:
            best_global = model_perf_global[global_max]
            best_global_model = model, global_max
        
        if model_perf[total_max] > best_total:
            best_total = model_perf[total_max]
            best_total_model = model, total_max

        print(f"Model: {model}")
        print(f"\tLocal Performance: {local_max}, {model_perf_local[local_max] / 2}")
        print(f"\tGlobal Performance: {global_max}, {model_perf_global[global_max] / 3}")
        print(f"\tTotal Performance: {total_max}, {model_perf[total_max] / 6}")
        print()

    print(f"Best Local Model: {best_local_model}")
    print(f"Best Global Model: {best_global_model}")
    print(f"Overall Best Model: {best_total_model}")


def plot_fig0():
    ds_names = []
    embeddings = []
    # sampling_ratios = {
    #     "deeploc2": 0.14,
    #     "stability": 0.06,
    #     "meltome_atlas": 0.135,
    #     "fluorescence": 0.07,
    #     "scope_40_208": 0.27,
    #     "solubility": 0.06,
    #     "gb1_sampled": 0.5,
    # }
    sampling_ratios = {
        "deeploc2": 0.07,
        "stability": 0.03,
        "meltome_atlas": 0.675,
        "fluorescence": 0.035,
        "scope_40_208": 0.135,
        "solubility": 0.03,
        "gb1_sampled": 0.25,
    }

    for dataset in ["deeploc2", "stability", "meltome_atlas", "fluorescence", "scope_40_208", "solubility", "gb1_sampled"]:
        print(f"Loading embeddings for {dataset}...")
        for filepath in (BASE / "embeddings" / "esmc_600m" / dataset / "layer_36").glob("P*.pkl"):
            if np.random.rand() > sampling_ratios[dataset]:
                continue
            print(f"\rLoading {filepath}...", end="")
            with(open(filepath, "rb")) as f:
                embeddings.append(pickle.load(f))
                ds_names.append(dataset)
        print(f"\rLoaded {filepath}...")

    embs = np.stack(embeddings)
    shuffle = np.random.permutation(len(embs))
    embs = embs[shuffle]
    ds = np.array(ds_names)[shuffle]

    # compressor = TSNE(
    #     n_components=2,
    #     perplexity=30,      # typical range: 5-50, lower for smaller datasets
    #     learning_rate='auto',
    #     init='pca',
    #     random_state=42,
    #     metric="cosine",
    # )
    compressor = PCA(n_components=50, random_state=42)
    # compressor = umap.UMAP(
    #     n_components=2,
    #     n_neighbors=50,
    #     min_dist=0.01,
    #     dens_frac=0.5,
    #     random_state=42,
    #     metric="cosine",
    #     densmap=True,
    # )
    X_tsne_full = compressor.fit_transform(embs)

    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    axs = [
        fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    ]

    axs[0].scatter(X_tsne_full[:, 0], X_tsne_full[:, 1], c=[ds_to_col[d] for d in ds], s=4, alpha=0.5, marker=".")
    axs[0].set_xlabel("principal component 1")
    axs[0].set_ylabel("principal component 2")
    axs[0].tick_params(axis='x', labelbottom=False, bottom=False)
    axs[0].tick_params(axis='y', labelleft=False, left=False)
    set_subplot_label(axs[0], fig, "A")

    with open("data_knn_mcc_pearson.pkl", "rb") as f:
        knn_metrics = pickle.load(f)
    sc_data = scale_data(knn_metrics)
    
    for dataset in ["fluorescence", "gb1", "stability", "meltome_atlas"]:
        axs[1].plot(np.linspace(0, 1, 1000), np.mean(list(sc_data[dataset].values()), axis=0), label=DATASET_NAMES[dataset], color=ds_to_col[dataset])
    axs[1].set_xlabel("Relative Layer")
    axs[1].set_ylabel(METRIC_TITLES["pearson"])
    axs[1].grid()
    set_subplot_label(axs[1], fig, "B")
    
    for dataset in ["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]:
        axs[2].plot(np.linspace(0, 1, 1000), np.mean(list(sc_data[dataset].values()), axis=0), label=DATASET_NAMES[dataset], color=ds_to_col[dataset])
    axs[2].set_xlabel("Relative Layer")
    axs[2].set_ylabel(METRIC_TITLES["mcc"])
    axs[2].grid()
    set_subplot_label(axs[2], fig, "C")

    handles, labels = axs[1].get_legend_handles_labels()
    add_handles, add_labels = axs[2].get_legend_handles_labels()
    handles.extend(add_handles)
    labels.extend(add_labels)
    handles.insert(7, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(7, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), bbox_transform=fig.transFigure, ncol=5)  # -0.08
    
    plt.tight_layout()
    plt.savefig("paper_figures/0_embeds_pca.pdf", bbox_inches="tight", dpi=300)
    print("Saved figure to paper_figures/0_embeds_pca.pdf")


if __name__ == "__main__":
    plot_fig0()
