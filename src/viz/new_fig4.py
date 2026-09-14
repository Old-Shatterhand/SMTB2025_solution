
import pickle

from matplotlib import gridspec, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from src.viz.constants import DATASET_COLORS, DATASET_NAMES
from src.viz.plot_utils import set_subplot_label
from src.viz.utils import XP, interpolate_data


def plot_new_fig4(models):
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1])
    gs_grid = gs[0].subgridspec(2, 3, hspace=0.25)
    axs = [fig.add_subplot(gs_grid[i, j]) for i in range(2) for j in range(3)]
    ax_sil = fig.add_subplot(gs[1])

    dataset_name_map = {
        "fluorescence": "Fluorescence",
        "gb1": "GB1",
        "stability": "Stability",
        "meltome_atlas": "Meltome Atlas",
        "tsuboyama": "Tsuboyama",
        "solubility": "Solubility",
        "scope_40_208_fold": "SCOPe40 protein level",
        "deeploc2": "DeepLoc2",
        "scope_40_208_3ssp": "SCOPe40 residue level",
        "binding": "Binding",
    }

    for m, metric in enumerate(["noverlap", "ids", "var@10"]):
        with open(f"data_{metric}_mcc_pearson.pkl", "rb") as f:
            data = pickle.load(f)
        for dataset in ["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama", "solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]:
            if metric == "noverlap":
                d_data = interpolate_data([list(filter(lambda x: sum(x) > 2, [data[model][dataset][:-1] for model in models]))])[0]
            else:
                d_data = interpolate_data([list(filter(lambda x: sum(x) > 2, [data[model][dataset] for model in models]))])[0]
            mean = np.nanmean(d_data, axis=0)
            axs[m].plot(XP, mean, label=dataset_name_map[dataset], linewidth=5, color=DATASET_COLORS[dataset])
            # axs[m].fill_between(XP, np.nanmin(d_data, axis=0), np.nanmax(d_data, axis=0), alpha=0.2)
        axs[m].grid()
        axs[m].set_ylabel({"ids": "2-NN Intrindic Dimension", "var@10": "Variance@10", "noverlap": r"Neighborhood Overlap $\chi_{10}^{l,l+1}$"}[metric])
        axs[m].set_xlabel("Relative layer")
        if metric == "ids":
            axs[m].set_ylim(0, 25)

    with open("mds_data.pkl", "rb") as f:
        X0_comp, X36_comp, seq_comp, mds_seqs_tmp, cats = pickle.load(f)

    colors = {
        "deeploc2": DATASET_COLORS['deeploc2'],
        "fluorescence": DATASET_COLORS['fluorescence'],
        "gb1_sampled": DATASET_COLORS['gb1'],
        "lysosomes": '#000000',
        "lysosomes_natural": '#777373',
        "meltome_atlas": DATASET_COLORS['meltome_atlas'],
        "scope_40_208": DATASET_COLORS['scope_40_208'],
        "solubility": DATASET_COLORS['solubility'],
        "stability_artificial": "#ffa95e",
        "stability_artificial_dms": "#fcd2ad",
        "stability_dms": '#ff7f0e',
        "tsuboyama": DATASET_COLORS['tsuboyama'],
    }

    axs[3].scatter(seq_comp[:, 0], seq_comp[:, 1], c=[colors[c[0]] for c in mds_seqs_tmp], s=5, alpha=0.7)
    axs[4].scatter(X0_comp[:, 0], X0_comp[:, 1], c=[colors[c] for c in cats], s=5, alpha=0.7)
    axs[5].scatter(X36_comp[:, 0], X36_comp[:, 1], c=[colors[c] for c in cats], s=5, alpha=0.7)

    axs[3].set_title("Sequence space")
    axs[4].set_title("Layer 0")
    axs[5].set_title("Layer 36")
    for i in range(3, 6):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel("t-SNE 1")
        axs[i].set_ylabel("t-SNE 2")

    with open("pw_silhouettes.pkl", "rb") as f:
        blab, embeddings = pickle.load(f)
    ds_names = list(embeddings["esm_t6"].keys())
    ds_names.remove("scope_40_208")

    for d, ds_name in enumerate(ds_names):
        ax_sil.plot(XP, blab.mean(axis=0)[d], label=DATASET_NAMES[ds_name], color=DATASET_COLORS["chorismate" if ds_name.startswith("chorismate") else ds_name], linewidth=5, linestyle='--' if ds_name in {"lysosomes", "chorismate"} else '-')
        # ax_sil.fill_between(XP, blab.min(axis=0)[d], blab.max(axis=0)[d], color=DATASET_COLORS[ds_name.replace("_natural", "")], alpha=0.2)
    ax_sil.grid()
    ax_sil.set_ylabel("pairwise silhouette score to SCOPe40")
    ax_sil.set_xlabel("Relative Layer")

    for i in range(6):
        set_subplot_label(axs[i], fig, label=f"{chr(ord('A') + i)}")
    set_subplot_label(ax_sil, fig, label="G")

    legend_handles = [Line2D(
            [0], [0], 
            color=color, 
            label=name, 
            linestyle='--' if name in {'Artificial lysozymes', 'Artificial chorismate mutase'} else '-'
        ) for name, color in [
        ("Fluorescence", DATASET_COLORS['fluorescence']),
        ("GB1", DATASET_COLORS['gb1']),
        ("Tsuboyama", DATASET_COLORS['tsuboyama']),
        ("DMS Rocklin", '#ff7f0e'),
        ("Artificial Rocklin", "#ffa95e"),
        ("Artificial DMS Rocklin", "#fcd2ad"),
        ("Meltome atlas", DATASET_COLORS['meltome_atlas']),
        ("DeepSol", DATASET_COLORS['solubility']),
        ("SCOPe40 protein-level", DATASET_COLORS['scope_40_208']),
        ("SCOPe40 residue-level", DATASET_COLORS['scope_40_208_3ssp']),
        ("DeepLoc2.0", DATASET_COLORS['deeploc2']),
        ("Binding", DATASET_COLORS['binding']),
        ("Natural lysozymes", DATASET_COLORS['lysosomes_natural']),
        ("Artificial lysozymes", DATASET_COLORS['lysosomes']),
        ("Natural chorismate mutase", "#BEBEBE"),
        ("Artificial chorismate mutase", '#BEBEBE')
    ]]
    # metric_handles, metric_labels = axs[0].get_legend_handles_labels()
    # space_handles, space_labels = axs[3].get_legend_handles_labels()
    # sil_handles, sil_labels = ax_sil.get_legend_handles_labels()
    # print(metric_labels, space_labels, sil_labels)

    # handles, labels = axs[0].get_legend_handles_labels()
    # # handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    # # labels.insert(5, "")
    # baseline = [Line2D([0], [0], color='black', linestyle="--")]
    # fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure, ncol=5)  # -0.08
    leg = plt.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure, ncol=4)  # -0.08
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.085, 1, 1])

    plt.savefig("paper_figures/fig_4_new.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("paper_figures/fig_4_new.png", bbox_inches="tight", dpi=300)

