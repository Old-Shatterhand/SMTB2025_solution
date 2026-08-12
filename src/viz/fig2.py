from pathlib import Path
import pickle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from src.viz.constants import DATASET_COLORS, DATASET_NAMES
from src.viz.plot_utils import plot_metric, set_subplot_label, plot_scope_minx_metric
from src.viz.utils import interpolate_data, XP

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def plot_fig2_grouped(models):
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]

    for m, metric in enumerate(["ids", "var@10", "noverlap"]):
        with open(f"data_{metric}_mcc_pearson.pkl", "rb") as f:
            data = pickle.load(f)
        for dataset in ["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama", "solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]:
            if metric == "noverlap":
                d_data = interpolate_data([list(filter(lambda x: sum(x) > 2, [data[model][dataset][:-1] for model in models]))])[0]
            else:
                d_data = interpolate_data([list(filter(lambda x: sum(x) > 2, [data[model][dataset] for model in models]))])[0]
            mean = np.nanmean(d_data, axis=0)
            axs[m].plot(XP, mean, label=DATASET_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])
            # axs[m].fill_between(XP, np.nanmin(d_data, axis=0), np.nanmax(d_data, axis=0), alpha=0.2)
        axs[m].grid()
        axs[m].set_ylabel({"ids": "2-NN Intrindic Dimension", "var@10": "Variance@10", "noverlap": r"Neighborhood Overlap $\chi_{10}^{l,l+1}$"}[metric])
        axs[m].set_xlabel("Relative layer")
        set_subplot_label(axs[m], fig, label=f"{chr(ord('A') + m)}")
    axs[0].set_ylim(0, 25)

    handles, labels = axs[0].get_legend_handles_labels()
    # handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    # labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure, ncol=5)  # -0.08

    plt.tight_layout(rect=[0, 0.085, 1, 1])
    plt.savefig("paper_figures/fig_2_layer_metrics.pdf", dpi=300, bbox_inches="tight")


def plot_fig2(models):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    axs = []
    for i in range(3):
        axs.append([])
        for j in range(4):
            axs[-1].append(fig.add_subplot(gs[i, j]))
            set_subplot_label(axs[-1][-1], fig, label=f"{chr(ord('A') + i * 4 + j)}")

    for m, metric in enumerate(["ids", "noverlap", "var@10"]):
        for d, dataset in enumerate(["fluorescence", "stability", "deeploc2"]):
            plot_metric(axs[m][d], BASE, dataset, model_prefix="", metric=metric, models=models, relative=True, grouped=True)
            if m != 2:
                axs[m][3].set_xlabel("")
        plot_scope_minx_metric(axs[m][3], BASE, metric, "fold", model_prefix="", models=models, relative=True, grouped=True)

    for d, dataset in enumerate(["Fluorescence", "Stability", "DeepLoc2", "SCOPe40 2.08 Protein Level"]):
        axs[0][d].set_title(dataset)

    for m, metric in enumerate(["2NN ID", "Neighborhood Overlap", "Variance @ 10"]):
        for d in range(4):
            axs[m][d].set_ylabel(metric if d == 0 else "")
            axs[m][d].grid()
            if m < 2:
                axs[m][d].set_xlabel("")
                axs[m][d].tick_params(axis='x', labelbottom=False, bottom=False)
            else:
                axs[m][d].set_xlabel("Relative layer")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig("paper_figures/fig_2_layer_metrics.pdf", dpi=300, bbox_inches="tight")


def plot_full_fig2_top(models):
    n_rows, n_cols = 5, 3
    fig = plt.figure(figsize=(20, 25))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    for i in range(n_rows):
        axs.append([])
        for j in range(n_cols):
            axs[-1].append(fig.add_subplot(gs[i, j]))
            set_subplot_label(axs[-1][-1], fig, label=f"{chr(ord('A') + i)} {j + 1}")
            axs[-1][-1].grid()

    METRIC_NAMES = ["2NN Intrindic Dimension", "Variance@10", r"Neighborhood Overlap $\chi_{10}^{l,l+1}$"]
    DS_NAME_MAP = {
        "fluorescence": "Fluorescence",
        "stability": "Rocklin Stability",
        "tsuboyama": "Tsuboyama Stability",
        "solubility": "DeepSol",
        "deeploc2": "DeepLoc2.0",
        "meltome_atlas": "Meltome Atlas",
        "gb1": "GB1",
    }
    for m, metric in enumerate(["ids", "var@10", "noverlap"]):
        with open(f"data_{metric}_mcc_pearson.pkl", "rb") as f:
            data = pickle.load(f)
        axs[0][m].set_title(METRIC_NAMES[m])
        axs[-1][m].set_xlabel("Relative layer")
        for d, dataset in enumerate(["fluorescence", "gb1", "tsuboyama", "stability", "meltome_atlas"]):
            plot_metric(axs[d][m], BASE, dataset, model_prefix="", metric=metric, models=models, relative=True, data=data)
            axs[d][m].tick_params(axis='x', labelbottom=False, bottom=False)
            if m == 0:
                axs[d][m].set_ylabel(DATASET_NAMES[dataset])

    axs[0][0].set_ylim(-0.05, 30.05)
    axs[1][0].set_ylim(-0.05, 25.05)
    axs[2][0].set_ylim(-0.05, 20.05)
    axs[3][0].set_ylim(-0.05, 15.05)
    for ax in axs:
        ax[1].set_ylim(-0.05, 1.05)
        ax[2].set_ylim(-0.05, 1.05)

    handles, labels = axs[0][0].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.05), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig("paper_figures/full_2_layer_metrics_top.pdf", dpi=300, bbox_inches="tight")


def plot_full_fig2_bot(models):
    n_rows, n_cols = 5, 3
    fig = plt.figure(figsize=(20, 25))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    for i in range(n_rows):
        axs.append([])
        for j in range(n_cols):
            axs[-1].append(fig.add_subplot(gs[i, j]))
            set_subplot_label(axs[-1][-1], fig, label=f"{chr(ord('A') + i + 5)} {j + 1}")
            axs[-1][-1].grid()

    METRIC_NAMES = ["2NN Intrindic Dimension", "Variance@10", r"Neighborhood Overlap $\chi_{10}^{l,l+1}$"]
    DS_NAME_MAP = {
        "fluorescence": "Fluorescence",
        "stability": "Rocklin Stability",
        "tsuboyama": "Tsuboyama Stability",
        "solubility": "DeepSol",
        "deeploc2": "DeepLoc2.0",
        "meltome_atlas": "Meltome Atlas",
        "gb1": "GB1",
    }
    for m, metric in enumerate(["ids", "var@10", "noverlap"]):
        with open(f"data_{metric}_mcc_pearson.pkl", "rb") as f:
            data = pickle.load(f)
        axs[0][m].set_title(METRIC_NAMES[m])
        for d, dataset in enumerate(["solubility", "deeploc2"]):
            plot_metric(axs[d][m], BASE, dataset, model_prefix="", metric=metric, models=models, relative=True, data=data)
            axs[d][m].tick_params(axis='x', labelbottom=False, bottom=False)
            if m == 0:
                axs[d][m].set_ylabel(DS_NAME_MAP[dataset])

        plot_scope_minx_metric(axs[-3][m], BASE, metric, "fold", model_prefix="", models=models, relative=True, data=data)
        axs[-3][m].tick_params(axis='x', labelbottom=False, bottom=False)
        plot_metric(axs[-2][m], BASE, "scope_40_208", metric=metric, relative=True, aa=True, n_classes=3, models=models, data=data)
        axs[-2][m].tick_params(axis='x', labelbottom=False, bottom=False)
        plot_metric(axs[-1][m], BASE, "binding", metric=metric, relative=True, aa=True, n_classes=2, models=models, data=data)
        axs[-1][m].set_xlabel("Relative layer")

        if m == 0:
            axs[-3][m].set_ylabel("SCOPe40 2.08 protein level")
            axs[-2][m].set_ylabel("SCOPe40 2.08 residue level")
            axs[-1][m].set_ylabel("Binding")

    for ax in axs:
        ax[1].set_ylim(-0.05, 1.05)
        ax[2].set_ylim(-0.05, 1.05)

    handles, labels = axs[0][0].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.05), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig("paper_figures/full_2_layer_metrics_bot.pdf", dpi=300, bbox_inches="tight")
