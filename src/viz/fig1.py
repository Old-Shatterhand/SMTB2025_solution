from pathlib import Path
import pickle
from typing import Literal

from matplotlib import gridspec, pyplot as plt, lines as mlines
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import auc

from src.viz.constants import DATASET2TASK, DATASET_COLORS, DATATASK_NAMES, METRIC_TITLES, MODEL_NAMES, TASK_METRICS, kill_axis
from src.viz.plot_utils import plot_improvement_heatmap, plot_performance, plot_scope_minx_performance, set_subplot_label
from src.viz.utils import interpolate_data, XP

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def scale_data(data):
    return data  # normalize(data, axis=2)


def plot_fig1(models: list[str], algo: Literal["lr", "knn"], class_metric: str = "mcc", reg_metric: str = "pearson"):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    gs_lower = gs[1].subgridspec(2, 1)
    axs = [
        fig.add_subplot(gs[0]), fig.add_subplot(gs_lower[0]), fig.add_subplot(gs_lower[1])
    ]

    with open(f"data_{algo}_{class_metric}_{reg_metric}.pkl", "rb") as f:
        metrics = pickle.load(f)

    df = pd.DataFrame(index=metrics.keys(), columns=metrics["esm_t6"].keys())
    for model in metrics.keys():
        for dataset in metrics[model].keys():
            if metrics[model][dataset][-1] == 0:
                df.loc[model, dataset] = np.nan
            else:
                df.loc[model, dataset] = max(metrics[model][dataset]) / metrics[model][dataset][-1]

    plot_improvement_heatmap(axs[0], df, models, [
        "fluorescence_classification", "fluorescence", "gb1", 
        "meltome_atlas_species", "meltome_atlas", "stability", "tsuboyama", 
        "solubility", "deeploc2_bin", "deeploc2", 
        "scope_40_208_fold", "scope_40_208_superfamily", "scope_40_208_3ssp", "scope_40_208_8ssp", 
        "binding"])

    class_data = scale_data(interpolate_data([[metrics[model][dataset] for model in models] for dataset in ["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]]))
    reg_data = scale_data(interpolate_data([[metrics[model][dataset] for model in models] for dataset in ["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]]))

    for d, dataset in enumerate(["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]):
        mean = np.nanmean(class_data[d], axis=0)
        # print(dataset, ":", auc(XP, mean))
        axs[1].plot(XP, mean, label=DATATASK_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])
        # axs[1].fill_between(XP, np.nanmin(class_data[d], axis=0), np.nanmax(class_data[d], axis=0), alpha=0.2)

    for d, dataset in enumerate(["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]):
        mean = np.nanmean(reg_data[d], axis=0)
        # print(dataset, ":", auc(XP, mean))
        axs[2].plot(XP, mean, label=DATATASK_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])
        # axs[2].fill_between(XP, np.nanmin(reg_data[d], axis=0), np.nanmax(reg_data[d], axis=0), alpha=0.2)

    set_subplot_label(axs[0], fig, "A")
    for i in range(1, 3):
        axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel("Relative layer")
        axs[i].set_ylabel(METRIC_TITLES[class_metric if i == 1 else reg_metric])
        set_subplot_label(axs[i], fig, chr(ord("A") + i))

    plt.tight_layout()
    plt.savefig(f"paper_figures/fig_1_{algo}_{class_metric}_{reg_metric}.pdf", dpi=300, bbox_inches="tight")


def plot_fig1_smtb(models: list[str], algo: Literal["lr", "knn"], class_metric: str = "mcc", reg_metric: str = "pearson"):
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    axs = [
        fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ]

    with open(f"data_{algo}_{class_metric}_{reg_metric}.pkl", "rb") as f:
        metrics = pickle.load(f)

    class_data = scale_data(interpolate_data([[metrics[model][dataset] for model in models] for dataset in ["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]]))
    reg_data = scale_data(interpolate_data([[metrics[model][dataset] for model in models] for dataset in ["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]]))

    for d, dataset in enumerate(["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]):
        mean = np.nanmean(class_data[d], axis=0)
        axs[0].plot(XP, mean, label=DATATASK_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])

    for d, dataset in enumerate(["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]):
        mean = np.nanmean(reg_data[d], axis=0)
        axs[1].plot(XP, mean, label=DATATASK_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])

    for i in range(2):
        axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel("Relative layer")
        axs[i].set_ylabel(METRIC_TITLES[class_metric if i == 0 else reg_metric])
        set_subplot_label(axs[i], fig, chr(ord("A") + i))

    plt.tight_layout()
    plt.savefig(f"paper_figures/smtb_1.png")


def plot_full_fig1_perfs(models: list[str], algo: Literal["knn", "lr"] = "knn", class_metric: str = "mcc", reg_metric: str = "pearson"):
    # models = ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh_base", "ankh_large", "prott5", "prostt5", "progen2_small", "progen2_medium", "progen2_large", "protgpt2"]
    # fig = plt.figure(figsize=(20, 12))
    # gs = gridspec.GridSpec(3, 5, figure=fig)
    # axs = []
    # for i in range(3):
    #     axs.append([])
    #     for j in range(5):
    #         axs[i].append(fig.add_subplot(gs[i, j]))
    
    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(5, 3, figure=fig)
    axs = []
    for i in range(3):
        axs.append([])
        for j in range(5):
            axs[i].append(fig.add_subplot(gs[j, i]))
    
    plot_performance(axs[0][0], BASE, "fluorescence_classification", algo, class_metric, task="binary", models=models, relative=True)
    plot_performance(axs[1][0], BASE, "fluorescence", algo, reg_metric, task="regression", models=models, relative=True)
    plot_performance(axs[2][0], BASE, "gb1", algo, reg_metric, task="regression", models=models, relative=True)
    
    # plot_performance(axs[0][1], BASE, "meltome_atlas_species", algo, class_metric, task="multi-class", models=models, relative=True)
    plot_performance(axs[1][1], BASE, "meltome_atlas", algo, reg_metric, task="regression", models=models, relative=True)
    plot_performance(axs[2][1], BASE, "stability", algo, reg_metric, task="regression", models=models, relative=True)

    plot_performance(axs[0][2], BASE, "deeploc2_bin", algo, class_metric, task="binary", models=models, relative=True)
    plot_performance(axs[1][2], BASE, "deeploc2", algo, class_metric, task="multi-label", models=models, relative=True)
    plot_performance(axs[2][2], BASE, "tsuboyama", algo, reg_metric, task="regression", models=models, relative=True)

    plot_scope_minx_performance(axs[0][3], BASE, algo, class_metric, "superfamily", model_prefix="", models=models, relative=True)
    plot_scope_minx_performance(axs[1][3], BASE, algo, class_metric, "fold", model_prefix="", models=models, relative=True)
    plot_performance(axs[2][3], BASE, "solubility", algo, class_metric, task="binary", models=models, relative=True)
    
    plot_performance(axs[0][4], BASE, "scope_40_208", "knn", class_metric, relative=True, aa=True, n_classes=3, task="multi-class", models=models)
    plot_performance(axs[1][4], BASE, "scope_40_208", "knn", class_metric, relative=True, aa=True, n_classes=8, task="multi-class", models=models)
    plot_performance(axs[2][4], BASE, "binding", algo, class_metric, task="binary", aa=True, n_classes=2, models=models, relative=True)

    for t, name in enumerate(["fluorescence_classification", "fluorescence", "gb1", "meltome_atlas_species", "meltome_atlas", "stability", "deeploc2_bin", "deeploc2", "tsuboyama", "scope_40_208_superfamily", "scope_40_208_fold", "solubility", "scope_40_208_3ssp", "scope_40_208_8ssp", "binding"]):
        set_subplot_label(axs[t % 3][t // 3], fig, chr(ord("A") + t))
        axs[t % 3][t // 3].set_title(DATATASK_NAMES[name])
        axs[t % 3][t // 3].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK[name]]])
        axs[t % 3][t // 3].grid()
        # if t % 3 == 2:
        if t > 11:
            axs[t % 3][t // 3].set_xlabel("Relative layer")
        else:
            axs[t % 3][t // 3].set_xlabel("")
            axs[t % 3][t // 3].tick_params(axis='x', labelbottom=False, bottom=False)

    handles, labels = axs[0][0].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

    plt.tight_layout(rect=[0, 0.085, 1, 1])
    plt.savefig(f"paper_figures/proteinbert_{algo}_{class_metric}_{reg_metric}.pdf", dpi=300, bbox_inches="tight")