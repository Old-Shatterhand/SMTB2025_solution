import pickle
from pathlib import Path
from typing import Literal

import matplotlib
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib import gridspec, pyplot as plt

from src.viz.utils import XP, compute_metric, interpolate_data
from src.viz.plot_utils import compute_performance, plot_improvement_heatmap, set_subplot_label
from src.viz.constants import DATASET_COLORS, DATATASK_NAMES, LAYERS, METRIC_TITLES, MODEL_MARKERS, MODEL_COLORS, DATASET2TASK, MODEL_NAMES, TASK_METRICS


BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def finetuned_mlm_losses():
    losses = []
    for model in ["dl_bin", "fl", "fl_bin", "ma_old", "stab"]:
        losses.append([])
        for layer in range(31):
            with open(BASE / "embeddings" / f"esm_fine_{model}" / "mlm" / f"layer_{layer}" / "predictions_knn_20.pkl", "rb") as f:
                y_hat, y = pd.read_pickle(f)[1]
            losses[-1].append(compute_metric(y_hat, y, "mlm", "multi-class"))
    return losses


def plot_new_fig1(models: list[str], algo: Literal["lr", "knn"], class_metric: str = "mcc", reg_metric: str = "pearson"):
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 2], hspace=0.3)

    gs_loss = gs[0].subgridspec(1, 2, width_ratios=[0.5, 1], wspace=0.2)
    gs_grid = gs_loss[1].subgridspec(2, 3, wspace=0.3, hspace=0.2)

    gs_perf = gs[1].subgridspec(1, 2, width_ratios=[2, 1], wspace=0.1)
    gs_curves = gs_perf[1].subgridspec(2, 1)

    ax_pt = fig.add_subplot(gs_loss[0])
    axs_ft = [
        fig.add_subplot(gs_grid[0, 0]), fig.add_subplot(gs_grid[0, 1]), fig.add_subplot(gs_grid[0, 2]),
        fig.add_subplot(gs_grid[1, 0]), fig.add_subplot(gs_grid[1, 1]), fig.add_subplot(gs_grid[1, 2])
    ]
    ax_grid = fig.add_subplot(gs_perf[0])
    ax_cls = fig.add_subplot(gs_curves[0])
    ax_reg = fig.add_subplot(gs_curves[1])

    # Plot the pretraining losses
    ax_pt.hlines(y=3.025442294717797, color="black", linestyle="--", xmin=0, xmax=1)

    for num_layers in [6, 12, 30, 33, 36]:
        losses = []
        for layer in range(num_layers + 1):
            with open(BASE / "aa_embeddings" / f"esm_t{num_layers}" / "mlm" / f"layer_{layer}" / "predictions_lr_20.pkl", "rb") as f:
                y_hat, y = pickle.load(f)[1]
            losses.append(compute_metric(np.array(y_hat), np.array(y), 'mlm', 'multi-class'))
        ax_pt.plot(np.arange(0, 1 + 1e-5, 1 / num_layers), losses, label=MODEL_NAMES[f"esm_t{num_layers}"], marker=MODEL_MARKERS[f"esm_t{num_layers}"], color=MODEL_COLORS[f"esm_t{num_layers}"])

    losses = finetuned_mlm_losses()
    # ax_pt.plot(np.arange(0, 1 + 1e-5, 1 / 30), np.min(losses, axis=0), color=MODEL_COLORS["esm_fine"], marker=MODEL_MARKERS["esm_fine"])
    # ax_pt.plot(np.arange(0, 1 + 1e-5, 1 / 30), np.max(losses, axis=0), color=MODEL_COLORS["esm_fine"], marker=MODEL_MARKERS["esm_fine"])
    ax_pt.fill_between(np.arange(0, 1 + 1e-5, 1 / 30), np.min(losses, axis=0), np.max(losses, axis=0), color=MODEL_COLORS["esm_fine"], alpha=0.3)

    ax_pt.set_xlabel("Relative layer")
    ax_pt.set_ylabel("MLM Loss (↓)")
    ax_pt.grid()

    # ntp = ax_pt.twinx()
    # ntp.hlines(y=3.025442294717797, color="black", linestyle="--", xmin=0, xmax=1)

    for size in ["small", "medium", "large"]:
        losses = []
        for layer in range(LAYERS[f"progen2_{size}"] + 1):
            with open(BASE / "aa_embeddings" / f"progen2_{size}" / "ntp" / f"layer_{layer}" / "predictions_knn_20.pkl", "rb") as f:
                y_hat, y = pd.read_pickle(f)[1]
            losses.append(compute_metric(np.array(y_hat), np.array(y), 'mlm', 'multi-class'))
        ax_pt.plot(np.arange(0, 1 + 1e-5, 1 / LAYERS[f"progen2_{size}"]), losses, label=f"ProGen2 {size.capitalize()}", marker=MODEL_MARKERS[f"progen2_{size}"], color=MODEL_COLORS[f"progen2_{size}"])
    ax_pt.set_ylabel("MLM & NTP Loss (↓)")
    set_subplot_label(ax_pt, fig, "A")

    # Plot the finetuning curves
    for i, dataset in enumerate(["stability", "fluorescence_classification", "deeploc2_bin", "meltome_atlas", "fluorescence", "deeploc2"]):
        task = DATASET2TASK[dataset]
        fine = [compute_performance(BASE, "esm_fine", dataset, layer, algo="knn", metric=TASK_METRICS[task], aa=False, n_classes=-1, task=task) for layer in range(31)]
        orig = [compute_performance(BASE, "esm_t30", dataset, layer, algo="knn", metric=TASK_METRICS[task], aa=False, n_classes=-1, task=task) for layer in range(31)]
        axs_ft[i].plot(np.arange(0, 1 + 1e-5, 1 / 30), fine, label="fine-tuned ESM-2 150M", marker=MODEL_MARKERS["esm_fine"], color=MODEL_COLORS["esm_fine"])
        axs_ft[i].plot(np.arange(0, 1 + 1e-5, 1 / 30), orig, label="original ESM-2 150M", marker=MODEL_MARKERS["esm_t30"], color=MODEL_COLORS["esm_t30"])

        axs_ft[i].set_title(DATATASK_NAMES[dataset].replace("Temperature", "Temp."))
        axs_ft[i].set_xlabel("Relative layer")
        axs_ft[i].set_ylabel(METRIC_TITLES[TASK_METRICS[task]])
    
    for i in range(6):
        axs_ft[i].grid()
        if i < 3:
            axs_ft[i].tick_params(axis='x', labelbottom=False, bottom=False)
            axs_ft[i].set_xlabel("")
        set_subplot_label(axs_ft[i], fig, chr(ord("B") + i))

    handles, labels = ax_pt.get_legend_handles_labels()
    fine_handles, fine_labels = axs_ft[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='black', linestyle="--"))
    handles.insert(5, fine_handles[0])
    labels.append("Random Baseline")
    labels.insert(5, fine_labels[0])

    # Plot the heatmap
    with open(f"data_{algo}_{class_metric}_{reg_metric}.pkl", "rb") as f:
        metrics = pickle.load(f)

    df = pd.DataFrame(index=metrics.keys(), columns=metrics["esm_t6"].keys())
    for model in metrics.keys():
        for dataset in metrics[model].keys():
            if metrics[model][dataset][-1] == 0:
                df.loc[model, dataset] = np.nan
            else:
                df.loc[model, dataset] = max(metrics[model][dataset]) / metrics[model][dataset][-1]

    # matplotlib.rc('font', **{'size': 11})
    plot_improvement_heatmap(ax_grid, df, models, [
        "fluorescence_classification", "fluorescence", "gb1", 
        "meltome_atlas_species", "meltome_atlas", "stability", "tsuboyama", 
        "solubility", "deeploc2_bin", "deeploc2", 
        "scope_40_208_fold", "scope_40_208_superfamily", "scope_40_208_3ssp", "scope_40_208_8ssp", 
        "binding"])
    # matplotlib.rc('font', **{'size': 13})

    # Plot the performance curves
    class_data = interpolate_data([[metrics[model][dataset] for model in models] for dataset in ["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]])
    reg_data = interpolate_data([[metrics[model][dataset] for model in models] for dataset in ["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]])

    for d, dataset in enumerate(["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]):
        mean = np.nanmean(class_data[d], axis=0)
        ax_cls.plot(XP, mean, label=DATATASK_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])
        # axs[1].fill_between(XP, np.nanmin(class_data[d], axis=0), np.nanmax(class_data[d], axis=0), alpha=0.2)

    for d, dataset in enumerate(["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]):
        mean = np.nanmean(reg_data[d], axis=0)
        ax_reg.plot(XP, mean, label=DATATASK_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])
        # axs[2].fill_between(XP, np.nanmin(reg_data[d], axis=0), np.nanmax(reg_data[d], axis=0), alpha=0.2)

    set_subplot_label(ax_grid, fig, "H")
    for i, ax in enumerate([ax_cls, ax_reg]):
        ax.grid()
        ax.legend()
        ax.set_xlabel("Relative layer")
        ax.set_ylabel(METRIC_TITLES[reg_metric if i == 1 else class_metric])
        set_subplot_label(ax, fig, chr(ord("I") + i))

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.585), bbox_transform=fig.transFigure, ncol=5)
    plt.tight_layout(rect=[0, 0.085, 1, 1])
    plt.savefig("paper_figures/fig_1_new.pdf", dpi=300, bbox_inches="tight")


def plot_heatmap(models: list[str], algo: Literal["lr", "knn"], class_metric: str = "mcc", reg_metric: str = "pearson"):
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

    class_data = interpolate_data([[metrics[model][dataset] for model in models] for dataset in ["solubility", "scope_40_208_fold", "deeploc2", "scope_40_208_3ssp", "binding"]])
    reg_data = interpolate_data([[metrics[model][dataset] for model in models] for dataset in ["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]])

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
    plt.savefig(f"paper_figures/supp_fig_1_new_{algo}_{class_metric}_{reg_metric}.pdf", dpi=300, bbox_inches="tight")
