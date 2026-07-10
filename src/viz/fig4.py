import pickle
from pathlib import Path

from matplotlib import gridspec, pyplot as plt
import numpy as np
import pandas as pd

from src.viz.utils import compute_metric
from src.viz.plot_utils import compute_performance, set_subplot_label
from src.viz.constants import DATASET_NAMES, LAYERS, METRIC_TITLES, MODEL_MARKERS, MODEL_COLORS, DATASET2TASK, TASK_METRICS


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


def plot_fig4():
    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.5, 1])
    gs_grid = gs[1].subgridspec(2, 3, wspace=0.2, hspace=0.2)

    mlm = fig.add_subplot(gs[0])
    axs = [
        fig.add_subplot(gs_grid[0, 0]), fig.add_subplot(gs_grid[0, 1]), fig.add_subplot(gs_grid[0, 2]),
        fig.add_subplot(gs_grid[1, 0]), fig.add_subplot(gs_grid[1, 1]), fig.add_subplot(gs_grid[1, 2])
    ]

    for num_layers in [6, 12, 30, 33, 36]:
        losses = []
        for layer in range(num_layers + 1):
            with open(BASE / "aa_embeddings" / f"esm_t{num_layers}" / "mlm" / f"layer_{layer}" / "predictions_lr_20.pkl", "rb") as f:
                y_hat, y = pickle.load(f)[1]
            losses.append(compute_metric(np.array(y_hat), np.array(y), 'mlm', 'multi-class'))
        mlm.plot(np.arange(0, 1 + 1e-5, 1 / num_layers), losses, label=f"ESM2 t{num_layers}", marker=MODEL_MARKERS[f"esm_t{num_layers}"], color=MODEL_COLORS[f"esm_t{num_layers}"])

    losses = finetuned_mlm_losses()
    mlm.plot(np.arange(0, 1 + 1e-5, 1 / 30), np.min(losses, axis=0), color=MODEL_COLORS["esm_fine"], marker=MODEL_MARKERS["esm_fine"])
    mlm.plot(np.arange(0, 1 + 1e-5, 1 / 30), np.max(losses, axis=0), color=MODEL_COLORS["esm_fine"], marker=MODEL_MARKERS["esm_fine"])
    mlm.fill_between(np.arange(0, 1 + 1e-5, 1 / 30), np.min(losses, axis=0), np.max(losses, axis=0), color=MODEL_COLORS["esm_fine"], alpha=0.3)

    mlm.set_xlabel("Relative Layer")
    mlm.set_ylabel("MLM Loss (↓)")
    mlm.grid()

    ntp = mlm.twinx()
    for size in ["small", "medium", "large"]:
        losses = []
        for layer in range(LAYERS[f"progen2_{size}"] + 1):
            with open(BASE / "aa_embeddings" / f"progen2_{size}" / "ntp" / f"layer_{layer}" / "predictions_knn_20.pkl", "rb") as f:
                y_hat, y = pd.read_pickle(f)[1]
            losses.append(compute_metric(np.array(y_hat), np.array(y), 'mlm', 'multi-class'))
        ntp.plot(np.arange(0, 1 + 1e-5, 1 / LAYERS[f"progen2_{size}"]), losses, label=f"ProGen2 {size.capitalize()}", marker=MODEL_MARKERS[f"progen2_{size}"], color=MODEL_COLORS[f"progen2_{size}"])
    ntp.set_ylabel("NTP Loss (↓)")
    set_subplot_label(mlm, fig, "A")

    for i, dataset in enumerate(["stability", "fluorescence_classification", "deeploc2_bin", "meltome_atlas", "fluorescence", "deeploc2"]):
        task = DATASET2TASK[dataset]
        fine = [compute_performance(BASE, "esm_fine", dataset, layer, algo="knn", metric=TASK_METRICS[task], aa=False, n_classes=-1, task=task) for layer in range(31)]
        orig = [compute_performance(BASE, "esm_t30", dataset, layer, algo="knn", metric=TASK_METRICS[task], aa=False, n_classes=-1, task=task) for layer in range(31)]
        axs[i].plot(np.arange(0, 1 + 1e-5, 1 / 30), fine, label="fine-tuned ESM-2 150M", marker=MODEL_MARKERS["esm_fine"], color=MODEL_COLORS["esm_fine"])
        axs[i].plot(np.arange(0, 1 + 1e-5, 1 / 30), orig, label="original ESM-2 150M", marker=MODEL_MARKERS["esm_t30"], color=MODEL_COLORS["esm_t30"])

        axs[i].set_title(DATASET_NAMES[dataset])
        axs[i].set_xlabel("Relative Layer")
        axs[i].set_ylabel(METRIC_TITLES[TASK_METRICS[task]])
    
    for i in range(6):
        axs[i].grid()
        if i < 3:
            axs[i].tick_params(axis='x', labelbottom=False, bottom=False)
            axs[i].set_xlabel("")
        set_subplot_label(axs[i], fig, chr(ord("B") + i))

    handles, labels = mlm.get_legend_handles_labels()
    fine_handles, fine_labels = axs[0].get_legend_handles_labels()
    ntp_handles, ntp_labels = ntp.get_legend_handles_labels()
    handles += [fine_handles[0]] + ntp_handles
    labels += [fine_labels[0]] + ntp_labels
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), bbox_transform=fig.transFigure, ncol=5)

    plt.tight_layout()
    # plt.savefig("paper_figures/4_finetuned.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper_figures/4_finetuned.png")
