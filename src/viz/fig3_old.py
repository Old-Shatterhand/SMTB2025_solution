from pathlib import Path

import numpy as np
from matplotlib import gridspec, pyplot as plt

from src.viz.plot_utils import plot_performance, set_subplot_label
from src.viz.constants import DATASET_NAMES, METRIC_TITLES, MODEL_NAMES, DATASET2TASK, TASK_METRICS


def correlate_perfs(perfs1: dict, perfs2: dict):
    avg = []
    for key in perfs1:
        corr = np.corrcoef(perfs1[key], perfs2[key])[0, 1]
        print(f"{key}: {corr:.4f}")
        if not np.isnan(corr):
            avg.append(corr)
    print(f"Average correlation: {np.mean(avg):.4f} ± {np.std(avg):.4f}")


def plot_fig3_old(models):
    ROOT = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    fcls_perfs = plot_performance(axs[0], ROOT, "fluorescence_classification", "knn", "mcc", relative=True, aa=False, n_classes=1, task="binary", models=models)
    freg_perfs = plot_performance(axs[3], ROOT, "fluorescence", "knn", "r2", relative=True, aa=False, n_classes=1, task="regression", models=models)
    axs[0].set_title("Fluorescence", fontsize=18, pad=30)
    axs[0].text(0.5, 1.02, "Binary", transform=axs[0].transAxes, ha='center', va='bottom')
    axs[0].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["fluorescence_classification"]]])
    axs[3].text(0.5, 1.02, "Regression", transform=axs[3].transAxes, ha='center', va='bottom')
    axs[3].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["fluorescence"]]])
    set_subplot_label(axs[0], fig, "A")
    set_subplot_label(axs[3], fig, "B")

    dbin_perfs = plot_performance(axs[1], ROOT, "deeploc2_bin", "knn", "mcc", relative=True, aa=False, n_classes=1, task="binary", models=models)
    d10_perfs = plot_performance(axs[4], ROOT, "deeploc2", "knn", "mcc", relative=True, aa=False, n_classes=1, task="multi-label", models=models)
    axs[1].set_title("DeepLoc2.0", fontsize=18, pad=30)
    axs[1].text(0.5, 1.02, "Binary", transform=axs[1].transAxes, ha='center', va='bottom')
    axs[1].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["deeploc2_bin"]]])
    axs[4].text(0.5, 1.02, "10-class", transform=axs[4].transAxes, ha='center', va='bottom')
    axs[4].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["deeploc2"]]])
    set_subplot_label(axs[1], fig, "C")
    set_subplot_label(axs[4], fig, "D")

    ssp3_perfs = plot_performance(axs[2], ROOT, "scope_40_208", "knn", "mcc", relative=True, aa=True, n_classes=3, task="multi-class", models=models)
    ssp8_perfs = plot_performance(axs[5], ROOT, "scope_40_208", "knn", "mcc", relative=True, aa=True, n_classes=8, task="multi-class", models=models)
    axs[2].set_title("SCOPe40 2.08 SSP", fontsize=18, pad=30)
    axs[2].text(0.5, 1.02, "3-class", transform=axs[2].transAxes, ha='center', va='bottom')
    axs[2].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["scope_40_208_3ssp"]]])
    axs[5].text(0.5, 1.02, "8-class", transform=axs[5].transAxes, ha='center', va='bottom')
    axs[5].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["scope_40_208_8ssp"]]])
    set_subplot_label(axs[2], fig, "E")
    set_subplot_label(axs[5], fig, "F")

    for i in range(6):
        axs[i].grid()
        if i < 3:
            axs[i].tick_params(axis='x', labelbottom=False, bottom=False)
            axs[i].set_xlabel("")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

    plt.tight_layout()
    plt.savefig("paper_figures/3_old_dataset_variants.pdf", dpi=300, bbox_inches='tight')
    
    print("Fluorescence:")
    correlate_perfs(freg_perfs, fcls_perfs)

    print("\nDeepLoc:")
    correlate_perfs(d10_perfs, dbin_perfs)

    print("\nSCOPe40 2.08 SSP:")
    correlate_perfs(ssp3_perfs, ssp8_perfs)
