from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import gridspec

from src.viz.constants import DATASET_NAMES
from src.viz.plot_utils import plot_metric, set_subplot_label, plot_scope_minx_metric

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


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
            plot_metric(axs[m][d], BASE, dataset, model_prefix="", metric=metric, models=models, relative=True)
            if m != 2:
                axs[m][3].set_xlabel("")
        plot_scope_minx_metric(axs[m][3], BASE, metric, "fold", model_prefix="", models=models, relative=True)

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
    plt.savefig("paper_figures/2_layer_metrics.pdf", dpi=300, bbox_inches="tight")


def plot_full_fig2():
    models = ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh_base", "ankh_large", "prott5", "prostt5", "progen2_small", "progen2_medium", "progen2_large", "protgpt2"]
    n_rows, n_cols = 9, 4
    fig = plt.figure(figsize=(25, 40))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    for i in range(n_rows):
        axs.append([])
        for j in range(n_cols):
            axs[-1].append(fig.add_subplot(gs[i, j]))
            set_subplot_label(axs[-1][-1], fig, label=f"{chr(ord('A') + i * n_cols + j)}")
            axs[-1][-1].grid()

    METRIC_NAMES = ["2NN ID", "Neighborhood Overlap", "Variance @ 10", "5D Volume"]
    DS_NAME_MAP = {
        "fluorescence": "Fluorescence",
        "stability": "Stability",
        "solubility": "DeepSol",
        "deeploc2": "DeepLoc2.0",
        "meltome_atlas": "Meltome Atlas",
        "gb1": "GB1",
    }
    for m, metric in enumerate(["ids", "noverlap", "var@10", "5dvol"]):
        axs[0][m].set_title(METRIC_NAMES[m])
        for d, dataset in enumerate(["fluorescence", "gb1", "meltome_atlas", "stability", "solubility", "deeploc2"]):
            plot_metric(axs[d][m], BASE, dataset, model_prefix="", metric=metric, models=models, relative=True)
            axs[d][m].tick_params(axis='x', labelbottom=False, bottom=False)
            if m == 0:
                axs[d][m].set_ylabel(DS_NAME_MAP[dataset])

        plot_scope_minx_metric(axs[-3][m], BASE, metric, "fold", model_prefix="", models=models, relative=True)
        axs[-3][m].tick_params(axis='x', labelbottom=False, bottom=False)
        plot_metric(axs[-2][m], BASE, "scope_40_208", metric=metric, relative=True, aa=True, n_classes=3)
        axs[-2][m].tick_params(axis='x', labelbottom=False, bottom=False)
        plot_metric(axs[-1][m], BASE, "binding", metric=metric, relative=True, aa=True, n_classes=2)
        axs[-1][m].set_xlabel("Relative Layer")

        if m == 0:
            axs[-3][m].set_ylabel("SCOPe40 2.08 protein level")
            axs[-2][m].set_ylabel("SCOPe40 2.08 residue level")
            axs[-1][m].set_ylabel("Binding")

    handles, labels = axs[0][0].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig("paper_figures/full_2_layer_metrics.pdf", dpi=300, bbox_inches="tight")
