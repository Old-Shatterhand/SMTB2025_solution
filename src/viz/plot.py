from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.viz.constants import DS_NAME_MAP, ROOT, REG_METRIC, CLASS_METRIC, MODELS, kill_axis
from src.viz.utils_general import plot_performance, plot_metric
from src.viz.utils_scope import plot_scope_minx_performance, plot_scope_minx_metric


def plot_empty(algo: Literal["lr", "knn"] = "lr"):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
    ]

    plot_performance(axs[0], ROOT, "fluorescence", algo, REG_METRIC, model_prefix="", relative=True, legend=False, colored="dimgrey", title="Fluorescence")
    plot_performance(axs[3], ROOT, "stability", algo, REG_METRIC, model_prefix="", relative=True, legend=False, colored="dimgrey", title="Stability")
    plot_performance(axs[1], ROOT, "deeploc2", algo, CLASS_METRIC, model_prefix="", relative=True, legend=False, colored="dimgrey", title="DeepLoc2 10-class")
    plot_performance(axs[4], ROOT, "deeploc2_bin", algo, CLASS_METRIC, model_prefix="", relative=True, legend=False, colored="dimgrey", title="DeepLoc2 binary")
    plot_scope_minx_performance(axs[2], ROOT, algo, CLASS_METRIC, "superfamily", model_prefix="", relative=True, colored="dimgrey", title="SCOPe40 Superfamily")
    plot_scope_minx_performance(axs[5], ROOT, algo, CLASS_METRIC, "fold", model_prefix="", relative=True, colored="dimgrey", title="SCOPe40 Fold")

    plot_performance(axs[0], ROOT, "fluorescence", algo, REG_METRIC, model_prefix="empty_", relative=True, legend=False, colored="darkgrey", title="Fluorescence")
    plot_performance(axs[3], ROOT, "stability", algo, REG_METRIC, model_prefix="empty_", relative=True, legend=False, colored="darkgrey", title="Stability")
    plot_performance(axs[1], ROOT, "deeploc2", algo, CLASS_METRIC, model_prefix="empty_", relative=True, legend=False, colored="darkgrey", title="DeepLoc2 10-class")
    plot_performance(axs[4], ROOT, "deeploc2_bin", algo, CLASS_METRIC, model_prefix="empty_", relative=True, legend=False, colored="darkgrey", title="DeepLoc2 binary")
    plot_scope_minx_performance(axs[2], ROOT, algo, CLASS_METRIC, "superfamily", model_prefix="empty_", relative=True, colored="darkgrey", title="SCOPe40 Superfamily")
    plot_scope_minx_performance(axs[5], ROOT, algo, CLASS_METRIC, "fold", model_prefix="empty_", relative=True, colored="darkgrey", title="SCOPe40 Fold")

    # handles, labels = axs[0].get_legend_handles_labels()
    # handles = [
    #     Line2D([0], [0], color='dimgrey', linewidth=0),
    #     Line2D([0], [0], color='darkgrey', linewidth=0)
    # ]
    # labels = ["trained", "random"]
    # fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=2)  # -0.08

    fig.suptitle(f"Comparison of trained and random PLMs", fontsize=16)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/empty_{algo}.pdf")
    plt.savefig(f"figures/empty_{algo}.png")


def plot_dataset_analysis(dataset: str, prefix: str = ""):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
    ]

    plot_performance(axs[0], ROOT, dataset, "lr", CLASS_METRIC if any(x in dataset for x in {"deeploc2", "class"}) else REG_METRIC, model_prefix=prefix, relative=True, legend=False, n_classes=10 if dataset in {"fluorescence_classification", "meltome_atlas"} else 42)
    plot_performance(axs[1], ROOT, dataset, "knn", CLASS_METRIC if any(x in dataset for x in {"deeploc2", "class"}) else REG_METRIC, model_prefix=prefix, relative=True, legend=False, n_classes=10 if dataset in {"fluorescence_classification", "meltome_atlas"} else 42)
    plot_metric(axs[2], ROOT, dataset, model_prefix=prefix, relative=True, legend=False, metric="noverlap")
    plot_metric(axs[3], ROOT, dataset, metric="pc@95", relative=True, legend=False)
    plot_metric(axs[4], ROOT, dataset, metric="var@10", relative=True, legend=False)
    plot_metric(axs[5], ROOT, dataset, metric="5dvol", relative=True, legend=False)

    handles, labels = axs[0].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 2)  # -0.08

    fig.suptitle(f"Model Analysis on the {DS_NAME_MAP[dataset]} Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/{prefix}{dataset}.pdf")
    plt.savefig(f"figures/{prefix}{dataset}.png")


def plot_scope_analysis(level: str, prefix: str = ""):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
    ]

    plot_scope_minx_performance(axs[0], ROOT, "lr", CLASS_METRIC, level, model_prefix=prefix, relative=True)
    plot_scope_minx_performance(axs[1], ROOT, "knn", CLASS_METRIC, level, model_prefix=prefix, relative=True)
    plot_scope_minx_metric(axs[2], ROOT, "noverlap", level, model_prefix=prefix, relative=True)
    plot_scope_minx_metric(axs[3], ROOT, "pc@95", level, model_prefix=prefix, relative=True)
    plot_scope_minx_metric(axs[4], ROOT, "var@10", level, model_prefix=prefix, relative=True)
    plot_scope_minx_metric(axs[5], ROOT, "5dvol", level, model_prefix=prefix, relative=True)

    handles, labels = axs[0].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 2)  # -0.08

    fig.suptitle(f"Model Analysis on the SCOPe 40 208 Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/{prefix}scope_{level}_analysis.pdf")
    plt.savefig(f"figures/{prefix}scope_{level}_analysis.png")


def plot_aa_analysis(n_classes: int):
    dataset = "binding" if n_classes == 2 else "scope_40_208"
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), 
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
    ]

    plot_performance(axs[0], ROOT, dataset, "lr", CLASS_METRIC, relative=True, aa=True, n_classes=n_classes)
    plot_performance(axs[1], ROOT, dataset, "knn", CLASS_METRIC, relative=True, aa=True, n_classes=n_classes)
    plot_metric(axs[2], ROOT, dataset, metric="noverlap", relative=True, aa=True, n_classes=n_classes)
    plot_metric(axs[3], ROOT, dataset, metric="pc@95", relative=True, aa=True, n_classes=n_classes)
    plot_metric(axs[4], ROOT, dataset, metric="var@10", relative=True, aa=True, n_classes=n_classes)
    plot_metric(axs[5], ROOT, dataset, metric="5dvol", relative=True, aa=True, n_classes=n_classes)
    
    handles, labels = axs[0].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 2)  # -0.08

    fig.suptitle(f"AA Analysis on the SCOPe 40 dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/aa_{dataset}_{n_classes}_analysis.pdf")
    plt.savefig(f"figures/aa_{dataset}_{n_classes}_analysis.png")


def plot_pca(dataset: str, aa: bool = False):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    ]
    plot_metric(axs[0], ROOT, dataset, metric="zero", relative=True, legend=False, aa=aa)
    plot_metric(axs[1], ROOT, dataset, metric="pc@95", relative=True, legend=False, aa=aa)
    plot_metric(axs[2], ROOT, dataset, metric="var@10", relative=True, legend=False, aa=aa)
    plot_metric(axs[3], ROOT, dataset, metric="5dvol", relative=True, legend=False, aa=aa)

    handles, labels = axs[0].get_legend_handles_labels()
    # handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    # labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 2)  # -0.08

    fig.suptitle(f"PCA Analysis on the {dataset} dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/pca_{dataset}.pdf")
    plt.savefig(f"figures/pca_{dataset}.png")


if __name__ == "__main__":
    Path("figures").mkdir(parents=True, exist_ok=True)
    plot_dataset_analysis("meltome_atlas")
    # plot_dataset_analysis("fluorescence_classification")
    exit(0)

    for ds in ["fluorescence", "stability", "deeploc2", "deeploc2_bin"]:
        print(f"Plotting analysis for the {ds} dataset...")
        plot_dataset_analysis(ds)

    print("Plotting empty LR...")
    plot_empty("lr")

    print("Plotting empty kNN...")
    plot_empty("knn")

    print("Plotting fold scope analysis...")
    plot_scope_analysis("fold", "")

    print("Plotting superfamily scope analysis...")
    plot_scope_analysis("superfamily", "")

    print("Plotting AA analysis of binding dataset...")
    plot_aa_analysis(2)

    print("Plotting AA analysis of 3-class SSP dataset...")
    plot_aa_analysis(3)
    
    print("Plotting AA analysis of 8-class SSP dataset...")
    plot_aa_analysis(8)
