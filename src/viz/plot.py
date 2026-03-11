from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.viz.constants import DS_NAME_MAP, ROOT, REG_METRIC, CLASS_METRIC, MODELS, kill_axis, DATASET2TASK
from src.viz.utils_general import compute_performance, plot_dataset_finetune_comparison, plot_performance, plot_metric
from src.viz.utils_scope import compute_scope_performance, plot_scope_minx_performance, plot_scope_minx_metric


def plot_empty(algo: Literal["lr", "knn"] = "lr"):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
    ]

    plot_performance(axs[0], ROOT, "fluorescence", algo, REG_METRIC, model_prefix="", relative=True, legend=False, colored="dimgrey", title="Fluorescence", task=DATASET2TASK["fluorescence"])
    plot_performance(axs[3], ROOT, "stability", algo, REG_METRIC, model_prefix="", relative=True, legend=False, colored="dimgrey", title="Stability", task=DATASET2TASK["stability"])
    plot_performance(axs[1], ROOT, "deeploc2", algo, CLASS_METRIC, model_prefix="", relative=True, legend=False, colored="dimgrey", title="DeepLoc2 10-class", task=DATASET2TASK["deeploc2"])
    plot_performance(axs[4], ROOT, "deeploc2_bin", algo, CLASS_METRIC, model_prefix="", relative=True, legend=False, colored="dimgrey", title="DeepLoc2 binary", task=DATASET2TASK["deeploc2_bin"])
    plot_scope_minx_performance(axs[2], ROOT, algo, CLASS_METRIC, "superfamily", model_prefix="", relative=True, colored="dimgrey", title="SCOPe40 Superfamily")
    plot_scope_minx_performance(axs[5], ROOT, algo, CLASS_METRIC, "fold", model_prefix="", relative=True, colored="dimgrey", title="SCOPe40 Fold")

    plot_performance(axs[0], ROOT, "fluorescence", algo, REG_METRIC, model_prefix="empty_", relative=True, legend=False, colored="darkgrey", title="Fluorescence", task=DATASET2TASK["fluorescence"])
    plot_performance(axs[3], ROOT, "stability", algo, REG_METRIC, model_prefix="empty_", relative=True, legend=False, colored="darkgrey", title="Stability", task=DATASET2TASK["stability"])
    plot_performance(axs[1], ROOT, "deeploc2", algo, CLASS_METRIC, model_prefix="empty_", relative=True, legend=False, colored="darkgrey", title="DeepLoc2 10-class", task=DATASET2TASK["deeploc2"])
    plot_performance(axs[4], ROOT, "deeploc2_bin", algo, CLASS_METRIC, model_prefix="empty_", relative=True, legend=False, colored="darkgrey", title="DeepLoc2 binary", task=DATASET2TASK["deeploc2_bin"])
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
    plt.savefig(f"figures/empty_{algo}.png", transparent=True)


def plot_dataset_analysis(dataset: str, prefix: str = ""):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
    ]

    plot_performance(
        axs[0], 
        ROOT, 
        dataset, 
        "lr", 
        CLASS_METRIC if any(x in dataset for x in {"deeploc2", "class"}) else REG_METRIC, 
        model_prefix=prefix, 
        relative=True, 
        legend=False, 
        n_classes=10 if dataset in {"fluorescence_classification", "meltome_atlas"} else 42,
        task=DATASET2TASK[dataset],
    )
    plot_performance(
        axs[1], 
        ROOT, 
        dataset, 
        "knn",
        CLASS_METRIC if any(x in dataset for x in {"deeploc2", "class"}) else REG_METRIC, 
        model_prefix=prefix, 
        relative=True, 
        legend=False, 
        n_classes=10 if dataset in {"fluorescence_classification", "meltome_atlas"} else 42,
        task=DATASET2TASK[dataset],
    )
    plot_metric(axs[2], ROOT, dataset, model_prefix=prefix, metric="noverlap", relative=True, legend=False)
    plot_metric(axs[3], ROOT, dataset, model_prefix=prefix, metric="pc@95", relative=True, legend=False)
    plot_metric(axs[4], ROOT, dataset, model_prefix=prefix, metric="var@10", relative=True, legend=False)
    plot_metric(axs[5], ROOT, dataset, model_prefix=prefix, metric="ids", relative=True, legend=False)

    handles, labels = axs[1].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 2)  # -0.08

    fig.suptitle(f"Model Analysis on the {DS_NAME_MAP[dataset]} Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/{prefix}{dataset}.pdf")
    plt.savefig(f"figures/{prefix}{dataset}.png", transparent=True)


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
    plot_scope_minx_metric(axs[5], ROOT, "ids", level, model_prefix=prefix, relative=True)

    handles, labels = axs[1].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 2)  # -0.08

    fig.suptitle(f"Model Analysis on the SCOPe 40 208 Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/{prefix}scope_{level}_analysis.pdf")
    plt.savefig(f"figures/{prefix}scope_{level}_analysis.png", transparent=True)


def plot_aa_analysis(n_classes: int):
    dataset = "binding" if n_classes == 2 else "scope_40_208"
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), 
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
    ]

    plot_performance(axs[0], ROOT, dataset, "lr", CLASS_METRIC, relative=True, aa=True, n_classes=n_classes, task="binary" if n_classes == 2 else "multi-class")
    plot_performance(axs[1], ROOT, dataset, "knn", CLASS_METRIC, relative=True, aa=True, n_classes=n_classes, task="binary" if n_classes == 2 else "multi-class")
    plot_metric(axs[2], ROOT, dataset, metric="noverlap", relative=True, aa=True, n_classes=n_classes)
    plot_metric(axs[3], ROOT, dataset, metric="pc@95", relative=True, aa=True, n_classes=n_classes)
    plot_metric(axs[4], ROOT, dataset, metric="var@10", relative=True, aa=True, n_classes=n_classes)
    plot_metric(axs[5], ROOT, dataset, metric="ids", relative=True, aa=True, n_classes=n_classes)
    
    handles, labels = axs[1].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 2)  # -0.08

    fig.suptitle(f"AA Analysis on the SCOPe 40 dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/aa_{dataset}_{n_classes}_analysis.pdf")
    plt.savefig(f"figures/aa_{dataset}_{n_classes}_analysis.png", transparent=True)


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
    plt.savefig(f"figures/pca_{dataset}.png", transparent=True)


def plot_dataset_comp(dataset_name: Literal["fluorescence", "deeploc2", "scope_40_208"]):
    ds1, ds2 = {"fluorescence": ("fluorescence", "fluorescence_classification"), "deeploc2": ("deeploc2", "deeploc2_bin")}[dataset_name]
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    ]

    plot_performance(axs[0], ROOT, ds1, "lr", REG_METRIC if ds1 == "fluorescence" else CLASS_METRIC, model_prefix="", relative=True, legend=False, aa=False, n_classes=42, task=DATASET2TASK[ds1])
    plot_performance(axs[1], ROOT, ds1, "knn", REG_METRIC if ds1 == "fluorescence" else CLASS_METRIC, model_prefix="", relative=True, legend=False, aa=False, n_classes=42, task=DATASET2TASK[ds1])
    plot_performance(axs[2], ROOT, ds2, "lr", CLASS_METRIC, model_prefix="", relative=True, legend=False, aa=False, n_classes=42, task=DATASET2TASK[ds2])
    plot_performance(axs[3], ROOT, ds2, "knn", CLASS_METRIC, model_prefix="", relative=True, legend=False, aa=False, n_classes=42, task=DATASET2TASK[ds2])

    handles, labels = axs[1].get_legend_handles_labels()
    handles.insert(7, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(7, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 5)  # -0.08

    fig.suptitle(f"Comparison of LR and kNN on the {dataset_name.capitalize()} dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(f"figures/{dataset_name}_comp.pdf")
    plt.savefig(f"figures/{dataset_name}_comp.png", transparent=True)


def plot_scope_ssp_comp():
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    ]

    plot_performance(axs[0], ROOT, "scope_40_208", "lr", CLASS_METRIC, relative=True, aa=True, n_classes=3, task="multi-class")
    plot_performance(axs[1], ROOT, "scope_40_208", "knn", CLASS_METRIC, model_prefix="", relative=True, legend=False, aa=True, n_classes=3, task="multi-class")
    plot_performance(axs[2], ROOT, "scope_40_208", "lr", CLASS_METRIC, model_prefix="", relative=True, legend=False, aa=True, n_classes=8, task="multi-class")
    plot_performance(axs[3], ROOT, "scope_40_208", "knn", CLASS_METRIC, model_prefix="", relative=True, legend=False, aa=True, n_classes=8, task="multi-class")

    handles, labels = axs[1].get_legend_handles_labels()
    handles.insert(7, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(7, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 5)  # -0.08

    fig.suptitle(f"Comparison of LR and kNN on the SCOPe40 2.08 3/8-class SSP datasets", fontsize=16)
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(f"figures/scope_40_208_ssp_comp.pdf")
    plt.savefig(f"figures/scope_40_208_ssp_comp.png", transparent=True)


def plot_olga():
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    ]

    plot_performance(axs[0], ROOT, "fluorescence", "lr", REG_METRIC, model_prefix="", relative=True, legend=False, aa=False, n_classes=42, title=r"$\bf{Fluorescence}$" + "\nRMSE of LR heads", task=DATASET2TASK["fluorescence"])
    plot_performance(axs[1], ROOT, "stability", "lr", REG_METRIC, model_prefix="", relative=True, legend=False, aa=False, n_classes=42, title=r"$\bf{Stability}$" + "\nRMSE of LR heads", task=DATASET2TASK["stability"])
    plot_performance(axs[2], ROOT, "deeploc2", "lr", CLASS_METRIC, model_prefix="", relative=True, legend=False, aa=False, n_classes=42, title=r"$\bf{Localization}$" + "\nMCC of LR heads", task=DATASET2TASK["deeploc2"])
    plot_scope_minx_performance(axs[3], ROOT, "lr", CLASS_METRIC, "fold", model_prefix="", relative=True, title=r"$\bf{Remote\ homology}$" + "\nMCC of LR heads (Fold Min-10)")

    handles, labels = axs[0].get_legend_handles_labels()
    handles.insert(5, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    labels.insert(5, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=(len(MODELS) + 1) // 2)  # -0.08

    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig(f"figures/plot_olga.pdf")


def plot_finetuning_comp():
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    ]

    for i, dataset in enumerate(["fluorescence", "stability", "deeploc2", "deeploc2_bin"]):
        plot_dataset_finetune_comparison(axs[i], ROOT, dataset, CLASS_METRIC if dataset.startswith("deeploc2") else REG_METRIC, n_classes=10, task=DATASET2TASK[dataset])

    handles, labels = axs[0].get_legend_handles_labels()
    # handles.insert(7, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))  #plt.Line2D([0], [0], color="black", lw=0, label="OHE"))
    # labels.insert(7, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=2)  # -0.08

    fig.suptitle(f"Comparison of layerwise trained LR and finetuned PLMs", fontsize=16)
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.savefig(f"figures/finetune_comp.pdf")
    plt.savefig(f"figures/finetune_comp.png", transparent=True)


if __name__ == "__main__":
    Path("figures").mkdir(parents=True, exist_ok=True)

    print("Hello")
    root = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"
    # print(compute_scope_performance(root, "ankh_base", "scope_40_208", 35, "lr", "mcc", "superfamily", min_x=10))
    print(compute_performance(root, "esm_t30", "deeploc2", 10, "lr", "mcc", aa=False, n_classes=10, task="multi-label"))

    # plot_finetuning_comp()
    # plot_dataset_comp("fluorescence")
    # plot_dataset_comp("deeploc2")
    # plot_dataset_analysis("meltome_atlas")
    # for ds in ["deeploc2", "fluorescence", "meltome_atlas", "stability"]:
    #     print(f"Plotting analysis for the {ds} dataset...")
    #     plot_dataset_analysis(ds)

    # print("Plotting empty LR...")
    # plot_empty("lr")

    # print("Plotting empty kNN...")
    # plot_empty("knn")

    # print("Plotting fold scope analysis...")
    # plot_scope_analysis("fold", "")

    # print("Plotting superfamily scope analysis...")
    # plot_scope_analysis("superfamily", "")

    # print("Plotting AA analysis of binding dataset...")
    # plot_aa_analysis(2)

    # print("Plotting AA analysis of 3-class SSP dataset...")
    # plot_aa_analysis(3)
    
    # print("Plotting AA analysis of 8-class SSP dataset...")
    # plot_aa_analysis(8)

    # plot_scope_ssp_comp()
