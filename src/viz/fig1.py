import pickle
from pathlib import Path
from typing import Literal

from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from matplotlib import gridspec, pyplot as plt, lines as mlines
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm

from src.viz.constants import CLASS_METRIC, DATASET_NAMES, METRIC_TITLES, MODEL_NAMES, DATASET2TASK, REG_METRIC, TASK_METRICS, kill_axis
from src.viz.plot_utils import plot_scope_minx_performance, plot_performance, set_subplot_label

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def plot_improvement_heatmap(ax, df, models, datasets):
    tmp_df = df.loc[models, datasets]
    data = np.array(tmp_df.values - 1, dtype=np.float32)
    rows, cols = data.shape
    
    base_cmap = LinearSegmentedColormap.from_list(
    "orange_violet_base",
        [
            "#ECD75C",   # lighter orange (midpoint between original cream and darker orange)
            "#9B72CF",   # medium violet
            "#4A1070",   # deep violet        (high)
        ],
    )

    # Sample 255 colors from the base cmap (for values 1–200)
    n = 255
    base_colors = base_cmap(np.linspace(0, 1, n))

    # Prepend white for the 0-value bin
    white = np.array([[1.0, 1.0, 1.0, 1.0]])
    all_colors = np.vstack([white, base_colors])  # shape: (256, 4)

    orange_violet = ListedColormap(all_colors, name="orange_violet")
    orange_violet.set_bad(color="lightgray")  # NaN → light gray

    # Usage with a Normalize so 0 maps to index 0 and 200 maps to index 255
    norm = PowerNorm(gamma=0.7, vmin=0, vmax=1)

    # ── Draw heatmap ─────────────────────────────────────────────────────────
    ax.imshow(
        data,
        cmap=orange_violet,
        # vmin=0, vmax=1,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    # ── Cell annotations ─────────────────────────────────────────────────────
    font_size = 13

    for r in range(rows):
        for c in range(cols):
            val = data[r, c]
            if np.isnan(val):
                label = "NaN"
                text_color = "#AAAAAA"
            else:
                if val < 0.1:
                    label = f"+{val * 100:.1f}%" if val > 0 else "±0%"
                else:
                    label = f"+{val * 100:.0f}%"
                text_color = "#1A1A2E" if val < 0.45 else "#FFFFFF"
            ax.text(
                c, r, label,
                ha="center", va="center",
                fontsize=font_size,
                color=text_color,
                fontfamily="Montserrat",
                fontweight="bold",
            )

    # ── Grid lines ───────────────────────────────────────────────────────────
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#1A1A2E", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # ── Axis labels ──────────────────────────────────────────────────────────
    # ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_yticklabels([MODEL_NAMES[m] for m in models], color="#1A1A2E")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


    # --- Row 2: group labels on a twin axis pushed further down ---
    fine_labels  = ["Binary", "Reg.", "", "Tm", "Species", "", "", "Binary", "10-class", "Fold", "Superf.", "3c SSP", "8c SSP", ""]
    group_labels = ['Fluorescence', "GB1", 'Meltome A.', 'Stability', "DeepSol", 'DeepLoc2.0', 'SCOPe40', 'Binding']
    group_sizes  = [2, 1, 2, 1, 1, 2, 4, 1]   # ← only change needed for different groupings

    second_labels = ["Protein Level Tasks", "Residue Level Tasks"]
    group_sizes_2 = [11, 3]

    # Derived positions
    starts  = np.cumsum([0] + group_sizes[:-1])          # first col index of each group
    centres = starts + (np.array(group_sizes) - 1) / 2   # label centre
    ends    = starts + np.array(group_sizes) - 1

    snd_starts  = np.cumsum([0] + group_sizes_2[:-1])          # first col index of each group
    snd_centres = snd_starts + (np.array(group_sizes_2) - 1) / 2   # label centre
    snd_ends    = snd_starts + np.array(group_sizes_2) - 1

    # Row 1: fine labels
    ax.set_xticks(range(sum(group_sizes)))
    ax.set_xticklabels(fine_labels)
    ax.xaxis.set_ticks_position('bottom')

    # Row 2: group labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(centres)
    ax2.set_xticklabels(group_labels, fontweight='bold')

    ax2.xaxis.set_label_position('bottom')   # ← move label row to bottom
    ax2.xaxis.set_ticks_position('bottom')   # ← move ticks to bottom
    ax2.tick_params(bottom=False, top=False) # ← hide tick marks
    ax2.spines['bottom'].set_position(('outward', 25))  # ← push below fine labels
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # Span lines per group
    inset = 0.3
    depth = 0.04
    trans = ax.get_xaxis_transform()  # data-x, axes-fraction-y
    for s, e in zip(starts, ends):
        if s == e:
            continue  # skip single-column groups (no line needed)
        line = mlines.Line2D([s - inset, e + inset], [-depth, -depth],
                            transform=trans, clip_on=False,
                            color='black', linewidth=1.2,
                            solid_capstyle='butt')
        ax.add_line(line)

    # Row 3: task type labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(snd_centres)
    ax2.set_xticklabels(second_labels, fontweight='bold')

    ax2.xaxis.set_label_position('bottom')   # ← move label row to bottom
    ax2.xaxis.set_ticks_position('bottom')   # ← move ticks to bottom
    ax2.tick_params(bottom=False, top=False) # ← hide tick marks
    ax2.spines['bottom'].set_position(('outward', 50))  # ← push below fine labels
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # Span lines per group
    inset = 0.3
    depth = 0.1
    trans = ax.get_xaxis_transform()  # data-x, axes-fraction-y
    for s, e in zip(snd_starts, snd_ends):
        if s == e:
            continue  # skip single-column groups (no line needed)
        line = mlines.Line2D([s - inset, e + inset], [-depth, -depth],
                            transform=trans, clip_on=False,
                            color='black', linewidth=1.2,
                            solid_capstyle='butt')
        ax.add_line(line)


def plot_fig1(models):
    with open(f"data_knn_{CLASS_METRIC}_{REG_METRIC}.pkl", "rb") as f:
        knn_metric = pickle.load(f)

    df = pd.DataFrame(index=knn_metric.keys(), columns=knn_metric["esm_t6"].keys())
    for model in knn_metric.keys():
        for dataset in knn_metric[model].keys():
            if knn_metric[model][dataset][-1] == 0:
                df.loc[model, dataset] = np.nan
            else:
                df.loc[model, dataset] = max(knn_metric[model][dataset]) / knn_metric[model][dataset][-1]

    datasets = ["fluorescence_classification", "fluorescence", "gb1", "meltome_atlas_species", "meltome_atlas", "stability", "solubility", "deeploc2_bin", "deeploc2", "scope_40_208_fold", "scope_40_208_superfamily", "scope_40_208_3ssp", "scope_40_208_8ssp", "binding"]

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
    gs_grid = gs[1].subgridspec(2, 2, wspace=0.2, hspace=0.2)
    axs = [
        fig.add_subplot(gs_grid[0, 0]), fig.add_subplot(gs_grid[0, 1]),
        fig.add_subplot(gs_grid[1, 0]), fig.add_subplot(gs_grid[1, 1]),
        fig.add_subplot(gs[0]),
    ]
    plot_performance(axs[0], BASE, "fluorescence", "knn", REG_METRIC, task=DATASET2TASK["fluorescence"], models=models, relative=True)
    plot_performance(axs[1], BASE, "stability", "knn", REG_METRIC, task=DATASET2TASK["stability"], models=models, relative=True)
    plot_performance(axs[2], BASE, "deeploc2", "knn", CLASS_METRIC, task=DATASET2TASK["deeploc2"], models=models, relative=True)
    plot_scope_minx_performance(axs[3], BASE, "knn", CLASS_METRIC, "fold", model_prefix="", models=models, relative=True)
    plot_improvement_heatmap(axs[4], df, models, datasets)

    for i, name in enumerate(["fluorescence", "stability", "deeploc2", "scope_40_208_fold"]):
        set_subplot_label(axs[i], fig, chr(ord("B") + i))
        axs[i].set_title(DATASET_NAMES[name])
        axs[i].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK[name]]])
        axs[i].grid()
        if i < 2:
            axs[i].tick_params(axis='x', labelbottom=False, bottom=False)
            axs[i].set_xlabel("")
        else:
            axs[i].set_xlabel("Relative Layer")
    set_subplot_label(axs[4], fig, "A")

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.78, 0.08), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

    plt.tight_layout(rect=[0, 0.085, 1, 1])
    # plt.savefig("paper_figures/1_layer_improvement.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper_figures/1_layer_improvement.png")


def plot_full_fig1_left():
    with open(f"data_knn_{CLASS_METRIC}_{REG_METRIC}.pkl", "rb") as f:
        knn_metric = pickle.load(f)

    df = pd.DataFrame(index=knn_metric.keys(), columns=knn_metric["esm_t6"].keys())
    for model in knn_metric.keys():
        for dataset in knn_metric[model].keys():
            if knn_metric[model][dataset][-1] == 0:
                df.loc[model, dataset] = np.nan
            else:
                df.loc[model, dataset] = max(knn_metric[model][dataset]) / knn_metric[model][dataset][-1]

    _, ax = plt.subplots(figsize=(15, 9))
    plot_improvement_heatmap(
        ax, 
        df, 
        ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh_base", "ankh_large", 
         "prott5", "prostt5", "progen2_small", "progen2_medium", "progen2_large", "protgpt2"], 
        ["fluorescence_classification", "fluorescence", "gb1", "meltome_atlas_species", "meltome_atlas", "stability", "solubility", "deeploc2_bin", "deeploc2", 
         "scope_40_208_fold", "scope_40_208_superfamily", "scope_40_208_3ssp", "scope_40_208_8ssp", "binding"],
    )
    plt.tight_layout(rect=[0, 0.085, 1, 1])
    plt.savefig("paper_figures/full_1_improvement_heatmap.pdf", dpi=300, bbox_inches="tight")


def plot_full_fig1_right(algo: Literal["knn", "lr"] = "knn"):
    models = ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh_base", "ankh_large", "prott5", "prostt5", "progen2_small", "progen2_medium", "progen2_large", "protgpt2"]
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
    
    plot_performance(axs[0][0], BASE, "fluorescence_classification", algo, CLASS_METRIC, task="binary", models=models, relative=True)
    plot_performance(axs[1][0], BASE, "fluorescence", algo, REG_METRIC, task="regression", models=models, relative=True)
    plot_performance(axs[2][0], BASE, "gb1", algo, REG_METRIC, task="regression", models=models, relative=True)
    
    plot_performance(axs[0][1], BASE, "meltome_atlas_species", algo, CLASS_METRIC, task="multi-class", models=models, relative=True)
    plot_performance(axs[1][1], BASE, "meltome_atlas", algo, REG_METRIC, task="regression", models=models, relative=True)
    plot_performance(axs[2][1], BASE, "stability", algo, REG_METRIC, task="regression", models=models, relative=True)

    plot_performance(axs[0][2], BASE, "deeploc2_bin", algo, CLASS_METRIC, task="binary", models=models, relative=True)
    plot_performance(axs[1][2], BASE, "deeploc2", algo, CLASS_METRIC, task="multi-label", models=models, relative=True)
    plot_performance(axs[2][2], BASE, "solubility", algo, CLASS_METRIC, task="binary", models=models, relative=True)

    plot_scope_minx_performance(axs[0][3], BASE, algo, CLASS_METRIC, "superfamily", model_prefix="", models=models, relative=True)
    plot_scope_minx_performance(axs[1][3], BASE, algo, CLASS_METRIC, "fold", model_prefix="", models=models, relative=True)
    
    plot_performance(axs[0][4], BASE, "scope_40_208", "knn", CLASS_METRIC, relative=True, aa=True, n_classes=3, task="multi-class", models=models)
    plot_performance(axs[1][4], BASE, "scope_40_208", "knn", CLASS_METRIC, relative=True, aa=True, n_classes=8, task="multi-class", models=models)
    plot_performance(axs[2][4], BASE, "binding", algo, CLASS_METRIC, task="binary", aa=True, n_classes=2, models=models, relative=True)

    for t, name in enumerate(["fluorescence_classification", "fluorescence", "gb1", "meltome_atlas_species", "meltome_atlas", "stability", "deeploc2_bin", "deeploc2", "solubility", "scope_40_208_superfamily", "scope_40_208_fold", "", "scope_40_208_3ssp", "scope_40_208_8ssp", "binding"]):
        if name == "":
            kill_axis(axs[t % 3][t // 3])
            continue
        axs[t % 3][t // 3].set_title(DATASET_NAMES[name])
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
    plt.savefig("paper_figures/full_1_improvement_layer_p.pdf", dpi=300, bbox_inches="tight")
