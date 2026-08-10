import pickle
from typing import Literal

from matplotlib import gridspec, pyplot as plt, lines as mlines
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm
import numpy as np
import pandas as pd
from sklearn.metrics import auc

from src.viz.constants import DATASET_COLORS, DATASET_NAMES, METRIC_TITLES, MODEL_NAMES
from src.viz.plot_utils import set_subplot_label
from src.viz.utils import interpolate_data, XP


def scale_data(data):
    return data  # normalize(data, axis=2)


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
    fine_labels  = ["Binary", "Reg.", "", "Species", "Tm", "Rocklin", "Tsuboyama", "", "Binary", "10-class", "Fold", "Superf.", "3c SSP", "8c SSP", ""]
    group_labels = ['Fluorescence', "GB1", 'Meltome A.', 'Stability', "DeepSol", 'DeepLoc2.0', 'SCOPe40', 'Binding']
    group_sizes  = [2, 1, 2, 2, 1, 2, 4, 1]   # ← only change needed for different groupings

    second_labels = ["Protein Level Tasks", "Residue Level Tasks"]
    group_sizes_2 = [12, 3]

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
    depth = 0.03
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
    depth = 0.065
    trans = ax.get_xaxis_transform()  # data-x, axes-fraction-y
    for s, e in zip(snd_starts, snd_ends):
        if s == e:
            continue  # skip single-column groups (no line needed)
        line = mlines.Line2D([s - inset, e + inset], [-depth, -depth],
                            transform=trans, clip_on=False,
                            color='black', linewidth=1.2,
                            solid_capstyle='butt')
        ax.add_line(line)


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
        axs[1].plot(XP, mean, label=DATASET_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])
        # axs[1].fill_between(XP, np.nanmin(class_data[d], axis=0), np.nanmax(class_data[d], axis=0), alpha=0.2)

    for d, dataset in enumerate(["fluorescence", "gb1", "stability", "meltome_atlas", "tsuboyama"]):
        mean = np.nanmean(reg_data[d], axis=0)
        # print(dataset, ":", auc(XP, mean))
        axs[2].plot(XP, mean, label=DATASET_NAMES[dataset], linewidth=5, color=DATASET_COLORS[dataset])
        # axs[2].fill_between(XP, np.nanmin(reg_data[d], axis=0), np.nanmax(reg_data[d], axis=0), alpha=0.2)

    set_subplot_label(axs[0], fig, "A")
    for i in range(1, 3):
        axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel("Relative Layer")
        axs[i].set_ylabel(METRIC_TITLES[class_metric if i == 1 else reg_metric])
        set_subplot_label(axs[i], fig, chr(ord("A") + i))

    plt.tight_layout()
    plt.savefig(f"paper_figures/fig_1_{algo}_{class_metric}_{reg_metric}.pdf", dpi=300, bbox_inches="tight")
