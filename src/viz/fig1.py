import pickle
from pathlib import Path

import matplotlib

from src.viz.utils_general import set_subplot_label

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.lines as mlines
from matplotlib import gridspec

from src.viz.utils_general import plot_performance
from src.viz.utils_scope import plot_scope_minx_performance
from src.viz.plot import DATASET2TASK
from src.viz.constants import MODEL_NAMES

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def plot_improvement_heatmap(ax, df, models, datasets):
    tmp_df = df.loc[models, datasets]
    data = np.array(tmp_df.values - 1, dtype=np.float32)
    rows, cols = data.shape
    
    # ── Colormap: orange → violet (perceptually distinct, colorblind-safe) ──
    # orange_violet = LinearSegmentedColormap.from_list(
    #     "orange_violet",
    #     [
    #         "#EAE2B7",   # warm cream    (mid)
    #         "#9B72CF",   # medium violet
    #         "#4A1070",   # deep violet   (high)
    #     ],
    # )
    # orange_violet.set_bad(color="lightgray")   # NaN → white cell
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
    from matplotlib.colors import BoundaryNorm, PowerNorm
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    norm = PowerNorm(gamma=0.7, vmin=0, vmax=1)

    # ── Figure layout ────────────────────────────────────────────────────────
    # cell_size = max(0.8, min(1.5, 8 / max(rows, cols)))
    # figsize = (cols * cell_size + 2, rows * cell_size + 1.2)

    # fig, ax = plt.subplots(figsize=figsize)
    # fig.patch.set_facecolor("#1A1A2E")
    # ax.set_facecolor("#1A1A2E")

    # ── Draw heatmap ─────────────────────────────────────────────────────────
    im = ax.imshow(
        data,
        cmap=orange_violet,
        # vmin=0, vmax=1,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    # ── Cell annotations ─────────────────────────────────────────────────────
    font_size = 13# max(6, min(13, int(120 / max(rows, cols))))

    for r in range(rows):
        for c in range(cols):
            val = data[r, c]
            if np.isnan(val):
                label = "NaN"
                text_color = "#AAAAAA"
            else:
                label = f"+{val * 100:.0f}%" if val > 0 else "±0%"
                # Dark text on light cells, light text on dark cells
                norm_val = val          # already 0-1
                text_color = "#1A1A2E" if norm_val < 0.45 else "#FFFFFF"
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
    # ax.set_xticklabels(dataset_names, color="#1A1A2E", fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels([MODEL_NAMES[m] for m in models], color="#1A1A2E", fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


    # --- Row 2: group labels on a twin axis pushed further down ---
    fine_labels  = ["Regression", "Binary", "", "", "", "10c", "Binary", "Fold", "Superfamily", "3c SSP", "8c SSP", ""]
    group_labels = ['Fluorescence', 'Meltome A.', 'Stability', "Solubility", 'DeepLoc2', 'SCOPe40', 'Binding']
    group_sizes  = [2, 1, 1, 1, 2, 4, 1]   # ← only change needed for different groupings

    second_labels = ["Protein Level Tasks", "Residue Level Tasks"]
    group_sizes_2 = [9, 3]

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


with open("knn_metric.pkl", "rb") as f:
    knn_metric = pickle.load(f)

df = pd.DataFrame(index=knn_metric.keys(), columns=knn_metric["esm_t6"].keys())
for model in knn_metric.keys():
    for dataset in knn_metric[model].keys():
        if knn_metric[model][dataset][-1] == 0:
            df.loc[model, dataset] = np.nan
        else:
            df.loc[model, dataset] = max(knn_metric[model][dataset]) / knn_metric[model][dataset][-1]

models = ["esm_t33", "esm_t36", "esmc_600m", "ankh_large", "prott5", "prostt5", "progen2_medium", "progen2_large", "protgpt2"]
datasets = ["fluorescence", "fluorescence_classification", "meltome_atlas", "stability", "solubility", "deeploc2", "deeploc2_bin", "scope_40_208_fold", "scope_40_208_superfamily", "scope_40_208_3ssp", "scope_40_208_8ssp", "binding"]

fig = plt.figure(figsize=(20, 9))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2])
gs_right = gs[0].subgridspec(2, 2, wspace=0.2)
axs = [
    fig.add_subplot(gs_right[0, 0]), fig.add_subplot(gs_right[0, 1]),
    fig.add_subplot(gs_right[1, 0]), fig.add_subplot(gs_right[1, 1]),
    fig.add_subplot(gs[1]),
]
plot_performance(axs[0], BASE, "fluorescence", "knn", "r2", task=DATASET2TASK["fluorescence"], models=models, relative=True, title=False)
plot_performance(axs[1], BASE, "stability", "knn", "r2", task=DATASET2TASK["stability"], models=models, relative=True, title=False)
# axs[2].set_ylim(bottom=-0.05, top=1.05)
plot_performance(axs[2], BASE, "deeploc2", "knn", "mcc", task=DATASET2TASK["deeploc2"], models=models, relative=True, title=False)
plot_scope_minx_performance(axs[3], BASE, "knn", "mcc", "fold", model_prefix="", models=models, relative=True, title=False)
plot_improvement_heatmap(axs[4], df, models, datasets)

handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.26, 0.05), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

set_subplot_label(axs[0], fig, "A")
axs[0].set_ylabel("$R^2 (↑)$")
set_subplot_label(axs[1], fig, "B")
axs[1].set_ylabel("$R^2 (↑)$")
set_subplot_label(axs[2], fig, "C")
axs[2].set_ylabel("MCC (↑)")
set_subplot_label(axs[3], fig, "D")
axs[3].set_ylabel("MCC (↑)")
set_subplot_label(axs[4], fig, "E")

plt.tight_layout(rect=[0, 0.085, 1, 1])
plt.savefig("layer_improvement.pdf", dpi=300)
plt.show()
