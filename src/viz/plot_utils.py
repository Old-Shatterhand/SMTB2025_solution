from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt, lines as mlines, transforms
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm

from src.viz.utils import compute_performance, compute_scope_performance, interpolate_data, read_metric, read_pca_metric, read_scope_metric
from src.viz.constants import LAYERS, MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, MODELS


def plot_performance(
        ax, 
        root: Path, 
        dataset: str, 
        algo: str, 
        metric: str, 
        task: Literal["regression", "binary", "multi-label", "multi-class"] = "regression",
        aa: bool = False, 
        n_classes: int = 42,
        model_prefix: Literal["", "empty_"] = "", 
        relative: bool = False, 
        models: list[str] = MODELS,
        colored: bool | str = True, 
    ) -> dict[str, list[float]]:
    """
    Plot performance metrics for different models on a given axis.

    Args:
        ax: Matplotlib axis to plot on.
        root: Root directory containing the embeddings and results.
        dataset: Name of the dataset.
        algo: Algorithm used (e.g., "lr", "knn").
        metric: Performance metric to plot (e.g., "pearson", "mcc").
        relative: Whether to plot relative layer positions.
        model_prefix: Prefix to add to model names. Either "" or "empty_" to indicate using normal or untrained models.
        legend: Whether to display the legend.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
        task: Type of task ("regression", "binary", "multi-label", "multi-class").
        models: List of model names to include in the plot.
        colored: Color setting for the plot lines.
    
    Returns:
        A dictionary mapping model names to their performance metrics across layers.
    """
    performances = {}
    for model in models:
        perfs = [compute_performance(root, model_prefix + model, dataset, layer, algo=algo, metric=metric, aa=aa, n_classes=n_classes, task=task) for layer in range(LAYERS[model] + 1)]
        performances[model] = perfs
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        ax.plot(
            np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])) if relative else np.arange(len(perfs)),
            perfs, 
            label=model_prefix + MODEL_NAMES.get(model, model), 
            c=MODEL_COLORS.get(model, None) if colored == True else colored,
            marker=MODEL_MARKERS.get(model, None) if colored == True else None,
        )
    return performances


def plot_metric(
        ax, 
        root: Path | None, 
        dataset: str | None, 
        relative: bool, 
        metric: Literal["ids", "density", "noverlap", "noverlap_50", "zero", "pc@95", "var@10", "5dvol"] = "ids", 
        model_prefix: Literal["", "empty_"] = "", 
        legend: bool = False, 
        aa: bool = False, 
        n_classes: int = 42,
        models: list[str] = MODELS,
        colored: bool | str = True,
        title: str | bool | None = None,
        grouped: bool = False,
        data: dict | None = None,
    ) -> None:
    """
    Plot a specific metric for different models on a given axis.

    Args:
        ax: Matplotlib axis to plot on.
        root: Root directory containing the embeddings and results.
        dataset: Name of the dataset.
        relative: Whether to plot relative layer positions.
        metric: Metric to plot ("ids", "density", "noverlap", "noverlap_50", "zero", "pc@95", "var@10", "5dvol").
        model_prefix: Prefix to add to model names. Either "" or "empty_" to indicate using normal or untrained models.
        legend: Whether to display the legend.
        aa: Whether to use amino acid level embeddings.
        n_classes: Number of classes for classification tasks. Only used for aa-tasks.
        models: List of model names to include in the plot.
        colored: Color setting for the plot lines.
        title: Optional title for the plot. Can be a string, boolean, or None.
    """
    title_map = {"ids": "Intrinsic Dimensions", "density": "Density", "noverlap": "Neighbor Overlap", "noverlap_50": "Neighbor Overlap (50)"}
    all_perfs = []
    for model in models:
        # if metric == "5dvol" and model.startswith("ankh"):
        #     continue  # ankh is crazy in this metric
        if data is None:
            perfs = []
            for layer in range(LAYERS[model] + 1):
                if metric.startswith("noverlap") and layer == LAYERS[model]:
                    continue
                if metric in {"zero", "pc@95", "var@10", "5dvol"}:
                    result = read_pca_metric(root, model_prefix + model, dataset, layer, metric=metric, aa=aa)
                else:
                    result = read_metric(root, model_prefix + model, dataset, layer, metric=metric, aa=aa)
                perfs.append(result)
            if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
                continue
        else:
            if dataset == "scope_40_208" and aa:
                dataset = "scope_40_208_3ssp"
            perfs = data[model][dataset]
        if sum(perfs) < 2:
            continue
        all_perfs.append(perfs)
        if not grouped:
            if relative:
                x_ticks = np.arange(0, 1 + 1e-5, 1 / LAYERS[model])
                if metric.startswith("noverlap"):
                    x_ticks = x_ticks[:-1]
                    x_ticks += 1 / (2 * LAYERS[model])
                    perfs = perfs[:LAYERS[model]]
                ax.plot(
                    x_ticks, 
                    perfs, 
                    label=model_prefix + MODEL_NAMES.get(model, model), 
                    c=MODEL_COLORS.get(model, None) if colored == True else colored, 
                    marker=MODEL_MARKERS.get(model, None)
                )
            else:
                ax.plot(
                    perfs, 
                    label=model_prefix + MODEL_NAMES.get(model, model), 
                    c=MODEL_COLORS.get(model, None) if colored == True else colored, 
                    marker=MODEL_MARKERS.get(model, None)
                )
    if grouped:
        interp_perfs = interpolate_data([all_perfs])[0]
        ax.plot(
            np.linspace(0, 1, 1000), 
            np.nanmean(interp_perfs, axis=0), 
            label=model_prefix + MODEL_NAMES.get(model, model), 
            # c=MODEL_COLORS.get(model, None) if colored == True else colored, 
            # marker=MODEL_MARKERS.get(model, None)
        )
        ax.fill_between(
            np.linspace(0, 1, 1000), 
            np.nanmin(interp_perfs, axis=0), 
            np.nanmax(interp_perfs, axis=0), 
            alpha=0.2, 
            # color=MODEL_COLORS.get(model, None) if colored == True else colored
        )


def plot_scope_minx_performance(
        ax, 
        root: Path, 
        algorithm: str, 
        metric: str, 
        level: Literal["fold", "superfamily"],
        model_prefix: str = "", 
        relative: bool = True, 
        colored: bool | str = True, 
        models: list[str] = MODELS
    ) -> None:
    """
    Plot performance metrics for different models on a given axis for the scope_minx task.

    Args:
        ax: Matplotlib axis to plot on.
        root: Root directory containing the embeddings and results.
        algorithm: Algorithm used (e.g., "lr", "knn").
        metric: Performance metric to plot (e.g., "pearson", "mcc").
        level: Level of the task.
        model_prefix: Prefix to add to model names. Either "" or "empty_" to indicate using normal or untrained models.
        relative: Whether to plot relative layer positions.
        colored: Color setting for the plot lines.
        models: List of model names to include in the plot.
    """
    for model in models:
        perfs = [compute_scope_performance(root, model_prefix + model, "scope_40_208", layer, algorithm, metric, level, min_x=10) for layer in range(LAYERS[model] + 1)]
        if sum([abs(p) for p in perfs]) == 0:  # drop performances that are 0 throughout
            continue
        ax.plot(
            np.arange(0, 1 + 1e-5, 1 / (LAYERS[model])) if relative else np.arange(LAYERS[model] + 1), 
            perfs, 
            label=model_prefix + MODEL_NAMES.get(model, model), 
            color=MODEL_COLORS.get(model, None) if colored == True else colored, 
            marker=MODEL_MARKERS.get(model, None) if colored == True else None
        )


def plot_scope_minx_metric(
        ax, 
        root: Path, 
        metric: str, 
        level: str, 
        model_prefix: str = "", 
        relative: bool = True, 
        models: list[str] = MODELS,
        grouped: bool = False,
        data: dict | None = None,
    ):
    all_perfs = []
    for model in models:
        if data is None:
            perfs = []
            for layer in range(LAYERS[model] + 1):
                if metric.startswith("noverlap") and layer == LAYERS[model]:
                    continue
                if metric in {"zero", "pc@95", "var@10", "5dvol"}:
                    result = read_pca_metric(root, model_prefix + model, "scope_40_208", layer, metric, f"pca_{level}_min10.pkl")
                else:
                    result = read_scope_metric(root, model_prefix + model, layer, metric, f"{metric}_{level}_min10.csv")
                perfs.append(result)
        else:
            perfs = data[model][f"scope_40_208_{level}"]
        if sum(perfs) < 2:
            continue
        all_perfs.append(perfs)
        if not grouped:
            if relative:
                x_ticks = np.arange(0, 1 + 1e-5, 1 / (LAYERS[model]))
                if metric.startswith("noverlap"):
                    x_ticks = x_ticks[:-1]
                    x_ticks += 1 / (2 * LAYERS[model])
                    perfs = perfs[:LAYERS[model]]
                ax.plot(x_ticks, perfs, label=model_prefix + MODEL_NAMES.get(model, model), color=MODEL_COLORS.get(model, None), marker=MODEL_MARKERS.get(model, None))
            else:
                ax.plot(perfs, label=model_prefix + MODEL_NAMES.get(model, model), color=MODEL_COLORS.get(model, None), marker=MODEL_MARKERS.get(model, None))
    
    if metric == "5dvol":
        ax.set_yscale("log")

    if grouped:
        interp_perfs = interpolate_data([all_perfs])[0]
        ax.plot(
            np.linspace(0, 1, 1000), 
            np.nanmean(interp_perfs, axis=0), 
            label=model_prefix + MODEL_NAMES.get(model, model), 
            c=MODEL_COLORS.get(model, None), 
            marker=MODEL_MARKERS.get(model, None)
        )
        ax.fill_between(
            np.linspace(0, 1, 1000), 
            np.nanmin(interp_perfs, axis=0), 
            np.nanmax(interp_perfs, axis=0), 
            alpha=0.2, 
            color=MODEL_COLORS.get(model, None)
        )


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
                label = "—"  # "NaN"
                text_color = "#666666"
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
    fine_labels  = ["bin.", "reg.", "", "Species", r"$T_\text{m}$", "Rocklin", "Tsuboy.", "", "binary", "10-class", "fold", "superf.", "3c SSP", "8c SSP", ""]
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
    depth = 0.033
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
    depth = 0.075
    trans = ax.get_xaxis_transform()  # data-x, axes-fraction-y
    for s, e in zip(snd_starts, snd_ends):
        if s == e:
            continue  # skip single-column groups (no line needed)
        line = mlines.Line2D([s - inset, e + inset], [-depth, -depth],
                            transform=trans, clip_on=False,
                            color='black', linewidth=1.2,
                            solid_capstyle='butt')
        ax.add_line(line)


def set_subplot_label(ax: plt.Axes, fig: plt.Figure, label: str) -> None:
    """
    Set the label for a subplot.
    Args:
        ax: The subplot
        fig: The figure
        label: The label to set
    """
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + transforms.ScaledTranslation(
            -25 / 72,
            10 / 72,
            fig.dpi_scale_trans
        ),
        fontsize="x-large",
        va="bottom",
        fontfamily="serif",
    )