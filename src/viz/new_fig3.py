from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

from src.viz.constants import DATASET_COLORS, DATASET_NAMES, LAYERS, MODEL_NAMES
from src.viz.plot_utils import set_subplot_label


PARITY_TOL = 0.02
PARITY_COLOR = "#BBBBBB"
MODEL_MARKERS = {
    "esm2_650m": "o",
    "prott5": "s",
    "progen2_medium": "^",
    "protgpt2": "D",
}
MODEL_ORDER = ["esm2_650m", "prott5", "progen2_medium", "protgpt2"]
DATASET_ORDER = ["fluorescence", "homology", "deeploc", "meltome", "solubility", "stability"]


def despine(ax, extra: tuple[str, ...] = ()) -> None:
    for side in ("top", "right", *extra):
        ax.spines[side].set_visible(False)


def _model_shape_legend_handles(model_order=MODEL_ORDER, color="#333333"):
    """Model legend keyed only by marker shape (colour is used for dataset)."""
    return [
        Line2D(
            [], [],
            marker=MODEL_MARKERS[m], linestyle="none",
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.4,
            label=f"{MODEL_NAMES[m]}, {LAYERS[m]} layers",
        )
        for m in model_order
    ]


def _lighten(hex_color, amount=0.5):
    """Blend a hex colour toward white by `amount` (0 = unchanged, 1 = white)."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def load_bench() -> pd.DataFrame:
    perf = pd.read_csv("plmsommelier_data/layer_performance.csv")
    size = pd.read_csv("plmsommelier_data/checkpoint_size.csv")
    thr = pd.read_csv("plmsommelier_data/throughput.csv")
    sel = pd.read_csv("plmsommelier_data/selection_time.csv")

    key = ["model", "dataset"]
    df = perf.merge(size, on=key, how="left", suffixes=("", "_size"))
    df = df.merge(thr, on=key, how="left", suffixes=("", "_thr"))
    df = df.merge(
        sel[
            [
                "model",
                "dataset",
                "task",
                "n_train",
                "n_val",
                "model_load_s",
                "embed_train_s",
                "embed_val_s",
                "probe_select_s",
                "selection_total_s",
                "embed_seqs_per_s",
            ]
        ],
        on=key,
        how="left",
        suffixes=("", "_sel"),
    )

    # --- derived columns -----------------------------------------------
    df["has_throughput"] = df["speedup"].notna()
    df["parity"] = df["rel_gain"].abs() <= PARITY_TOL
    df["degenerate"] = df["best_layer"] == 0
    df["unstable_selection"] = df["seed_agreement"] < 0.5
    # `confidence` is a newer, plateau-aware column that isn't in the CSVs
    # this benchmark was captured from -- kept optional so old data still
    # loads. Where present it supersedes `unstable_selection` above, which
    # exact-matches seed picks against `best_layer` and can call a run
    # "unstable" when every seed actually agreed on the tied region.
    if "confidence" in df:
        df["low_confidence"] = df["confidence"].eq("low")

    # Break-even sequence count: how many sequences you must process before
    # the one-time selection cost is repaid by the per-sequence inference
    # saving. NaN where throughput wasn't benchmarked; +inf where the
    # per-10k saving is ~0 or negative (never breaks even).
    per10k_saving = df["full_s_per_10k"] - df["trunc_s_per_10k"]
    with np.errstate(divide="ignore", invalid="ignore"):
        break_even = df["selection_total_s"] / (per10k_saving / 1e4)
    break_even = break_even.where(per10k_saving > 1e-9, np.inf)
    break_even = break_even.where(df["has_throughput"], np.nan)
    df["break_even_n"] = break_even

    df["model_label"] = df["model"]
    df["case"] = df["model"] + " / " + df["dataset"]

    return df


def plot_new_fig3():
    df = load_bench()
    d = df.copy()
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.32},
    )

    # -- panel a: Pareto payoff plane --------------------------------------
    ax.axhspan(-PARITY_TOL * 100, PARITY_TOL * 100, color=PARITY_COLOR, alpha=0.35, linewidth=0, zorder=0)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax.axvline(1, color="black", linewidth=0.5, linestyle=":", zorder=1)

    has = d["has_throughput"] & ~d["degenerate"]
    MARKER_SIZE = 150

    # Label offset is up-and-right by default; a few points sit close enough
    # together (mostly the near-x=1 cluster) that the tag is nudged to a
    # nearby clear spot instead. (dx, dy, ha, va) in points.
    LABEL_OFFSETS = {
        ("esm2_650m", "solubility"): (-5, -6, "right", "top"),
        ("esm2_650m", "meltome"): (5, -9, "left", "top"),
        ("prott5", "solubility"): (6, 8, "left", "bottom"),
    }

    for idx, row in d[has].iterrows():
        color = DATASET_COLORS[row["dataset"]]
        marker = MODEL_MARKERS[row["model"]]
        ax.scatter(
            row["speedup"], row["rel_gain"] * 100, s=MARKER_SIZE,
            facecolor=color, edgecolor="white", linewidth=0.4,
            marker=marker, alpha=0.9, zorder=3,
        )
        dx, dy, ha, va = LABEL_OFFSETS.get((row["model"], row["dataset"]), (5, 5, "left", "bottom"))
        ax.annotate(
            f"l={int(row['best_layer'])}",
            xy=(row["speedup"], row["rel_gain"] * 100), xytext=(dx, dy),
            textcoords="offset points", ha=ha, va=va, color="#333333",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.9, 8)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7"])
    ax.set_xlabel("Inference speed-up")
    ax.set_ylabel("Relative improvement over the last layer [%]")
    despine(ax)
    set_subplot_label(ax, fig, "A")

    ax.text(1.75, 0.7, "parity band (±2%)", va="center", ha="left", style="italic", color="#444444")

    ax.legend(handles=_model_shape_legend_handles(MODEL_ORDER), loc="upper left", frameon=False, bbox_to_anchor=(0.05, 1.0), markerscale=1.8)

    # -- panel b: mean time & space saved per dataset, averaged over models -
    time_saved_pct = (df["full_s_per_10k"] - df["trunc_s_per_10k"]) / df["full_s_per_10k"] * 100
    df_b = df.assign(time_saved_pct=time_saved_pct)

    space_by_ds = df_b.groupby("dataset")["pct_saved"].mean().reindex(DATASET_ORDER)
    time_by_ds = df_b[df_b["has_throughput"]].groupby("dataset")["time_saved_pct"].mean().reindex(DATASET_ORDER)

    y = np.arange(len(DATASET_ORDER))
    bar_h = 0.35
    ds_colors = [DATASET_COLORS[ds] for ds in DATASET_ORDER]
    ds_colors_light = [_lighten(c, 0.6) for c in ds_colors]
    # y - bar_h/2 sits above y + bar_h/2 once the axis is inverted below, so
    # this (time, hatched) bar is the one drawn on top within each pair.
    ax_bar.barh(
        y - bar_h / 2, time_by_ds.values, height=bar_h, facecolor=ds_colors_light,
        edgecolor=ds_colors, linewidth=0.6, hatch="////",
    )
    ax_bar.barh(
        y + bar_h / 2, space_by_ds.values, height=bar_h, facecolor=ds_colors,
        edgecolor="white", linewidth=0.6,
    )
    metric_handles = [
        Patch(facecolor="#AAAAAA", edgecolor="#777777", hatch="////", label="Inference time"),
        Patch(facecolor="#777777", edgecolor="white", label="Checkpoint size"),
    ]

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([DATASET_NAMES[ds] for ds in DATASET_ORDER])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean inference time/size fraction of truncated models")
    ax_bar.set_xlim(0, 100)
    despine(ax_bar, extra=("left",))
    ax_bar.tick_params(axis="y", length=0)
    set_subplot_label(ax_bar, fig, "B")
    ax_bar.legend(handles=metric_handles, loc="lower right", frameon=False)

    plt.savefig("paper_figures/variant2_pareto_plane.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_new_fig3()
