import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from src.viz.utils import XP, compute_metric, compute_performance
from src.viz.constants import CLASS_METRIC, LAYERS, MODEL_COLORS, MODEL_MARKERS, REG_METRIC, MODELS, DATATASK_NAMES, DATASET2TASK, MODEL_NAMES
from src.viz.plot_utils import set_subplot_label

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

def scale_data(data):
    scaled = []
    for y in data:
        x = np.arange(0, 1 + 1e-5, 1 / (len(y) - 1))
        yp = np.interp(XP, x, y)
        scaled.append(yp)
    return scaled

def plot_fig6():
    stab = pd.read_csv(BASE / "datasets" / "stability_random.csv")
    art_map = dict(stab[["ID", "artificial"]].values)

    art, non_art, data = [], [], []
    for model in MODELS:
        if model.startswith("ankh"):
            continue
        tmp_dms, tmp_non_dms, tmp_data = [], [], []
        for layer in range(LAYERS[model] + 1):
            perf, _, _ = compute_performance(BASE, model, "stability_random", layer, algo="knn", metric="macro-pearson", id_cls_map=art_map, no_mean=True)
            tmp_non_dms.append(perf[0])
            tmp_dms.append(perf[1])
            tmp_data.append(compute_performance(BASE, model, "stability_random", layer, algo="knn", metric="pearson"))
        non_art.append(tmp_non_dms)
        art.append(tmp_dms)
        data.append(tmp_data)
    new_dms = scale_data(art)
    new_non_dms = scale_data(non_art)
    new_data = scale_data(data)

    data_art = [[compute_performance(BASE, model, "lysosomes", layer, algo="lr", metric="pearson") for layer in range(LAYERS[model] + 1)] for model in MODELS]
    data_wild = [[compute_performance(BASE, model, "lysosomes_natural", layer, algo="lr", metric="pearson") for layer in range(LAYERS[model] + 1)] for model in MODELS]

    wild_vals = np.array(scale_data(data_wild))
    art_vals = np.array(scale_data(data_art))

    tsuboyama_df = pd.read_csv(BASE / "datasets" / "tsuboyama.csv")
    tsuboyama_id_cls_map = dict(tsuboyama_df[["ID", "source"]].values)

    data = []
    for model in list(filter(lambda x: not x.startswith("ankh"), MODELS)):
        if model.startswith("ankh"):
            continue
        data.append([compute_performance(BASE, model, "tsuboyama", layer, algo="knn", metric="pearson", id_cls_map=tsuboyama_id_cls_map) for layer in range(LAYERS[model] + 1)])
    tsuboyama_data = np.array(scale_data(data))

    data = []
    for model in list(filter(lambda x: not x.startswith("ankh"), MODELS)):
        data.append([compute_performance(BASE, model, "solubility_ds50_3", layer, algo="lr", metric=CLASS_METRIC) for layer in range(LAYERS[model])])
    sd = scale_data(data)


    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 2)
    axs = [fig.add_subplot(gs[i]) for i in range(2)]

    for ax_id, data, label in [(0, new_non_dms, "Rocklin natural"), (0, new_dms, "Rocklin artificial"), (0, new_data, "Rocklin combined"), (0, tsuboyama_data, "Tsuboyama"), (1, wild_vals, "Natural lysozymes"), (1, art_vals, "Artificial lysozymes")]:  # , (2, sd, "DeepSol50")]:  # , (2, tsuboyama_data, "Tsuboyama")]:
        axs[ax_id].plot(XP, np.nanmean(data, axis=0), label=label, linewidth=5)
        axs[ax_id].fill_between(
            XP, 
            np.nanmin(data, axis=0), 
            np.nanmax(data, axis=0), 
            alpha=0.2
        )

    axs[0].set_ylabel("Pearson's r")
    for i in range(2):
        axs[i].set_xlabel("Relative layer")
        axs[i].grid()
        axs[i].legend()
        set_subplot_label(axs[i], fig, chr(65 + i))

    plt.tight_layout()  # rect=[0, 0.075, 1, 1])
    plt.savefig("paper_figures/fig_6_dms.pdf", bbox_inches="tight", dpi=300)


def plot_fig6_smtb():
    stab = pd.read_csv(BASE / "datasets" / "stability_random.csv")
    art_map = dict(stab[["ID", "artificial"]].values)
    dms_map = dict(stab[["ID", "dms"]].values)
    stab_map = {idx: 0 if not art_map[idx] else (1 if not dms_map[idx] else 2) for idx in art_map.keys()}

    art, dms, art_dms, data = [], [], [], []
    for model in MODELS:
        if model.startswith("ankh"):
            continue
        tmp_art, tmp_dms, tmp_art_dms, tmp_data = [], [], [], []
        for layer in range(LAYERS[model] + 1):
            perf, _, clss = compute_performance(BASE, model, "stability_random", layer, algo="lr", metric="macro-pearson", id_cls_map=stab_map, no_mean=True)
            tmp_art.append(perf[clss.index(1)])
            tmp_dms.append(perf[clss.index(0)])
            tmp_art_dms.append(perf[clss.index(2)])
            tmp_data.append(compute_performance(BASE, model, "stability_random", layer, algo="lr", metric="pearson"))
        art.append(tmp_art)
        dms.append(tmp_dms)
        art_dms.append(tmp_art_dms)
        data.append(tmp_data)
    new_art = scale_data(art)
    new_dms = scale_data(dms)
    new_art_dms = scale_data(art_dms)
    new_data = scale_data(data)

    data_art = [[compute_performance(BASE, model, "lysosomes", layer, algo="lr", metric="pearson") for layer in range(LAYERS[model] + 1)] for model in MODELS]
    data_wild = [[compute_performance(BASE, model, "lysosomes_natural", layer, algo="lr", metric="pearson") for layer in range(LAYERS[model] + 1)] for model in MODELS]

    wild_vals = np.array(scale_data(data_wild))
    art_vals = np.array(scale_data(data_art))

    tsuboyama_df = pd.read_csv(BASE / "datasets" / "tsuboyama.csv")
    tsuboyama_id_cls_map = dict(tsuboyama_df[["ID", "source"]].values)

    data = []
    for model in list(filter(lambda x: not x.startswith("ankh"), MODELS)):
        if model.startswith("ankh"):
            continue
        data.append([compute_performance(BASE, model, "tsuboyama", layer, algo="knn", metric="pearson", id_cls_map=tsuboyama_id_cls_map) for layer in range(LAYERS[model] + 1)])
    tsuboyama_data = np.array(scale_data(data))

    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 2)
    axs = [fig.add_subplot(gs[i]) for i in range(2)]

    for ax_id, data, label in [(0, new_dms, "Rocklin nat. DMS"), (0, new_art, "Rocklin artificial"), (0, new_art_dms, "Rocklin art. DMS"), (0, new_data, "Rocklin combined"), (0, tsuboyama_data, "Tsuboyama"), (1, wild_vals, "Natural Lysozymes"), (1, art_vals, "Artificial Lysozymes")]:  # , (2, sd, "DeepSol50")]:  # , (2, tsuboyama_data, "Tsuboyama")]:
        axs[ax_id].plot(XP, np.nanmean(data, axis=0), label=label, linewidth=5)
        axs[ax_id].fill_between(
            XP, 
            np.nanmin(data, axis=0), 
            np.nanmax(data, axis=0), 
            alpha=0.2
        )

    axs[0].set_ylabel("Pearson's r")
    for i in range(2):
        axs[i].set_xlabel("Relative layer")
        axs[i].grid()
        axs[i].legend()
        set_subplot_label(axs[i], fig, chr(65 + i))

    plt.tight_layout()  # rect=[0, 0.075, 1, 1])
    plt.savefig("paper_figures/smtb_3.png", bbox_inches="tight", dpi=300)
