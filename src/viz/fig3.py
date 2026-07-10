from collections import defaultdict
from pathlib import Path
import pickle

from matplotlib import gridspec, pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.viz.constants import CLASS_METRIC, DATASET2TASK, DATASET_NAMES, METRIC_TITLES, MODEL_COLORS, MODEL_MARKERS, MODELS, REG_METRIC, TASK_METRICS
from src.viz.plot_utils import plot_performance, set_subplot_label
from src.viz.utils import compute_metric


BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"
RATIOS = [0.01, 0.05, 0.1, 0.15, 0.2]

def load_full_data():
    full_data = {}
    for layers in [6, 12, 30, 33, 36]:
        full_data[f"esm_t{layers}"] = defaultdict(list)
        for dataset in ["fluorescence", "fluorescence_classification", "deeploc2", "deeploc2_bin", "stability", "solubility", "meltome_atlas", "scope_40_208"]:
            metric = "r2" if DATASET2TASK[dataset] == "regression" else "mcc"
            task = DATASET2TASK[dataset]
            for layer in range(layers + 1):
                print(f"\rLoading esm_t{layers} - {dataset} - layer {layer}", end=" " * 25)
                if dataset != "scope_40_208":
                    with open(BASE / "embeddings" / f"esm_t{layers}" / dataset / f"layer_{layer}" / f"predictions_knn.pkl", "rb") as f:
                        y_hat, y = pickle.load(f)[1]
                    full_data[f"esm_t{layers}"][dataset].append(compute_metric(np.array(y_hat), np.array(y), metric=metric, task=task))
                else:
                    for level in ["fold", "superfamily"]:
                        with open(BASE / "embeddings" / f"esm_t{layers}" / dataset / f"layer_{layer}" / f"predictions_knn_{level}_min10.pkl", "rb") as f:
                            y_hat, y = pickle.load(f)[1]
                        full_data[f"esm_t{layers}"][f"{dataset}_{level}"].append(compute_metric(np.array(y_hat), np.array(y), metric=metric, task=task))
    return full_data


def load_sampled_data():
    data = {}
    for layers in [6, 12, 30, 33, 36]:
        data[f"esm_t{layers}"] = {}
        for dataset in ["fluorescence", "fluorescence_classification", "deeploc2", "deeploc2_bin", "stability", "solubility", "meltome_atlas", "scope_40_208_fold", "scope_40_208_superfamily"]:
            data[f"esm_t{layers}"][dataset] = {}
            metric = "r2" if DATASET2TASK["_".join(dataset.split("_")[:3])] == "regression" else "mcc"
            task = DATASET2TASK["_".join(dataset.split("_")[:3])]
            for ratio in RATIOS:
                data[f"esm_t{layers}"][dataset][ratio] = {}
                for seed in [1234, 42, 7331]:
                    data[f"esm_t{layers}"][dataset][ratio][seed] = []
                    for layer in range(layers + 1):
                        print(f"\rLoading esm_t{layers} - {dataset} - ratio {ratio} - layer {layer} - seed {seed}", end=" " * 25)
                        if not dataset.startswith("scope_40_208"):
                            try:
                                with open(fn := (BASE / "embeddings" / f"esm_t{layers}" / dataset / f"layer_{layer}" / f"predictions_knn_max{ratio}_{seed}.pkl"), "rb") as f:
                                    y_hat, y = pickle.load(f)[1]
                                data[f"esm_t{layers}"][dataset][ratio][seed].append(compute_metric(np.array(y_hat), np.array(y), metric=metric, task=task))
                            except FileNotFoundError:
                                print("\rFile not found:", fn)
                                data[f"esm_t{layers}"][dataset][ratio][seed].append(np.nan)
                        else:
                            level = "fold" if dataset.endswith("fold") else "superfamily"
                            try:
                                with open(fn := (BASE / "embeddings" / f"esm_t{layers}" / "scope_40_208" / f"layer_{layer}" / f"predictions_knn_max{ratio}_{seed}_{level}_min10.pkl"), "rb") as f:
                                    y_hat, y = pickle.load(f)[1]
                                data[f"esm_t{layers}"][dataset][ratio][seed].append(compute_metric(np.array(y_hat), np.array(y), metric=metric, task=task))
                            except FileNotFoundError:
                                print("\rFile not found:", fn)
                                data[f"esm_t{layers}"][dataset][ratio][seed].append(np.nan)
    return data


def diff_best(ax, dataset, model, sampled_data, full_data):
    metric = []
    for ratio in RATIOS:
        metric.append([])
        for seed in [1234, 42, 7331]:
            idx = np.argmax(sampled_data[model][dataset][ratio][seed])
            metric[-1].append(full_data[model][dataset][idx] / np.max(full_data[model][dataset]))
        metric[-1] = np.mean(metric[-1])
    ax.plot(RATIOS, metric, label=model, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model])
    ax.set_xticks(RATIOS)


def correlate(ds1, ds2):
    coeffs = []
    for model in ds1.keys():
        coeffs.append(np.corrcoef(np.array(ds1[model]), np.array(ds2[model]))[0, 1])
        print(f"\t{model}: {coeffs[-1]:.3f}")
    print(f"Mean correlation: {np.nanmean(coeffs):.3f} ± {np.nanstd(coeffs):.3f}")


def plot_fig3_old(models):
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2)
    gs_left = gs[0].subgridspec(4, 2, wspace=0.2)
    gs_right = gs[1].subgridspec(4, 2, wspace=0.2)
    axs = [fig.add_subplot(gs_left[i, j]) for i in range(4) for j in range(2)] + [fig.add_subplot(gs_right[i, j]) for i in range(4) for j in range(2)]

    fcls_perfs = plot_performance(axs[0], BASE, "fluorescence_classification", "knn", "mcc", relative=True, task="binary", models=models)
    axs[0].set_title("Fluorescence Binary")
    axs[0].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["fluorescence_classification"]]])
    set_subplot_label(axs[0], fig, "A")
    freg_perfs = plot_performance(axs[1], BASE, "fluorescence", "knn", "r2", relative=True, task="regression", models=models)
    axs[1].set_title("Fluorescence Regression")
    axs[1].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["fluorescence"]]])
    set_subplot_label(axs[1], fig, "B")

    dbin_perfs = plot_performance(axs[2], BASE, "deeploc2_bin", "knn", "mcc", relative=True, task="binary", models=models)
    axs[2].set_title("DeepLoc2.0 Binary")
    axs[2].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["deeploc2_bin"]]])
    set_subplot_label(axs[2], fig, "C")
    d10_perfs = plot_performance(axs[3], BASE, "deeploc2", "knn", "mcc", relative=True, task="multi-label", models=models)
    axs[3].set_title("DeepLoc2.0 10-class")
    axs[3].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["deeploc2"]]])
    set_subplot_label(axs[3], fig, "D")

    ma_temp = plot_performance(axs[4], BASE, "meltome_atlas_species", "knn", "mcc", relative=True, task="multi-class", models=models)
    axs[4].set_title("Meltome Atlas Species")
    axs[4].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["meltome_atlas_species"]]])
    set_subplot_label(axs[4], fig, "E")
    ma_species = plot_performance(axs[5], BASE, "meltome_atlas", "knn", "r2", relative=True, task="regression", models=models)
    axs[5].set_title("Meltome Atlas Temperature")
    axs[5].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["meltome_atlas"]]])
    set_subplot_label(axs[5], fig, "F")

    ssp3_perfs = plot_performance(axs[6], BASE, "scope_40_208", "knn", "mcc", relative=True, aa=True, n_classes=3, task="multi-class", models=models)
    axs[6].set_title("SCOPe40 3-class SSP")
    axs[6].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["scope_40_208_3ssp"]]])
    set_subplot_label(axs[6], fig, "G")
    ssp8_perfs = plot_performance(axs[7], BASE, "scope_40_208", "knn", "mcc", relative=True, aa=True, n_classes=8, task="multi-class", models=models)
    axs[7].set_title("SCOPe40 8-class SSP")
    axs[7].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["scope_40_208_8ssp"]]])
    set_subplot_label(axs[7], fig, "H")

    if (fp := Path("paper_figures/sampled.pkl")).exists():
        with open(fp, "rb") as f:
            full_data, sampled_data = pickle.load(f)
    else:
        full_data = load_full_data()
        sampled_data = load_sampled_data()
        with open(fp, "wb") as f:
            pickle.dump((full_data, sampled_data), f)

    scale = {x: x + 8 for x in range(8)}  # dict(enumerate([2, 3, 6, 7, 10, 11]))
    for d, dataset in enumerate(["fluorescence_classification", "fluorescence", "deeploc2_bin", "deeploc2", "meltome_atlas", "stability", "scope_40_208_superfamily", "solubility"]):
        for model in ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36"]:
            diff_best(axs[scale[d]], dataset, model, sampled_data, full_data)
        axs[scale[d]].set_title(DATASET_NAMES[dataset])
        set_subplot_label(axs[scale[d]], fig, label=f"{chr(ord('A') + d + 8)}")
        if d % 2 == 0:
            axs[scale[d]].set_ylabel("Relative-to-best performance")

    for i in range(12):
        axs[i].grid()
        if i not in {6, 7, 14, 15}:
            axs[i].tick_params(axis='x', labelbottom=False, bottom=False)
            axs[i].set_xlabel("")
        else:
            axs[i].set_xlabel("Ratio of training data")
        if i in {1, 3, 7}:
            axs[i].sharey(axs[i - 1])

    sp_handles, sp_labels = axs[9].get_legend_handles_labels()
    handles, labels = axs[0].get_legend_handles_labels()
    handles = sp_handles + handles[2:]
    labels = sp_labels + labels[2:]
    handles.insert(7, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    labels.insert(7, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), bbox_transform=fig.transFigure, ncol=len(handles) // 2 + 1)  # -0.08

    plt.tight_layout()
    plt.savefig("paper_figures/3_lp_ablation.pdf", dpi=300, bbox_inches="tight")


def plot_fig3(models, algo: str = "knn"):
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2)
    gs_left = gs[0].subgridspec(2, 2, wspace=0.2)
    gs_right = gs[1].subgridspec(2, 2, wspace=0.2)
    axs = [fig.add_subplot(gs_left[i, j]) for i in range(2) for j in range(2)] + [fig.add_subplot(gs_right[i, j]) for i in range(2) for j in range(2)]

    plot_performance(axs[0], BASE, "fluorescence_classification", algo, CLASS_METRIC, relative=True, task="binary", models=models)
    axs[0].set_title(DATASET_NAMES["fluorescence_classification"])
    axs[0].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["fluorescence_classification"]]])
    plot_performance(axs[1], BASE, "fluorescence", algo, REG_METRIC, relative=True, task="regression", models=models)
    axs[1].set_title(DATASET_NAMES["fluorescence"])
    axs[1].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["fluorescence"]]])

    plot_performance(axs[2], BASE, "scope_40_208", algo, CLASS_METRIC, relative=True, aa=True, n_classes=3, task="multi-class", models=models)
    axs[2].set_title(DATASET_NAMES["scope_40_208_3ssp"])
    axs[2].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["scope_40_208_3ssp"]]])
    plot_performance(axs[3], BASE, "scope_40_208", algo, CLASS_METRIC, relative=True, aa=True, n_classes=8, task="multi-class", models=models)
    axs[3].set_title(DATASET_NAMES["scope_40_208_8ssp"])
    axs[3].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["scope_40_208_8ssp"]]])

    if (fp := Path("paper_figures/sampled.pkl")).exists():
        with open(fp, "rb") as f:
            full_data, sampled_data = pickle.load(f)
    else:
        full_data = load_full_data()
        sampled_data = load_sampled_data()
        with open(fp, "wb") as f:
            pickle.dump((full_data, sampled_data), f)

    scale = {x: x + 4 for x in range(4)}  # dict(enumerate([2, 3, 6, 7, 10, 11]))
    for d, dataset in enumerate(["fluorescence", "stability", "deeploc2", "scope_40_208_fold"]):
        for model in ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36"]:
            diff_best(axs[scale[d]], dataset, model, sampled_data, full_data)
        axs[scale[d]].set_title(DATASET_NAMES[dataset])
        if d % 2 == 0:
            axs[scale[d]].set_ylabel("Relative-to-best performance")

    for i in range(8):
        axs[i].grid()
        if i not in {2, 3, 6, 7}:
            axs[i].tick_params(axis='x', labelbottom=False, bottom=False)
            axs[i].set_xlabel("")
        else:
            axs[i].set_xlabel("Relative layer" if i < 4 else "Ratio of training data")
        if i in {1, 3}:
            axs[i].sharey(axs[i - 1])
        set_subplot_label(axs[i], fig, label=f"{chr(ord('A') + i)}")

    sp_handles, sp_labels = axs[5].get_legend_handles_labels()
    handles, labels = axs[0].get_legend_handles_labels()
    handles = sp_handles + handles[2:]
    labels = sp_labels + labels[2:]
    handles.insert(7, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    labels.insert(7, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), bbox_transform=fig.transFigure, ncol=len(handles) // 2 + 1)  # -0.08

    plt.tight_layout()
    plt.savefig("paper_figures/3_lp_ablation.pdf", dpi=300, bbox_inches="tight")


def plot_full_fig3_left(models: list[str] = MODELS, algo: str = "knn"):
    fig = plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.2)
    axs = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(2)]

    fcls_perfs = plot_performance(axs[0], BASE, "fluorescence_classification", algo, CLASS_METRIC, relative=True, task="binary", models=models)
    axs[0].set_title(DATASET_NAMES["fluorescence_classification"])
    axs[0].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["fluorescence_classification"]]])
    freg_perfs = plot_performance(axs[1], BASE, "fluorescence", algo, REG_METRIC, relative=True, task="regression", models=models)
    axs[1].set_title(DATASET_NAMES["fluorescence"])
    axs[1].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["fluorescence"]]])

    ma_temp = plot_performance(axs[2], BASE, "meltome_atlas", algo, REG_METRIC, relative=True, task="regression", models=models)
    axs[2].set_title(DATASET_NAMES["meltome_atlas"])
    axs[2].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["meltome_atlas"]]])
    ma_species = plot_performance(axs[3], BASE, "meltome_atlas_species", algo, CLASS_METRIC, relative=True, task="multi-class", models=models)
    axs[3].set_title(DATASET_NAMES["meltome_atlas_species"])
    axs[3].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["meltome_atlas_species"]]])

    dbin_perfs = plot_performance(axs[4], BASE, "deeploc2_bin", algo, CLASS_METRIC, relative=True, task="binary", models=models)
    axs[4].set_title(DATASET_NAMES["deeploc2_bin"])
    axs[4].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["deeploc2_bin"]]])
    d10_perfs = plot_performance(axs[5], BASE, "deeploc2", algo, CLASS_METRIC, relative=True, task="multi-label", models=models)
    axs[5].set_title(DATASET_NAMES["deeploc2"])
    axs[5].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["deeploc2"]]])

    ssp3_perfs = plot_performance(axs[6], BASE, "scope_40_208", algo, CLASS_METRIC, relative=True, aa=True, n_classes=3, task="multi-class", models=models)
    axs[6].set_title(DATASET_NAMES["scope_40_208_3ssp"])
    axs[6].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["scope_40_208_3ssp"]]])
    ssp8_perfs = plot_performance(axs[7], BASE, "scope_40_208", algo, CLASS_METRIC, relative=True, aa=True, n_classes=8, task="multi-class", models=models)
    axs[7].set_title(DATASET_NAMES["scope_40_208_8ssp"])
    axs[7].set_ylabel(METRIC_TITLES[TASK_METRICS[DATASET2TASK["scope_40_208_8ssp"]]])

    print("Correlating Fluorescence Binary and Fluorescence Regression:")
    correlate(fcls_perfs, freg_perfs)
    print("Correlating Meltome Atlas Species and Meltome Atlas Temperature:")
    correlate(ma_temp, ma_species)
    print("Correlating DeepLoc2.0 Binary and DeepLoc2.0 10-class:")
    correlate(dbin_perfs, d10_perfs)
    print("Correlating SCOPe40 3-class SSP and SCOPe40 8-class SSP:")
    correlate(ssp3_perfs, ssp8_perfs)

    for i in range(8):
        axs[i].grid()
        if i not in {6, 7}:
            axs[i].tick_params(axis='x', labelbottom=False, bottom=False)
            axs[i].set_xlabel("")
        else:
            axs[i].set_xlabel("Ratio of training data")
        if i in {1, 3, 5, 7}:
            axs[i].sharey(axs[i - 1])
        set_subplot_label(axs[i], fig, chr(ord("A") + i))


    handles, labels = axs[0].get_legend_handles_labels()
    # handles.insert(7, Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    # labels.insert(7, "")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), bbox_transform=fig.transFigure, ncol=len(handles) // 3 + 1)  # -0.08

    plt.tight_layout()
    plt.savefig("paper_figures/full_3_pairs.pdf", dpi=300, bbox_inches="tight")




def plot_full_fig3_right():
    if (fp := Path("paper_figures/sampled.pkl")).exists():
        with open(fp, "rb") as f:
            full_data, sampled_data = pickle.load(f)
    else:
        full_data = load_full_data()
        sampled_data = load_sampled_data()
        with open(fp, "wb") as f:
            pickle.dump((full_data, sampled_data), f)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.2, hspace=0.2)
    axs = [fig.add_subplot(gs[j, i]) for i in range(3) for j in range(3)]

    for d, dataset in enumerate(["fluorescence_classification", "fluorescence", "meltome_atlas", "deeploc2_bin", "deeploc2", "stability", "scope_40_208_fold", "scope_40_208_superfamily", "solubility"]):
        for model in ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36"]:
            diff_best(axs[d], dataset, model, sampled_data, full_data)
        axs[d].set_title(DATASET_NAMES[dataset])
        axs[d].grid()
        set_subplot_label(axs[d], fig, label=f"{chr(ord('A') + d)}")
        if d % 3 < 2:
            axs[d].set_xlabel("")
            axs[d].tick_params(axis='x', labelbottom=False, bottom=False)
        else:
            axs[d].set_xlabel("Ratio of training data")
        if d < 3:
            axs[d].set_ylabel("Relative performance to best layer")

    axs[0].legend()
    plt.tight_layout()
    plt.savefig("paper_figures/full_3_sparse.pdf", dpi=300, bbox_inches="tight")
