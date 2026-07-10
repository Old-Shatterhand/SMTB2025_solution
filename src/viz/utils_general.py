import pickle
from pathlib import Path
from typing import Literal

from matplotlib import pyplot as plt, transforms
import scipy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef, roc_auc_score
import torch

from src.viz.constants import MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, MODELS, SPLIT_ID, LAYERS, DATASET2TASK
from src.downstream.utils import multioutput_mcc, multiclass_mcc


FINETUNE_LAYERS = [0, 10, 15, 20, 22, 24, 26, 28, 30]





def comp_finetune_performance(base, dataset, metric):
    perfs = []
    for layer in FINETUNE_LAYERS:
        with open(base / "semifrozen_esm" / dataset / f"esm_t30" / f"unfreeze_{layer}" / "lr_1e-4" / f"predictions_unfrozen_esm_t30_{layer}_end_0.0001.pkl", "rb") as f:
            y_hat, y = pickle.load(f)[1]
        y = np.array(y).squeeze()
        y_hat = np.array(y_hat).squeeze()
        if DATASET2TASK[dataset] == "multi-label":
            y_hat = scipy.special.expit(y_hat)
        perfs.append(compute_metric(y_hat, y, metric, DATASET2TASK[dataset]))
    return perfs


def plot_dataset_finetune_comparison(ax, root, dataset, metric, n_classes, task):
    try:
        layer_perfs = [compute_performance(root, "esm_t30", dataset, layer, algo="knn", metric=metric, aa=False, n_classes=n_classes, task=task) for layer in range(LAYERS["esm_t30"] + 1)]
        ax.plot(
            np.arange(len(layer_perfs)),
            layer_perfs, 
            label="layer-trained", 
            c=MODEL_COLORS.get("esm_t30", None),
            marker=MODEL_MARKERS.get("esm_t30", None),
        )

        ft_perfs = comp_finetune_performance(root, dataset, metric)
        ax.plot(
            FINETUNE_LAYERS,
            ft_perfs, 
            label="finetuned", 
            c="orange",
            marker=MODEL_MARKERS.get("esm_t30", None),
        )
    except:
        pass
    # full, lora = fetch_rost(dataset, "esm_t30")
    # ax.plot([0, 30], [lora, lora], label="LoRA", c="salmon", linestyle="--")
    # ax.plot([0, 30], [full, full], label="Full", c="darkred", linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel({"r2": "$R^2$", "mcc": "MCC"}.get(metric, metric))


def fetch_rost(dataset, model):
    data_to_rost = {
        "fluorescence": "GFP",
        "stability": "Stability",
        "deeploc2": "Subcellular",
        "meltome_atlas": "Meltome",
    }
    model_to_rost = {
        "esm_t6": "ESM2 8M",
        "esm_t12": "ESM2 35M",
        "esm_t30": "ESM2 150M",
        "esm_t33": "ESM2 650M",
        "esm_t36": "ESM2 3B",
        "ankh_base": "Ankh base",
        "ankh_large": "Ankh large",
        "prott5": "ProtT5",
    }
    assert dataset in data_to_rost, f"Dataset {dataset} not found in ROST table mapping."
    assert model in model_to_rost, f"Model {model} not found in ROST table mapping."

    pt = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB" / "rost_results" / "Table_S1_Individual_training runs_pre_trained_embeddings.csv"
    ft = pt.parent / "Table_S2_Individual_training_runs_fine_tuning.csv"
    df_pt = pd.read_csv(pt, sep=";")
    df_ft = pd.read_csv(ft, sep=";")

    pt_val = np.mean([float(x[:-1].replace(",", ".")) for x in df_pt[df_pt["Model"] == model_to_rost[model]][data_to_rost[dataset]].values[:3]])
    lora_val = np.mean([float(x[:-1].replace(",", ".")) for x in df_ft[df_ft["Model"] == model_to_rost[model]][data_to_rost[dataset]].values[:3]])

    return pt_val / 100, lora_val / 100
