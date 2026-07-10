from pathlib import Path
from typing import Literal


def kill_axis(ax):
    ax.set_axis_off()

SPLIT = "valid"
SPLIT_ID = {"train": 0, "valid": 1, "test": 2}[SPLIT]

MODELS = ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh_base", "ankh_large", "prostt5", "prott5", "progen2_small", "progen2_medium", "progen2_large", "protgpt2"]
WP_DATASETS = ["fluorescence", "fluorescence_classification", "stability", "deeploc2", "deeploc2_bin", "meltome_atlas", "meltome_atlas_species", "scope_40_208", "solubility", "gb1"]
AA_DATASETS = ["binding", "scope_40_208"]
DATASETS = WP_DATASETS + AA_DATASETS

REG_METRIC = "pearson"
CLASS_METRIC = "mcc"

LAYERS = {
    "esm_t6": 6,
    "esm_t12": 12,
    "esm_t30": 30,
    "esm_t33": 33,
    "esm_t36": 36,
    "esmc_300m": 30,
    "esmc_600m": 36,
    "ankh_base": 48,
    "ankh_large": 48,
    "prostt5": 24,
    "prott5": 24,
    "progen2_small": 12,
    "progen2_medium": 27,
    "progen2_large": 32,
    "protgpt2": 36,
    "ohe": 0,
}

MODEL_COLORS = {
    "esm_fine": "navy",
    "esm_t6": "royalblue",
    "esm_t12": "royalblue",
    "esm_t30": "royalblue",
    "esm_t33": "royalblue",
    "esm_t36": "royalblue",
    "esmc_300m": "mediumblue",
    "esmc_600m": "mediumblue",
    "ankh_base": "green",
    "ankh_large": "green",
    "prott5": "darkorange",
    "prostt5": "darkorange",
    "progen2_small": "darkred",
    "progen2_medium": "darkred",
    "progen2_large": "darkred",
    "protgpt2": "violet",
    "ohe": "gray",
}

MODEL_MARKERS = {
    "esm_fine": "X",
    "esm_t6": "|",
    "esm_t12": "2",
    "esm_t30": "x",
    "esm_t33": "*",
    "esm_t36": "o",
    "esmc_300m": "x",
    "esmc_600m": "o",
    "ankh_base": "x",
    "ankh_large": "o",
    "prott5": "x",
    "prostt5": "o",
    "progen2_small": "|",
    "progen2_medium": "x",
    "progen2_large": "o",
    "protgpt2": "x",
    "ohe": "|",
}

MODEL_NAMES = {
    "esm_fine": "fine-tuned ESM-2 150M",
    "esm_t6": "ESM-2 8M",
    "esm_t12": "ESM-2 35M",
    "esm_t30": "ESM-2 150M",
    "esm_t33": "ESM-2 650M",
    "esm_t36": "ESM-2 3B",
    "esmc_300m": "ESMC 300M",
    "esmc_600m": "ESMC 600M",
    "ankh_base": "Ankh Base",
    "ankh_large": "Ankh Large",
    "prott5": "ProtT5",
    "prostt5": "ProstT5",
    "progen2_small": "ProGen2 Small",
    "progen2_medium": "ProGen2 Medium",
    "progen2_large": "ProGen2 Large",
    "protgpt2": "ProtGPT2",
    "ohe": "OHE",
}

DATASET_NAMES = {
    "fluorescence_classification": "Fluorescence Binary",
    "fluorescence": "Fluorescence Regression",
    "meltome_atlas": "Meltome Atlas Temperature",
    "meltome_atlas_species": "Meltome Atlas Species",
    "stability": "Stability",
    "deeploc2_bin": "DeepLoc2.0 Binary",
    "deeploc2": "DeepLoc2.0 10-class",
    "scope_40_208_fold": "SCOPe40 Fold",
    "scope_40_208_superfamily": "SCOPe40 Superfamily",
    "scope_40_208_3ssp": "SCOPe40 3-class SSP",
    "scope_40_208_8ssp": "SCOPe40 8-class SSP",
    "binding": "Binding",
    "solubility": "DeepSol",
    "gb1": "GB1",
}

DATASET2TASK: dict[str, Literal["regression", "binary", "multi-label", "multi-class"]] = {
    "fluorescence": "regression",
    "fluorescence_classification": "binary",
    "stability": "regression",
    "deeploc2": "multi-label",
    "deeploc2_bin": "binary",
    "meltome_atlas": "regression",
    "meltome_atlas_species": "multi-class",
    "binding": "binary",
    "scope_40_208": "multi-class",
    "scope_40_208_fold": "multi-class",
    "scope_40_208_superfamily": "multi-class",
    "scope_40_208_3ssp": "multi-class",
    "scope_40_208_8ssp": "multi-class",
    "stability": "regression",
    "solubility": "binary",
    "gb1": "regression",
}

TASK_METRICS = {
    "regression": REG_METRIC,
    "binary": CLASS_METRIC,
    "multi-label": CLASS_METRIC,
    "multi-class": CLASS_METRIC,
}

METRIC_TITLES = {
    "r2": r"$R^2\ (↑)$",
    "pearson": "Pearson's r (↑)",
    "spearman": "Spearman's ρ (↑)",
    "rmse": "RMSE (↓)",
    "mcc": "MCC (↑)",
    "ids": "2NN ID",
    "noverlap": "Neighborhood Overlap",
    "var@10": "Variance @ 10",
}

PLM_MODELS = {
    "esm_t6": "facebook/esm2_t6_8M_UR50D",
    "esm_t12": "facebook/esm2_t12_35M_UR50D",
    "esm_t30": "facebook/esm2_t30_150M_UR50D",
    "esm_t33": "facebook/esm2_t33_650M_UR50D",
    "esm_t36": "facebook/esm2_t36_3B_UR50D",
    "ankh_base": "ElnaggarLab/ankh-base",
    "ankh_large": "ElnaggarLab/ankh-large",
    "prott5": "Rostlab/prot_t5_xl_uniref50",
    "prostt5": "Rostlab/ProstT5",
    "progen2_small": "hugohrban/progen2-small",
    "progen2_medium": "hugohrban/progen2-medium",
    "progen2_large": "hugohrban/progen2-large",
}

LAST_PROTEIN = {
    "stability": "P68976",
    "fluorescence": "P54024",
    "fluorescence_classification": "P54024",
    "deeploc2": "P28302",
    "deeploc2_bin": "P28302",
    "meltome_atlas": "P69276",
    "scope_40_208": "P15176",
    "solubility": "P71094",
}