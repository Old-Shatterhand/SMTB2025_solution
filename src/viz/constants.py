from pathlib import Path
from typing import Literal

SPLIT = "valid"
SPLIT_ID = {"train": 0, "valid": 1, "test": 2}[SPLIT]

MODELS = ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh_base", "ankh_large", "prostt5", "prott5", "progen2_small", "progen2_medium", "progen2_large", "ohe"]
DATASETS = ["fluorescence", "fluorescence_classification", "stability", "deeploc2", "deeploc2_bin", "meltome_atlas"]  # , "esol"
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
    "rita_small": 12,
    "rita_medium": 24,
    "rita_large": 24,
    "rita_xlarge": 24,
    "protgpt2": 36,
    "ohe": 0,
}

MODEL_COLORS = {
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
    "progen2_small": "yellow",
    "progen2_medium": "yellow",
    "progen2_large": "yellow",
    "protgpt2": "purple",
    "ohe": "gray",
}

MODEL_MARKERS = {
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

DS_NAME_MAP = {
    "fluorescence": "Fluorescence",
    "fluorescence_classification": "Fluorescence (Classification)",
    "stability": "Stability",
    "deeploc2": "DeepLoc2 (10-class)",
    "deeploc2_bin": "DeepLoc2 (Binary)",
    "meltome_atlas": "Meltome Atlas",
}

REG_METRIC = "rmse"
CLASS_METRIC = "mcc"
ROOT = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

def kill_axis(ax):
    ax.set_axis_off()


DATASET2TASK: dict[str, Literal["regression", "binary", "multi-label", "multi-class"]] = {
    "fluorescence": "regression",
    "fluorescence_classification": "binary",
    "stability": "regression",
    "deeploc2": "multi-label",
    "deeploc2_bin": "binary",
    "meltome_atlas": "multi-label",
    "binding": "binary",
    "scope_40_208": "multi-class",
    "stability": "regression",
}