from pathlib import Path

SPLIT = "valid"
SPLIT_ID = {"train": 0, "valid": 1, "test": 2}[SPLIT]

MODELS = ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh-base", "ankh-large", "prostt5", "prott5", "ohe"]
DATASETS = ["fluorescence", "stability", "deeploc2", "deeploc2_bin"]  # , "esol"]
LAYERS = {
    "esm_t6": 6,
    "esm_t12": 12,
    "esm_t30": 30,
    "esm_t33": 33,
    "esm_t36": 36,
    "esmc_300m": 30,
    "esmc_600m": 36,
    "ankh-base": 48,
    "ankh-large": 48,
    "prostt5": 24,
    "prott5": 24,
    "ohe": 0,
}
MODEL_COLORS = {
    "esm_t6": "lightsteelblue",
    "esm_t12": "cornflowerblue",
    "esm_t30": "royalblue",
    "esm_t33": "mediumblue",
    "esm_t36": "darkblue",
    "esmc_300m": "lightblue",
    "esmc_600m": "blue",
    "ankh-base": "lime",
    "ankh-large": "darkgreen",
    "prostt5": "orangered",
    "prott5": "darkorange",
    "ohe": "gray",
}


DS_NAME_MAP = {
    "fluorescence": "Fluorescence",
    "stability": "Stability",
    "deeploc2": "DeepLoc2 (10-class)",
    "deeploc2_bin": "DeepLoc2 (Binary)",
}

REG_METRIC = "rmse"
CLASS_METRIC = "mcc"
ROOT = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

def kill_axis(ax):
    ax.set_axis_off()
