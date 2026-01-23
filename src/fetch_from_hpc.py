import os
from pathlib import Path

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

MODELS = ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh-base", "ankh-large", "prostt5", "prott5", "ohe"]
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

def scp(origin: str, target: str) -> str:
    return f"scp uds:/scratch/chair_kalinina/s8rojoer/SMTB/{origin} /scratch/SCRATCH_SAS/roman/SMTB/{target}"

for model in MODELS:
    if model in {"esm_t36", "esmc_600m", "ankh-large"}:
        continue  # too large
    for dataset in ["scope_40_208"]:  # "binding", 
        for layer in range(LAYERS[model] + 1):
            print(f"Fetching {model} layer {layer} results for {dataset}...")
            stumb = f"aa_embeddings/{model}/{dataset}/layer_{layer}"
            (BASE / stumb).mkdir(parents=True, exist_ok=True)
            for class_ in [3, 8]:
                os.system(scp(stumb + f"/ids_{class_}.csv", stumb))
                os.system(scp(stumb + f"/noverlap_10_{class_}.csv", stumb))
                os.system(scp(stumb + f"/predictions_lr_{class_}.pkl", stumb))
                os.system(scp(stumb + f"/predictions_knn_{class_}.pkl", stumb))
