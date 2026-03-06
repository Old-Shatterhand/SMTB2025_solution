import os
import time
import shutil
from pathlib import Path

from src.viz.constants import LAYERS


BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"
start = time.time()

def foo(src_file, dst_file):
    if not src_file.exists():  # skip if source file doesn't exist
        return False
    if not dst_file.exists():  # copy if destination file doesn't exist
        shutil.copy(src_file, dst_file)
        return True
    if os.path.getmtime(src_file) > os.path.getmtime(dst_file) + 3600:  # if source file is newer than destination file by more than 1 hour, copy
        shutil.copy(src_file, dst_file)
        return True
    return False  # both files exists and destination file is newer than source file


def copy_newest(folder: Path, search_prefix: str, filename: str):
    try:
        candidates = list(folder.glob(search_prefix + "*"))
        if len(candidates) == 0:
            return
        winner = max(candidates, key=lambda f: f.stat().st_mtime)
        if winner.name != filename:
            shutil.copy(winner, folder / filename)
    except Exception as e:
        print(f"Error {e}\n\twhen copying F:", folder, "P:", search_prefix, "FN:", filename)


for model in ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "esm_t36", "esmc_300m", "esmc_600m", "ankh_base", "ankh_large", "prostt5", "prott5", "progen2_small", "progen2_medium", "progen2_large", "protgpt2"]:
    for fill_prefix in ["", "empty_"]:
        model_name = fill_prefix + model
        for layer in range(LAYERS[model] + 1):
            print(f"\rProcessing model: {model_name}, layer {layer}", end=" "*10)
            if (BASE / "embeddings" / model_name).exists():
                for dataset in ["deeploc2", "deeploc2_bin", "fluorescence", "fluorescence_classification", "meltome_atlas", "stability"]:
                    for prefix, filename in [("noverlap", "noverlap.csv"), ("ids", "ids.csv"), ("pca", "pca.pkl"), ("predictions_lr", "predictions_lr.pkl"), ("predictions_knn", "predictions_knn.pkl")]:
                        copy_newest(BASE / "embeddings" / model_name / dataset / f"layer_{layer}", prefix, filename)
                
                for level in ["fold", "superfamily"]:
                    for prefix, filename in [
                        (f"ids*{level}*min10", f"ids_{level}_min10.csv"), 
                        (f"noverlap*{level}*min10", f"noverlap_{level}_min10.csv"), 
                        (f"pca*{level}*min10", f"pca_{level}_min10.pkl"),
                        (f"predictions_lr*{level}*min10", f"predictions_lr_{level}_min10.pkl"),
                        (f"predictions_knn*{level}*min10", f"predictions_knn_{level}_min10.pkl"),
                    ]:
                        copy_newest(BASE / "embeddings" / model_name / "scope_40_208" / f"layer_{layer}", prefix, filename)
            
            if (BASE / "aa_embeddings" / model_name).exists():
                for dataset in ["binding", "scope_40_208"]:
                    for prefix, filename in [("noverlap", "noverlap.csv"), ("ids", "ids.csv"), ("pca", "pca.pkl")]:
                        copy_newest(BASE / "aa_embeddings" / model_name / dataset / f"layer_{layer}", prefix, filename)
                
                for algo in ["lr", "knn"]:
                    copy_newest(BASE / "aa_embeddings" / model_name / "binding" / f"layer_{layer}", f"predictions_{algo}", f"predictions_{algo}.pkl")
                    for classes in [3, 8]:
                        copy_newest(BASE / "aa_embeddings" / model_name / "scope_40_208" / f"layer_{layer}", f"predictions_{algo}*{classes}", f"predictions_{algo}_{classes}.pkl")
            
print(f"\nDone in {time.time() - start:.2f} seconds.")
