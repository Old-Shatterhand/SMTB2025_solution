from multiprocessing.pool import ThreadPool
import os
from pathlib import Path

from src.viz.constants import LAYERS, MODELS

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def scp(origin: str, target: str) -> str:
    return f"scp uds:/scratch/chair_kalinina/s8rojoer/SMTB/{origin} /scratch/SCRATCH_SAS/roman/SMTB/{target}"


def fetch(cmd):
    err = os.system(cmd)
    return err


cmds = []
for model in {"progen2_small", "progen2_medium", "progen2_large", "protgpt2"}:
    for layer in range(LAYERS[model] + 1):
        for dataset in {"deeploc2", "deeploc2_bin", "fluorescence", "fluorescence_classification", "meltome_atlas", "stability"}:
            stumb = f"aa_embeddings/{model}/binding/layer_{layer}"
            (BASE / stumb).mkdir(parents=True, exist_ok=True)
            cmds.append(scp(stumb + f"/ids_.csv", stumb))
            cmds.append(scp(stumb + f"/noverlap_.csv", stumb))
            cmds.append(scp(stumb + f"/predictions_lr_.pkl", stumb))
            cmds.append(scp(stumb + f"/predictions_knn_.pkl", stumb))
            cmds.append(scp(stumb + f"/pca_.pkl", stumb))

        print(f"Fetching {model} layer {layer} results for SCOPe 40 2.08...")
        stumb_scope = f"embeddings/{model}/scope_40_208/layer_{layer}"
        (BASE / stumb_scope).mkdir(parents=True, exist_ok=True)
        for level in ["fold", "superfamily"]:
            # cmds.append(scp(stumb_scope + f"/ids_10_{class_}.csv", stumb_scope))
            cmds.append(scp(stumb_scope + f"/noverlap_{level}_min10.csv", stumb_scope))
            cmds.append(scp(stumb_scope + f"/ids_{level}_min10.csv", stumb_scope))
            cmds.append(scp(stumb_scope + f"/predictions_lr_{level}_min10.pkl", stumb_scope))
            cmds.append(scp(stumb_scope + f"/predictions_knn_{level}_min10.pkl", stumb_scope))
            cmds.append(scp(stumb_scope + f"/pca_{level}_min10.pkl", stumb_scope))

pool = ThreadPool(processes=32)
tasks = [None for _ in range(len(cmds))]
for i, cmd in enumerate(cmds):
    tasks[i] = pool.apply_async(fetch, (cmd,))
code = 0
for i, task in enumerate(tasks):
    print(f"\rFetching task {i+1} of {len(tasks)}...", end='')
    code += task.get()
print(f"All done with exit code-sum {code}.")
