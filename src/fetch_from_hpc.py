from multiprocessing.pool import ThreadPool
import os
from pathlib import Path

from tqdm import tqdm

from src.viz.constants import LAYERS, MODELS

BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def scp(origin: str, target: str) -> str:
    # if target == "embeddings/progen2_medium/fluorescence/layer_0":
    #     print("/scratch/SCRATCH_SAS/roman/SMTB" / Path(target) / origin.split("/")[-1])
    #     print(("/scratch/SCRATCH_SAS/roman/SMTB" / Path(target) / origin.split("/")[-1]).exists())
    if ("/scratch/SCRATCH_SAS/roman/SMTB" / Path(target) / origin.split("/")[-1]).exists():
        return ""
    return f"scp uds:/scratch/chair_kalinina/s8rojoer/SMTB/{origin} /scratch/SCRATCH_SAS/roman/SMTB/{target}"


def fetch(cmd):
    err = os.system(cmd)
    return err


cmds = []
for model in ["progen2_small", "progen2_medium"]:# ["prostt5", "prott5"]:  # {"progen2_small", "progen2_medium", "progen2_large", "protgpt2"}:
    for layer in range(LAYERS[model] + 1):
        # for dataset in {"deeploc2", "deeploc2_bin", "fluorescence", "fluorescence_classification", "meltome_atlas", "stability"}:
        #     stumb = f"embeddings/{model}/{dataset}/layer_{layer}"
        #     (BASE / stumb).mkdir(parents=True, exist_ok=True)
        #     cmds.append(scp(stumb + f"/ids.csv", stumb))
        #     cmds.append(scp(stumb + f"/noverlap.csv", stumb))
        #     if dataset in {"fluorscence", "meltome_atlas", "stability"}:
        #         cmds.append(scp(stumb + f"/predictions_lr.pkl", stumb))
        #     cmds.append(scp(stumb + f"/predictions_knn.pkl", stumb))
        #     cmds.append(scp(stumb + f"/pca.pkl", stumb))

        # stumb_scope = f"embeddings/{model}/scope_40_208/layer_{layer}"
        # (BASE / stumb_scope).mkdir(parents=True, exist_ok=True)
        # for level in ["fold", "superfamily"]:
        #     cmds.append(scp(stumb_scope + f"/noverlap_{level}_min10.csv", stumb_scope))
        #     cmds.append(scp(stumb_scope + f"/ids_{level}_min10.csv", stumb_scope))
        #     if dataset in {"fluorscence", "meltome_atlas", "stability"}:
        #         cmds.append(scp(stumb_scope + f"/predictions_lr_{level}_min10.pkl", stumb_scope))
        #     cmds.append(scp(stumb_scope + f"/predictions_knn_{level}_min10.pkl", stumb_scope))
        #     cmds.append(scp(stumb_scope + f"/pca_{level}_min10.pkl", stumb_scope))
        
        if model != "protgpt2":
            stumb = f"aa_embeddings/{model}/binding/layer_{layer}"
            (BASE / stumb).mkdir(parents=True, exist_ok=True)
            cmds.append(scp(stumb + f"/ids.csv", stumb))
            cmds.append(scp(stumb + f"/noverlap.csv", stumb))
            cmds.append(scp(stumb + f"/predictions_lr.pkl", stumb))
            cmds.append(scp(stumb + f"/predictions_knn.pkl", stumb))
            cmds.append(scp(stumb + f"/pca.pkl", stumb))

            # stumb_scope = f"aa_embeddings/{model}/scope_40_208/layer_{layer}"
            # (BASE / stumb_scope).mkdir(parents=True, exist_ok=True)
            # cmds.append(scp(stumb_scope + f"/noverlap.csv", stumb_scope))
            # cmds.append(scp(stumb_scope + f"/ids.csv", stumb_scope))
            # cmds.append(scp(stumb_scope + f"/pca.pkl", stumb_scope))
            # for classes in [3, 8]:
            #     cmds.append(scp(stumb_scope + f"/predictions_lr_{classes}.pkl", stumb_scope))
            #     cmds.append(scp(stumb_scope + f"/predictions_knn_{classes}.pkl", stumb_scope))


cmds = list(sorted(filter(lambda c: c != "", cmds)))
print(len(cmds), "files to fetch.")
for cmd in tqdm(cmds):
    # print(cmd)
    fetch(cmd)

# print(len(cmds), "files to fetch.")
# pool = ThreadPool(processes=32)
# tasks = [None for _ in range(len(cmds))]
# for i, cmd in enumerate(cmds):
#     tasks[i] = pool.apply_async(fetch, (cmd,))
# code = 0
# for i, task in enumerate(tasks):
#     print(f"\rFetching task {i+1} of {len(tasks)}...", end='')
#     code += task.get()
# print(f"All done with exit code-sum {code}.")
