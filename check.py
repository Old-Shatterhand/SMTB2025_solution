from pathlib import Path

from src.viz.constants import MODELS, WP_DATASETS, AA_DATASETS, LAYERS, LAST_PROTEIN


BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"

NUM_OUTPUTS = {
    "stability": 1,
    "fluorescence": 1,
    "fluorescence_classification": 1,
    "deeploc2": 10,
    "deeploc2_bin": 1,
    "meltome_atlas": 1,
    "scope_40_208": -1,
}


prints = []
found, total = 0, 0


def foo(fpath, prefix):
    global prints, found, total
    total += 1
    if not fpath.exists():
        prints.append(f"[{prefix}] {fpath.parent.parent.parent.name} {fpath.parent.parent.name} {fpath.parent.name.split('_')[-1]} does not have {fpath.name}")
        return
    found += 1


for model in MODELS:
    for dataset in WP_DATASETS:
        # embedding dataset name
        emb_dataset = dataset.split("_")[0] if dataset not in {"meltome_atlas", "scope_40_208"} else dataset

        ds_root = BASE / "embeddings" / model / emb_dataset
        if not ds_root.exists():
            prints.append(f"[WP] {model} {dataset} does not exist")
            continue
        
        # check for global probes
        if not (ds_root / f"probe_weights_{dataset}_{10 if dataset == 'deeploc2' else 1}.pkl").exists():
            prints.append(f"[WP] {model} {dataset} does not have global probes")
            continue

        for layer in range(LAYERS[model]):
            print(f"\r{model}-{layer}-{dataset}", end=" " * 20)
            if not (ds_root / f"layer_{layer}" / f"{LAST_PROTEIN[dataset]}.pkl").exists() and emb_dataset == dataset:
                prints.append(f"[WP] {model} {dataset} layer {layer} does not have all embeddings")
                continue
            
            # check for embedding space statistics
            for filename, ext in [("ids", "csv"), ("pca", "pkl"), ("noverlap", "csv"), ("predictions_knn", "pkl"), ("predictions_lr", "pkl")]:
                # no noverlap for last layer as it's an in-between-layer metric
                if filename == "noverlap" and layer == LAYERS[model] - 1:
                    continue

                if dataset == "scope_40_208":
                    # Check for both superfamily and fold probes for scope_40_208
                    for level in ["superfamily", "fold"]:
                        foo(ds_root / f"layer_{layer}" / f"{filename}_{level}_min10.{ext}", "WP")
                else:
                    # For other datasets, check for the standard probe
                    foo(ds_root / f"layer_{layer}" / f"{filename}.{ext}", "WP")
    
    for dataset in AA_DATASETS:
        if model == "protgpt2":  # no AA embeddings for protgpt2
            continue
        if model in {"progen2_large", "ankh_large", "esmc_600m", "esm_t36"}:
            continue  # these models don't have AA embeddings yet
        
        ds_root = BASE / "aa_embeddings" / model / dataset
        if not ds_root.exists():
            prints.append(f"[AA] {model} {dataset} does not exist")
            continue
        for layer in range(LAYERS[model]):
            print(f"\r{model}-{layer}-{dataset}", end=" " * 20)

            for filename, ext in [("ids", "csv"), ("pca", "pkl"), ("noverlap", "csv"), ("predictions_knn", "pkl"), ("predictions_lr", "pkl")]:
                if filename == "noverlap" and layer == LAYERS[model] - 1:  # no noverlap for last layer as it's an in-between-layer metric
                    continue
                
                if filename.startswith("predictions") and dataset == "scope_40_208":
                    for classes in [3, 8]:
                        foo(ds_root / f"layer_{layer}" / f"{filename}_{classes}.pkl", "AA")
                else:
                    foo(ds_root / f"layer_{layer}" / f"{filename}.{ext}", "AA")

print(f"\rFound {found} out of {total} files.")
print("\n".join(sorted(prints)))
