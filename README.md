# Repository with Protein Language Model Layer Inspection

## Folder structure

The idea is to have two folders, one holding the data and embeddings (a data heavy one) and one holding the code and results (more lightweight). How the code is structured is not too important, but the data should be structured as follows, so that the sampling script still works.

### Data-directory structure

```shell
$ROOT
   ├── datasets/
   │   ├── fluorescence.csv
   │   └── ... (other datasets)
   ├── embeddings/
   │   ├── esm_t6/
   │   │   ├── fluorescence/
   │   │   │   ├── layer_0/
   │   |   │   │   ├── P00000.pkl
   │   |   │   │   └── ... (more embeddings)
   │   │   │   └── ... (more layers)
   │   │   └── ... (other datasets)
   │   └── ... (other models)
   └── aa_embeddings/
       ├── esm_t6/
       │   ├── binding/
       │   │   ├── layer_0/
       │   │   │   ├── P00000.pkl
       │   │   │   └── ... (more embeddings)
       │   │   └── ... (more layers)
       │   └── scope_40_208/
       │       ├── layer_0/ 
       │       │   ├── P00000.pkl
       │       │   └── ... (more embeddings)
       │       └── ... (more layers)
       └── ... (other models)
```

## Requirements

All dependencies are listed in `requirements.txt`. To install them, run:

```shell
pip install -r requirements.txt
```

## How to run the code

Assuming the there is a root directory `$ROOT` where the data and embeddings are stored, the following commands can be used to run the code for the fluorescence dataset. The scripts can be adjusted to run other models or datasets.

```shell
$ROOT='/scratch/SCRATCH_SAS/roman/SMTB'
```

### Fetching a dataset

```shell
python -m src.datasets.fluorescence --save-path $ROOT/datasets
```

There are more options in the src/datasets folder for fetching other datasets.

### Computing ESM-t6 embeddings

For a "normal" dataset:

```shell
python -m src.plm --model-name esm_t6 --data-path $ROOT/datasets/fluorescence.csv --output-path $ROOT/embeddings/esm_t6/fluorescence/
```

For amino acid level embeddings:

```shell
python -m src.plm --model-name esm_t6 --data-path $ROOT/datasets/binding.csv --output-path $ROOT/aa_embeddings/esm_t6/binding/ --aa-level
```

Available models are: `esm_t6` (6), `esm_t12` (12), `esm_t30` (30), `esm_t33` (33), `esm_t36` (36), `esmc_300m` (30), `esmc_600m` (36), `ankh_base` (48), `ankh_large` (48), `prott5` (24), `prostt5` (24), `progen2_small` (12), `progen2_medium` (27), `progen2_large` (32), and `protgpt2` (36). In brackets their number of layers, which `src.plm` infers per family from the checkpoint.

### Computing latent space metrics and training models

```shell
python -m src.downstream.probe_layer --data-path $ROOT/datasets/fluorescence.csv --embed-base $ROOT/embeddings/esm_t6/fluorescence/ --max-layer 6 --task regression
```

### Training SemiFrozenESM

```shell
python -m src.downstream.semifrozen_esm --data-path $ROOT/datasets/fluorescence.csv --out-folder path/to/folder --model-name esm_t6 --unfreeze 3 6 --task regression --lr 1e-4
```

## Picking the best layer for your own dataset

The analysis above reproduces the paper. To *apply* its finding to a new dataset,
use the `plmlayer` package in this repository, which probes a subsample of your
data and returns a truncated model. See [README_plmlayer.md](README_plmlayer.md).

```shell
pip install -e .
plmlayer suggest --model facebook/esm2_t33_650M_UR50D --data my.csv --out ./my-esm
```
