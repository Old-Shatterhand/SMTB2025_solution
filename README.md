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
python -m src.plm --model-name esm_t6 --data-path $ROOT/datasets/fluorescence.csv --output-path $ROOT/embeddings/esm_t6/fluorescence/ --num-layers 6
```

For amino acid level embeddings:

```shell
python -m src.plm --model-name esm_t6 --data-path $ROOT/datasets/binding.csv --output-path $ROOT/aa_embeddings/esm_t6/binding/ --num-layers 6 --aa-level 
```

Avaliable models are: `esm_t6` (6), `esm_t12` (12), `esm_t30` (30), `esm_t33` (33), `esm_t36` (36), `esmc_300m` (30), `esmc_600m` (36), `ankh-base` (48), `ankh-large` (48), `prott5` (24), and `prostt5` (24). In brackets their number of layers (for the `--num-layers` argument).

### Computing latent space metrics and training models

```shell
python -m src.analyze --model-name esm_t6 --data-path $ROOT/datasets/fluorescence.csv --embed-base $ROOT/embeddings/esm_t6/fluorescence/ --max-layer 6 --task regression
```

### Training SemiFrozenESM

```shell
python -m src.downstream.semi_frozen_esm --data-path $ROOT/datasets/fluorescence.csv --out-folder path/to/folder --model-name esm_t6 --unfreeze 3 6 --task regression --lr 1e-4
```
