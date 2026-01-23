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

## How to run the code

Assuming the there is a root directory `$ROOT` where the data and embeddings are stored, the following commands can be used to run the code for the fluorescence dataset. The scripts can be adjusted to run other models or datasets.

```shell
$ROOT =  /scratch/SCRATCH_SAS/roman/SMTB
```

### Fetching a dataset

```shell
python src/datasets/fluorescence.py --save-path $ROOT/datasets
```

### Computing ESM-t6 embeddings

For a "normal" dataset:

```shell
python src/plm.py --model-name esm_t6 --data-path $ROOT/datasets/fluorescence.csv --output-path $ROOT/embeddings/esm_t6/fluorescence/ --num-layers 6
```

For amino acid level embeddings:

```shell
python src/plm.py --model-name esm_t6 --data-path $ROOT/datasets/binding.csv --output-path $ROOT/aa_embeddings/esm_t6/binding/ --num-layers 6 --aa-level 
```

### Training the MLP model on the Fluorescence embeddings

```shell
python mlp.py --input-dim 320 --hidden-dim 64 --output-dim 1 --mode regression --data-path $ROOT/datasets/fluorescence.csv --embeds-path $ROOT/embeddings/esm_t6/fluorescence/layer_0/ --log-folder ../test_logs/
```

### Sampling the embeddings and a dataset for faster development and debugging
