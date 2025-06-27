# Repository with Solutions for  SMTB 2025

**_DISCLAIMER:_** This repository contains solution scripts and notes for the SMTB 2025 competition and is **NOT** to be shared with the students. This is also not **THE** one solution, I want to import on the students, but rather a lookup for use to see how to approach the task in case the students struggle with it to provide them with a hint to a potential solution.

## Folder structure

The idea is to have two folders, one holding the data and embeddings (a data heavy one) and one holding the code and results (more lightweight). How the code is structured is not too important, but the data should be structured as follows, so that the sampling script still works.

### Data-directory structure

```shell
$ROOT
   ├── datasets/
   │   ├── fluorescence.csv
   │   └── ... (other datasets)
   └── embeddings/
       ├── esm_t6/
       │   ├── fluorescence/
       │   │   ├── layer_0/
       |   │   │   ├── P00000.pkl
       |   │   │   └── ... (more embeddings)
       │   │   └── ... (more layers)
       │   └── ... (other datasets)
       └── ... (other models)
```

## How to run the code

Assuming the there is a root directory `$ROOT` where the data and embeddings are stored, the following commands can be used to run the code for the fluorescence dataset. The scripts can be adjusted to run other models or datasets.

```shell
$ROOT =  /scratch/SCRATCH_SAS/roman/SMTB
```

### Fetching the Fluorescence dataset

```shell
python fluorescence.py --save-path $ROOT/datasets
```

### Computing ESM-t6 embeddings for the Fluorescence dataset

```shell
python esm.py --data-path $ROOT/datasets/fluorescence.csv --output-path $ROOT/embeddings/esm_t6/fluorescence/ --num-layers 6
```

### Training the MLP model on the Fluorescence embeddings

```shell
python mlp.py --input-dim 320 --hidden-dim 64 --output-dim 1 --mode regression --data-path $ROOT/datasets/fluorescence.csv --embeds-path $ROOT/embeddings/esm_t6/fluorescence/layer_0/ --log-folder ../test_logs/
```

### Sampling the embeddings and a dataset for faster development and debugging

```shell
python sample_embeds.py --data-path $ROOT/datasets/fluorescence.csv --embed-path $ROOT/embeddings/esm_t6/fluorescence/ --save-path $ROOT/sampled/
```

## Handling the students - Timeline

Tasks to be done:

1. Fetch Fluorescence dataset and preprocess
2. Fetch ESM t6 model and make it embed protein sequence
3. Extend datasets to 2-3 of
   - DeepLoc 2.0 (from DTU) (resulting in two tasks)
   - DeepSol
   - (Thermo-)Stability
4. Train MLP model on the embeddings
5. Plot the results
6. Extend PLM collection to include 2 of
    - ProstT5
    - Ankh
    - proteinBERT

We'll take 4 students (A, B, C, D) and assign tasks as follows:

| ??.07. | Task                  | A | B | C | D |
|--------|-----------------------|---|---|---|---|
|  05.   | General Introduction  | X | X | X | X |
|  06.   | 1.: Fluorescence      | X | X |   |   |
|  06.   | 2.: ESM-t6            |   |   | X | X |
|  09.   | 5.1: +1 Dataset       | X |   | X |   |
|  09.   | 5.2: +1 Dataset       |   | X |   | X |
|--------|-----------------------|---|---|---|---|
|        | >>> BREAK DAY <<<     |   |   |   |   |
|        | Run bigger ESM models |   |   |   |   |
|--------|-----------------------|---|---|---|---|
|  07.   | Downstream implement. | ? | ? | ? | ? |
|  10.   | 6.1: +1 PLM           | X |   |   | X |
|  10.   | 6.2: +1 PLM           |   | X | X |   |
|  11.   |                       |   |   |   |   |
|  12.   |                       |   |   |   |   |
|  13.   | Poster Generation     | X | X | X | X |
|  14.   | >>> Conference <<<    | X | X | X | X |
