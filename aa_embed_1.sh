#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com --upgrade --force-reinstall cuml-cu12
# pip install --upgrade --force-reinstall torch numpy"<=2.2" pandas scikit-learn matplotlib transformers datasets tqdm esm sentencepiece

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

for plm in "progen2_small 12" "progen2_medium 27" "progen2_large 32"; do
    set -- $plm
    echo "python -m src.downstream.analyze --data-path $BASE/datasets/deeploc2.csv --embed-base $BASE/embeddings/$1/deeploc2 --max-layer $2 --task multi-label"
    # python -m src.downstream.analyze --data-path $BASE/datasets/deeploc2_bin.csv --embed-base $BASE/embeddings/$1/deeploc2 --max-layer $2 --task binary
    # python -m src.downstream.analyze --data-path $BASE/datasets/fluorescence.csv --embed-base $BASE/embeddings/$1/fluorescence --max-layer $2 --task regression
    # python -m src.downstream.analyze --data-path $BASE/datasets/fluorescence_classification.csv --embed-base $BASE/embeddings/$1/fluorescence --max-layer $2 --task binary
    # python -m src.downstream.analyze --data-path $BASE/datasets/meltome_atlas.csv --embed-base $BASE/embeddings/$1/meltome_atlas --max-layer $2 --task regression
    # python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/embeddings/$1/scope_40_208 --max-layer $2 --task multi-class --level superfamily --min 10
    # python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/embeddings/$1/scope_40_208 --max-layer $2 --task multi-class --level fold --min 10
    # python -m src.downstream.analyze --data-path $BASE/datasets/stability.csv --embed-base $BASE/embeddings/$1/stability --max-layer $2 --task regression
    # python -m src.downstream.analyze --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/$1/binding --max-layer $2 --n-classes 2 --task binary
    # python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/$1/scope_40_208 --max-layer $2 --n-classes 3 --task multi-class
    # python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/$1/scope_40_208 --max-layer $2 --n-classes 8 --task multi-class
done | xargs -P 3 -I {} bash -c "{}"
