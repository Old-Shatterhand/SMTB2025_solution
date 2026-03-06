#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com --upgrade --force-reinstall cuml-cu12
# pip install --upgrade --force-reinstall torch numpy"<=2.2" pandas scikit-learn matplotlib transformers datasets tqdm esm sentencepiece

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

for dataset in deeploc2 meltome_atlas fluorescence scope_40_208 stability; do
    python -m src.plm --model-name protgpt2 --data-path $BASE/datasets/$dataset.csv --output-path $BASE/embeddings/protgpt2/$dataset
done

{
    python -m src.downstream.analyze --data-path $BASE/datasets/deeploc2.csv --embed-base $BASE/embeddings/protgpt2/deeploc2 --max-layer 36 --task multi-label
    python -m src.downstream.analyze --data-path $BASE/datasets/deeploc2_bin.csv --embed-base $BASE/embeddings/protgpt2/deeploc2 --max-layer 36 --task binary
    python -m src.downstream.analyze --data-path $BASE/datasets/fluorescence.csv --embed-base $BASE/embeddings/protgpt2/fluorescence --max-layer 36 --task regression
    python -m src.downstream.analyze --data-path $BASE/datasets/fluorescence_classification.csv --embed-base $BASE/embeddings/protgpt2/fluorescence --max-layer 36 --task binary
    python -m src.downstream.analyze --data-path $BASE/datasets/meltome_atlas.csv --embed-base $BASE/embeddings/protgpt2/meltome_atlas --max-layer 36 --task regression
    python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/embeddings/protgpt2/scope_40_208 --max-layer 36 --task multi-class --level superfamily --min 10
    python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/embeddings/protgpt2/scope_40_208 --max-layer 36 --task multi-class --level fold --min 10
    python -m src.downstream.analyze --data-path $BASE/datasets/stability.csv --embed-base $BASE/embeddings/protgpt2/stability --max-layer 36 --task regression
} | xargs -P 8 -I {} bash -c "{}"
