#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com --upgrade --force-reinstall cuml-cu12
# pip install --upgrade --force-reinstall torch numpy"<=2.2" pandas scikit-learn matplotlib transformers datasets tqdm

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

for num in 6 12 30 33; do
    python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esm_t$num/binding/ --max-layer $num --task binary --n-classes 2 --force
    python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esm_t$num/scope_40_208/ --max-layer $num --task class --n-classes 3 --force
    python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esm_t$num/scope_40_208/ --max-layer $num --task class --n-classes 8 --force
done
