#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com --upgrade --force-reinstall cuml-cu12
# pip install --upgrade --force-reinstall torch numpy"<=2.2" pandas scikit-learn matplotlib transformers datasets tqdm esm sentencepiece

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

for plm in "ankh-base 48" "esm_t6 6" "esm_t12 12" "esm_t30 30" "esm_t33 33" "esmc_300m 30" "prostt5 24" "prott5 24"; do
    set -- $plm
    python -m src.downstream.analyze --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/$1/binding --max-layer $2 --n-classes 2 --task binary --calcs id --force
done

python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/prott5/scope_40_208 --max-layer 24 --n-classes 3 --task multi-class --calcs id --force
# python -m src.downstream.analyze --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esm_t6/binding --max-layer 6 --n-classes 2 --task binary --calcs id --force
