#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com --upgrade --force-reinstall cuml-cu12
# pip install --upgrade --force-reinstall torch numpy"<=2.2" pandas scikit-learn matplotlib transformers datasets tqdm

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Train on ESMC model embeddings
# python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esmc_300m/binding/ --n-classes 2 --max-layer 30
python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esmc_300m/scope_40_208/ --max-layer 30 --task class --n-classes 3 --force
python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esmc_300m/scope_40_208/ --max-layer 30 --task class --n-classes 8 --force

# Train on Pro(s)tT5 model embeddings
# for model in prott5; do  # prostt5 
#     python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/$model/binding/ --n-classes 2 --max-layer 24
python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/prott5/scope_40_208/ --max-layer 24 --task class --n-classes 3 --force
python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/prott5/scope_40_208/ --max-layer 24 --task class --n-classes 8 --force
# done

# Train on one-hot encodings
# python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/ohe/binding/ --n-classes 2 --max-layer 0
python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/ohe/scope_40_208/ --max-layer 0 --task class --n-classes 3 --force
python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/ohe/scope_40_208/ --max-layer 0 --task class --n-classes 8 --force
