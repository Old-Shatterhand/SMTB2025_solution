#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com --upgrade --force-reinstall cuml-cu12
# pip install --upgrade --force-reinstall torch numpy"<=2.2" pandas scikit-learn matplotlib transformers datasets tqdm

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Embed sequences (not to be paralellized to not exceed GPU RAM)
for model in esm_t6 esm_t12 esm_t30 esm_t33 esmc_600m ankh-base esmc_300m prott5 prostt5 ohe; do
    rm -rf $BASE/aa_embeddings/$model/binding
    python -m src.plm --model-name $model --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --aa-level
done
