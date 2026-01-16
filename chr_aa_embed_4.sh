#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# Embed sequences (not to be paralellized to not exceed GPU RAM)
for model in esmc-300m prostt5 prostt5 ohe; do
    python -m src.plm --model-name $model --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --aa-level
    python -m src.plm --model-name $model --data-path $BASE/datasets/scope_40_208.csv --output-path $BASE/aa_embeddings/$model/scope_40_208 --aa-level
done
