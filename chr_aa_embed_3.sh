#!/usr/bin/env bash

BASE="/scratch/SCRATCH_SAS/roman/SMTB"

echo "Go Home"
cd $BASE/SMTB2025_solution

# ROOT="/scratch/SCRATCH_SAS/roman/SMTB"
# ROOT="/scratch/chair_kalinina/s8rojoer/SMTB"

# Embed sequences (not to be paralellized to not exceed GPU RAM)
# for model in esm_t6 esm_t12 esm_t30 esm_t33 esm_t36 esmc-300m esmc-600m ankh-base ankh-large prostt5 prostt5 ohe; do
for model in esmc-600m ankh-base; do
    python -m src.plm --model-name $model --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --aa-level
    python -m src.plm --model-name $model --data-path $BASE/datasets/scope_40_208.csv --output-path $BASE/aa_embeddings/$model/scope_40_208 --aa-level
done
