#!/usr/bin/env bash

BASE="/scratch/SCRATCH_SAS/roman/SMTB"

echo "Go Home"
cd $BASE/SMTB2025_solution

# Embed sequences (not to be paralellized to not exceed GPU RAM)
for model in esm_t6 esm_t12 esm_t30 esm_t33; do
    conda run -n plm --no-capture-output python -m src.plm --model-name $model --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --aa-level
    conda run -n plm --no-capture-output python -m src.plm --model-name $model --data-path $BASE/datasets/scope_40_208.csv --output-path $BASE/aa_embeddings/$model/scope_40_208 --aa-level
done
