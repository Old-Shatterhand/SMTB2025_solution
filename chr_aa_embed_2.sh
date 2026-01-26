#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# rm -rf .local/lib/python3.11/ # only if absolutely necessary
# pip install --extra-index-url=https://pypi.nvidia.com --force-reinstall cuml-cu12
# pip install --force-reinstall torch"<2.10" numpy"<=2.2" pandas"<2.5" scikit-learn matplotlib transformers datasets tqdm esm sentencepiece

# Embed sequences (not to be paralellized to not exceed GPU RAM)
for model in esm_t36 ankh-large; do
    python -m src.plm --model-name $model --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --aa-level
    python -m src.plm --model-name $model --data-path $BASE/datasets/scope_40_208.csv --output-path $BASE/aa_embeddings/$model/scope_40_208 --aa-level
done

# for model in esmc_300m esmc_600m prott5; do
#     rm -rf $BASE/aa_embeddings/$model/scope_40_208
#     python -m src.plm --model-name $model --data-path $BASE/datasets/scope_40_208.csv --output-path $BASE/aa_embeddings/$model/scope_40_208 --aa-level
# done
