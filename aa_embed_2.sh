#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com --upgrade --force-reinstall cuml-cu12
# pip install --upgrade --force-reinstall torch numpy"<=2.2" pandas scikit-learn matplotlib transformers datasets tqdm esm sentencepiece

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Embed sequences (not to be paralellized to not exceed GPU RAM)
# for model in rita_small rita_medium rita_large rita_xlarge; do
#     for dataset in deeploc2 meltome_atlas fluorescence scope_40_208 stability; do
#         python -m src.plm --model-name $model --data-path $BASE/datasets/$dataset.csv --output-path $BASE/aa_embeddings/$model/$dataset
#     done
#     python -m src.plm --model-name $model --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --aa-level
#     python -m src.plm --model-name $model --data-path $BASE/datasets/scope_40_208.csv --output-path $BASE/aa_embeddings/$model/scope_40_208 --aa-level
# done

for plm in "progen2_small 12" "progen2_medium 27" "progen2_large 32" "rita_small 12" "rita_medium 24" "rita_large 24" "rita_xlarge 24" "protgpt2 36"; do
    set -- $plm
    python -m src.plm --model-name $1 --data-path $BASE/datasets/deeploc2.csv --output-path $BASE/embeddings/$1/deeploc2 --max-layer $2 --task mulit-label
    python -m src.plm --model-name $1 --data-path $BASE/datasets/deeploc2_bin.csv --output-path $BASE/embeddings/$1/deeploc2 --max-layer $2 --task binary
    python -m src.plm --model-name $1 --data-path $BASE/datasets/fluorescence.csv --output-path $BASE/embeddings/$1/fluorescence --max-layer $2 --task regression
    python -m src.plm --model-name $1 --data-path $BASE/datasets/fluorescence_classification.csv --output-path $BASE/embeddings/$1/fluorescence --max-layer $2 --task binary
    # python -m src.plm --model-name $1 --data-path $BASE/datasets/meltome_atlas.csv --output-path $BASE/embeddings/$1/meltome_atlas --max-layer $2 --task regression
    # python -m src.plm --model-name $1 --data-path $BASE/datasets/scope_40_208.csv --output-path $BASE/embeddings/$1/scope_40_208 --max-layer $2 --task multi-class --level superfamily --min 10
    # python -m src.plm --model-name $1 --data-path $BASE/datasets/scope_40_208.csv --output-path $BASE/embeddings/$1/scope_40_208 --max-layer $2 --task multi-class --level fold --min 10
    # python -m src.plm --model-name $1 --data-path $BASE/datasets/stability.csv --output-path $BASE/embeddings/$1/stability --max-layer $2 --task regression
    python -m src.plm --model-name $1 --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --max-layer $2 --task binary --num-classes 2
    python -m src.plm --model-name $1 --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --max-layer $2 --task multi-class --num-classes 3
    # python -m src.plm --model-name $1 --data-path $BASE/datasets/binding.csv --output-path $BASE/aa_embeddings/$model/binding --max-layer $2 --task multi-class --num-classes 8
done | xargs -P 16 -I {} bash -c "{}"
