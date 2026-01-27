#!/usr/bin/env bash

BASE="/scratch/chair_kalinina/s8rojoer/SMTB"
NCORES=6

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.12.*" "dask-cudf-cu12==25.12.*" "cuml-cu12==25.12.*" "cugraph-cu12==25.12.*" "nx-cugraph-cu12==25.12.*" "cuxfilter-cu12==25.12.*" "cucim-cu12==25.12.*" "pylibraft-cu12==25.12.*" "raft-dask-cu12==25.12.*" "cuvs-cu12==25.12.*" "nx-cugraph-cu12==25.12.*"
# pip install torch numpy pandas dadapy scikit-learn matplotlib transformers datasets tqdm

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

{
    python -m src.downstream.analyze --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esm_t36/binding/ --max-layer 36 --task binary --n-classes 2 --force
    python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esm_t36/scope_40_208/ --max-layer 36 --task class --n-classes 3 --force
    python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esm_t36/scope_40_208/ --max-layer 36 --task class --n-classes 8 --force

    python -m src.downstream.analyze --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/ankh-large/binding/ --max-layer 48 --task binary --n-classes 2 --force
    python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/ankh-large/scope_40_208/ --max-layer 48 --task class --n-classes 3 --force
    python -m src.downstream.analyze --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/ankh-large/scope_40_208/ --max-layer 48 --task class --n-classes 8 --force
} | xargs -P $NCORES -I {} bash -c "{}"
