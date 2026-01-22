BASE="/scratch/chair_kalinina/s8rojoer/SMTB"
NCORES=8

echo "Go Home"
cd $HOME/SMTB2025_solution

# pip install --extra-index-url=https://pypi.nvidia.com --upgrade --force-reinstall cuml-cu12
# pip install --upgrade --force-reinstall torch numpy"<=2.2" pandas scikit-learn matplotlib transformers datasets tqdm

echo "Start AA CuML Predictions"
python -c "import cuml; print('CuML installed successfully.')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Train on ESMC model embeddings
python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esmc_300m/binding/ --n-classes 2 --max-layer 30
# python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esmc_300m/scope_40_208/ --n-classes 3 --max-layer 30
# python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esmc_300m/scope_40_208/ --n-classes 8 --max-layer 30

# Train on Pro(s)tT5 model embeddings
for model in prostt5 prott5; do
    python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/$model/binding/ --n-classes 2 --max-layer 24
    # python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/$model/scope_40_208/ --n-classes 3 --max-layer 24
    # python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/$model/scope_40_208/ --n-classes 8 --max-layer 24
done

# Train on one-hot encodings
python -m src.downstream.aa_cuml --data-base $ROOT/datasets/binding.csv --embed-path $ROOT/aa_embeddings/ohe/binding/ --n-classes 2 --max-layer 0
# python -m src.downstream.aa_cuml --data-base $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/ohe/scope_40_208/ --n-classes 3 --max-layer 0
# python -m src.downstream.aa_cuml --data-base $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/ohe/scope_40_208/ --n-classes 8 --max-layer 0
