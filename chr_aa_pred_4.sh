BASE="/scratch/chair_kalinina/s8rojoer/SMTB"
NCORES=12

echo "Go Home"
cd $HOME/SMTB2025_solution

conda run -n plm --no-capture-output pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.12.*" "dask-cudf-cu12==25.12.*" "cuml-cu12==25.12.*" "cugraph-cu12==25.12.*" "nx-cugraph-cu12==25.12.*" "cuxfilter-cu12==25.12.*" "cucim-cu12==25.12.*" "pylibraft-cu12==25.12.*" "raft-dask-cu12==25.12.*" "cuvs-cu12==25.12.*" "nx-cugraph-cu12==25.12.*"

{
    # Train on ESMC model embeddings
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esmc_300m/binding/ --n-classes 2 --max-layer 30"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esmc_300m/scope_40_208/ --n-classes 3 --max-layer 30"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esmc_300m/scope_40_208/ --n-classes 8 --max-layer 30"

    # Train on Pro(s)tT5 model embeddings
    for model in prostt5 prott5; do
        echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/$model/binding/ --n-classes 2 --max-layer 24"
        echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/$model/scope_40_208/ --n-classes 3 --max-layer 24"
        echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/$model/scope_40_208/ --n-classes 8 --max-layer 24"
    done

    # Train on one-hot encodings
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-base $ROOT/datasets/binding.csv --embed-path $ROOT/aa_embeddings/ohe/binding/ --n-classes 2 --max-layer 0"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-base $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/ohe/scope_40_208/ --n-classes 3 --max-layer 0"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-base $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/ohe/scope_40_208/ --n-classes 8 --max-layer 0"
} | xargs -P $NCORES -I {} bash -c "{}"
