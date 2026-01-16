BASE="/scratch/chair_kalinina/s8rojoer/SMTB"
NCORES=6

echo "Go Home"
cd $HOME/SMTB2025_solution

conda run -n plm --no-capture-output pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.12.*" "dask-cudf-cu12==25.12.*" "cuml-cu12==25.12.*" "cugraph-cu12==25.12.*" "nx-cugraph-cu12==25.12.*" "cuxfilter-cu12==25.12.*" "cucim-cu12==25.12.*" "pylibraft-cu12==25.12.*" "raft-dask-cu12==25.12.*" "cuvs-cu12==25.12.*" "nx-cugraph-cu12==25.12.*"

{
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esmc_600m/binding/ --n-classes 2 --max-layer 36"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esmc_600m/scope_40_208/ --n-classes 3 --max-layer 36"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esmc_600m/scope_40_208/ --n-classes 8 --max-layer 36"

    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/ankh-base/binding/ --n-classes 2 --max-layer 48"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/ankh-base/scope_40_208/ --n-classes 3 --max-layer 48"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/ankh-base/scope_40_208/ --n-classes 8 --max-layer 48"
} | xargs -P $NCORES -I {} bash -c "{}"
