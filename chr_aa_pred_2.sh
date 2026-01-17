BASE="/scratch/chair_kalinina/s8rojoer/SMTB"
NCORES=6

echo "Go Home"
cd $HOME/SMTB2025_solution

{
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esm_t36num/binding/ --n-classes 2 --max-layer $num"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esm_t36num/scope_40_208/ --n-classes 3 --max-layer $num"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esm_t36/scope_40_208/ --n-classes 8 --max-layer $num"

    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/ankh-large/binding/ --n-classes 2 --max-layer 48"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/ankh-large/scope_40_208/ --n-classes 3 --max-layer 48"
    echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/ankh-large/scope_40_208/ --n-classes 8 --max-layer 48"
} | xargs -P $NCORES -I {} bash -c "{}"
