BASE="/scratch/chair_kalinina/s8rojoer/SMTB"
NCORES=12

echo "Go Home"
cd $HOME/SMTB2025_solution

{
    for num in 6 12 30 33; do
        echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/binding.csv --embed-base $BASE/aa_embeddings/esm_t$num/binding/ --n-classes 2 --max-layer $num"
        echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esm_t$num/scope_40_208/ --n-classes 3 --max-layer $num"
        echo "conda run -n plm --no-capture-output python -m src.downstream.aa_cuml --data-path $BASE/datasets/scope_40_208.csv --embed-base $BASE/aa_embeddings/esm_t$num/scope_40_208/ --n-classes 8 --max-layer $num"
    done
} | xargs -P $NCORES -I {} bash -c "{}"
