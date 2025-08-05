ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

for model in 6 12 30 33; do
    for l in $(seq 0 $((model))); do
        echo "python -m src.downstream.model --data-path $ROOT/datasets/deeploc2_bin.csv --embed-path $ROOT/embeddings/esm_t$model/deeploc2/layer_$l/ --function knn --task classification --binary"
    done
done | xargs -P 8 -I {} bash -c "{}"

for algo in "lr" "knn"; do
    echo "python -m src.downstream.model --data-path $ROOT/datasets/fluorescence.csv --embed-path $ROOT/embeddings/ohe/fluorescence/layer_0/ --function $algo --task regression"
    echo "python -m src.downstream.model --data-path $ROOT/datasets/stability.csv --embed-path $ROOT/embeddings/ohe/stability/layer_0/ --function $algo --task regression"
    echo "python -m src.downstream.model --data-path $ROOT/datasets/deeploc2.csv --embed-path $ROOT/embeddings/ohe/deeploc2/layer_0/ --function $algo --task classification"
    echo "python -m src.downstream.model --data-path $ROOT/datasets/deeploc2_bin.csv --embed-path $ROOT/embeddings/ohe/deeploc2/layer_0/ --function $algo --task classification --binary"
done | xargs -P 4 -I {} bash -c "{}"
