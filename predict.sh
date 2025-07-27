ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

for dataset in fluorescence stability; do
    for model in 6 12 30 33; do
        for l in $(seq 0 $((model))); do
            # echo "Running model $model with layer $l on dataset $dataset"
            python -m src.downstream.model --data-path $ROOT/datasets/$dataset.csv --embed-path $ROOT/embeddings/esm_t$model/$dataset/layer_$l/ --function lr --task regression --seed 42
        done
    done
done
