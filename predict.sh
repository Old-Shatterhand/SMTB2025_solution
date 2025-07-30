ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

#for dataset in fluorescence; do
#    #for model in 6 12 30 33; do
#    #    for l in $(seq 0 $((model))); do
#    #        # echo "Running model $model with layer $l on dataset $dataset"
#    #        echo "python -m src.downstream.model --data-path $ROOT/datasets/$dataset.csv --embed-path $ROOT/embeddings/esm_t$model/$dataset/layer_$l/ --function xgb --task regression --seed 42"
#    #    done
#    #done
for dataset in fluorescence stability; do
    for algo in lr xgb; do
        for model in ankh-base ankh-large; do
            for l in $(seq 0 48); do
                # echo "Running model $model with layer $l on dataset $dataset"
                echo "python -m src.downstream.model --data-path $ROOT/datasets/$dataset.csv --embed-path $ROOT/embeddings/$model/$dataset/layer_$l/ --function $algo --task regression --seed 42"
            done
        done
    done
done | xargs -P 8 -I {} bash -c "{}"
#done | xargs -P 8 -I {} bash -c "{}"
