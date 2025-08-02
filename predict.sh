ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

for model in 6 12 30 33; do
    for l in $(seq 0 $((model))); do
        echo "python -m src.downstream.model --data-path $ROOT/datasets/deeploc2_bin.csv --embed-path $ROOT/embeddings/esm_t$model/deeploc2/layer_$l/ --function lr --task classification --seed 42 --binary"
    done
done | xargs -P 8 -I {} bash -c "{}"

for model in ankh-base ankh-large; do
    for l in $(seq 0 48); do
        echo "python -m src.downstream.model --data-path $ROOT/datasets/deeploc2.csv --embed-path $ROOT/embeddings/$model/deeploc2/layer_$l/ --function lr --task classification --seed 42"
    done
done | xargs -P 8 -I {} bash -c "{}"
