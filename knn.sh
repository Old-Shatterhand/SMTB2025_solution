ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

for model in ankh-base ankh-large; do
    for l in $(seq 0 48); do
        echo "python -m src.downstream.model --data-path $ROOT/datasets/deeploc2_bin.csv --embed-path $ROOT/embeddings/$model/deeploc2/layer_$l/ --function knn --task classification --binary"
    done
done | xargs -P 8 -I {} bash -c "{}"
