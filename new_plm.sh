ROOT="/scratch/SCRATCH_SAS/roman/SMTB"
NCORES=16
PLM=prott5

for dataset in esol fluorescence stability deeploc2; do
    python -m src.plms.plm --model-name $PLM --data-path $ROOT/datasets/$dataset.csv --output-path $ROOT/embeddings/$PLM/$dataset/
done

for l in $(seq 0 25); do
    for dataset in esol fluorescence stability; do
        for algo in lr knn 2nn; do
            echo "python -m src.downstream.model --function $algo --data-path $ROOT/datasets/$dataset.csv --embed-path $ROOT/embeddings/$PLM/$dataset/layer_$l --task regression"
        done
    done
    for algo in lr knn; do
        echo "python -m src.downstream.model --function $algo --data-path $ROOT/datasets/deeploc2.csv --embed-path $ROOT/embeddings/$PLM/deeploc2/layer_$l --task classification"
        echo "python -m src.downstream.model --function $algo --data-path $ROOT/datasets/deeploc2_bin.csv --embed-path $ROOT/embeddings/$PLM/deeploc2/layer_$l --task classification --binary"
    done
    echo "python -m src.downstream.model --function 2nn --data-path $ROOT/datasets/deeploc2.csv --embed-path $ROOT/embeddings/$PLM/deeploc2/layer_$l"
done | xargs -P $NCORES -I {} bash -c "{}"
