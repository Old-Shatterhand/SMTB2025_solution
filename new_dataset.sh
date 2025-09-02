ROOT="/scratch/SCRATCH_SAS/roman/SMTB"
NCORES=16
DATASET=esol

# Preprocess dataset
python -m src.datasets.$DATASET --save-path $ROOT/datasets/

# Embed sequences (not to be paralellized to not exceed GPU RAM)
for model in esm_t6 esm_t12 esm_t30 esm_t33 esm_t36 esmc-300m esmc-600m ankh-base ankh-large prostt5 prostt5 ohe; do
    python -m src.plms.plm --model-name $model --data-path $ROOT/datasets/$DATASET.csv --output-path $ROOT/embeddings/$model/$DATASET
done

# Train on ESM2 model embeddings
for num in 6 12 30 33 36; do
    for l in $(seq 0 $num); do
        for algo in lr knn 2nn; do
            echo "python -m src.downstream.model --data-path $ROOT/datasets/$DATASET.csv --embed-path $ROOT/embeddings/esm_t$num/$DATASET/layer_$l/ --function $algo --task regression"
        done
    done
done | xargs -P $NCORES -I {} bash -c "{}"

# Train on ESMC model embeddings
for l in $(seq 0 30); do
    for algo in lr knn 2nn; do
        echo "python -m src.downstream.model --data-path $ROOT/datasets/$DATASET.csv --embed-path $ROOT/embeddings/esmc_300m/$DATASET/layer_$l/ --function $algo --task regression"
    done
done | xargs -P $NCORES -I {} bash -c "{}"
for l in $(seq 0 36); do
    for algo in lr knn 2nn; do
        echo "python -m src.downstream.model --data-path $ROOT/datasets/$DATASET.csv --embed-path $ROOT/embeddings/esmc_600m/$DATASET/layer_$l/ --function $algo --task regression"
    done
done | xargs -P $NCORES -I {} bash -c "{}"

# Train on Ankh model embeddings
for model in base large; do
    for l in $(seq 0 48); do
        for algo in lr knn 2nn; do
            echo "python -m src.downstream.model --data-path $ROOT/datasets/$DATASET.csv --embed-path $ROOT/embeddings/ankh-$model/$DATASET/layer_$l/ --function $algo --task regression"
        done
    done
done | xargs -P $NCORES -I {} bash -c "{}"

# Train on Pro(s)tT5 model embeddings
for model in prostt5 prott5; do
    for l in $(seq 0 24); do
        for algo in lr knn 2nn; do
            echo "python -m src.downstream.model --data-path $ROOT/datasets/$DATASET.csv --embed-path $ROOT/embeddings/$model/$DATASET/layer_$l/ --function $algo --task regression"
        done
    done
done | xargs -P $NCORES -I {} bash -c "{}"

# Train on one-hot encodings
for algo in lr knn 2nn; do
    python -m src.downstream.model --data-path $ROOT/datasets/$DATASET.csv --embed-path $ROOT/embeddings/ohe/$DATASET/layer_0/ --function $algo --task regression
done
