ROOT="/scratch/SCRATCH_SAS/roman/SMTB"
NCORES=64

{
    # Train on ESM2 model embeddings
    for num in 6 12 30 33 36; do
        for l in $(seq 0 $num); do
            for algo in lr knn; do
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/esm_t$num/scope_40_208/layer_$l/ --function $algo --n-classes 3"
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/esm_t$num/scope_40_208/layer_$l/ --function $algo --n-classes 8"
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/binding.csv --embed-path $ROOT/aa_embeddings/esm_t$num/binding/layer_$l/ --function $algo --n-classes 2"
            done
        done
    done

    # Train on ESMC model embeddings
    for l in $(seq 0 30); do
        for algo in lr knn; do
            echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/esmc_300m/scope_40_208/layer_$l/ --function $algo --n-classes 3"
            echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/esmc_300m/scope_40_208/layer_$l/ --function $algo --n-classes 8"
            echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/binding.csv --embed-path $ROOT/aa_embeddings/esmc_300m/binding/layer_$l/ --function $algo --n-classes 2"
        done
    done
    for l in $(seq 0 36); do
        for algo in lr knn; do
            echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/esmc_600m/scope_40_208/layer_$l/ --function $algo --n-classes 3"
            echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/esmc_600m/scope_40_208/layer_$l/ --function $algo --n-classes 8"
            echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/binding.csv --embed-path $ROOT/aa_embeddings/esmc_600m/binding/layer_$l/ --function $algo --n-classes 2"
        done
    done

    # Train on Ankh model embeddings
    for model in base large; do
        for l in $(seq 0 48); do
            for algo in lr knn; do
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/ankh-$model/scope_40_208/layer_$l/ --function $algo --n-classes 3"
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/ankh-$model/scope_40_208/layer_$l/ --function $algo --n-classes 8"
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/binding.csv --embed-path $ROOT/aa_embeddings/ankh-$model/binding/layer_$l/ --function $algo --n-classes 2"
            done
        done
    done

    # Train on Pro(s)tT5 model embeddings
    for model in prostt5 prott5; do
        for l in $(seq 0 24); do
            for algo in lr knn; do
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/$model/scope_40_208/layer_$l/ --function $algo --n-classes 3"
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/$model/scope_40_208/layer_$l/ --function $algo --n-classes 8"
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/binding.csv --embed-path $ROOT/aa_embeddings/$model/binding/layer_$l/ --function $algo --n-classes 2"
            done
        done
    done

    # Train on one-hot encodings
    for algo in lr knn; do
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/ohe/scope_40_208/layer_$l/ --function $algo --n-classes 3"
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/scope_40_208.csv --embed-path $ROOT/aa_embeddings/ohe/scope_40_208/layer_$l/ --function $algo --n-classes 8"
                echo "python -m src.downstream.aa_model --data-path $ROOT/datasets/binding.csv --embed-path $ROOT/aa_embeddings/ohe/binding/layer_$l/ --function $algo --n-classes 2"
    done
} | xargs -P $NCORES -I {} bash -c "{}"