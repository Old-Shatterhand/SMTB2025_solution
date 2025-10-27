ROOT="/scratch/SCRATCH_SAS/roman/SMTB"
NCORES=128
SEED=42

# python -m src.plm --model-name prott5 --data-path $HOME/datasets/scope_40_208.csv --output-path $HOME/embeddings/prott5/scope_40_208/

for level in superfamily fold; do
    # Train on ESM2 model embeddings
    for num in 6 12 30 33 36; do
        for l in $(seq 0 $num); do
            echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/esm_t$num/scope_40_208/layer_$l/ --top-k 4"
            for algo in rf lr; do
                echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/esm_t$num/scope_40_208/layer_$l/ --top-k 4 --function $algo"
            done
        done
    done

    # Train on ESMC model embeddings
    for l in $(seq 0 30); do
        echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/esmc_300m/scope_40_208/layer_$l/ --top-k 4"
        for algo in rf lr; do
            echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/esmc_300m/scope_40_208/layer_$l/ --top-k 4 --function $algo"
        done
    done
    for l in $(seq 0 36); do
        echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/esmc_600m/scope_40_208/layer_$l/ --top-k 4"
        for algo in rf lr; do
            echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/esmc_600m/scope_40_208/layer_$l/ --top-k 4 --function $algo"
        done
    done

    # Train on Ankh model embeddings
    for model in base large; do
        for l in $(seq 0 48); do
            echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/ankh-$model/scope_40_208/layer_$l/ --top-k 4"
            for algo in rf lr; do
                echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/ankh-$model/scope_40_208/layer_$l/ --top-k 4 --function $algo"
            done
        done
    done

    # Train on Pro(s)tT5 model embeddings
    for model in prostt5 prott5; do
        for l in $(seq 0 24); do
            echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/$model/scope_40_208/layer_$l/ --top-k 4"
            for algo in rf lr; do
                echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/$model/scope_40_208/layer_$l/ --top-k 4 --function $algo"
            done
        done
    done

    # Train on one-hot encodings
    echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/ohe/scope_40_208/layer_$l/ --top-k 4"
    for algo in rf lr; do
        echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/ohe/scope_40_208/layer_$l/ --top-k 4 --function $algo"
    done

    for k in 6 8 10 15 20; do
        # Train on Ankh model embeddings
        for l in $(seq 0 48); do
            echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/ankh-base/scope_40_208/layer_$l/ --top-k $k"
            echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/ankh-large/scope_40_208/layer_$l/ --top-k $k"
            for algo in rf lr; do
                echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/ankh-base/scope_40_208/layer_$l/ --top-k $k --function $algo"
                echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/ankh-large/scope_40_208/layer_$l/ --top-k $k --function $algo"
            done
        done

        for model in prostt5 prott5; do
            # Train on ProtT5 model embeddings
            for l in $(seq 0 24); do
                echo "python -m src.downstream.correlation --level $level --embed-path $ROOT/embeddings/$model/scope_40_208/layer_$l/ --top-k $k"
                for algo in rf lr; do
                    echo "python -m src.downstream.scope_model --level $level --embed-path $ROOT/embeddings/$model/scope_40_208/layer_$l/ --top-k $k --function $algo"
                done
            done
        done
    done
done | xargs -P $NCORES -I {} bash -c "{}"
