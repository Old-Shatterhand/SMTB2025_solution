ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

for model in ankh-base ankh-large; do
    for dataset in fluorescence stability; do
        python src/plms/esm.py --model-name ElnaggarLab/$model --data-path $ROOT/datasets/$dataset.csv --output-path $ROOT/embeddings/$model/$dataset
    done
done

for dataset in deeploc2 deeploc2_bin; do
    # for model in esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D; do
    #     python -m src.plms.esm --model-name facebook/$model --data-path $ROOT/datasets/$dataset.csv --output-path $ROOT/embeddings/$model/$dataset
    # done
    for model in ankh-base ankh-large; do
        python -m src.plms.esm --model-name ElnaggarLab/$model --data-path $ROOT/datasets/$dataset.csv --output-path $ROOT/embeddings/$model/$dataset
    done
done
