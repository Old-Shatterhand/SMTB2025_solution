ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

for model in ankh-base ankh-large; do
    for dataset in fluorescence stability; do
        python src/plms/esm.py --model-name ElnaggarLab/$model --data-path $ROOT/datasets/$dataset.csv --output-path $ROOT/embeddings/$model/$dataset
    done
done
