ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

for model in esm_t6 esm_t12 esm_t30 esm_t33 ankh-base ankh-large; do
    python -m src.plms.plm --model-name $model --data-path $ROOT/datasets/deeploc2.csv --output-path $ROOT/embeddings/$model/deeploc2
done
