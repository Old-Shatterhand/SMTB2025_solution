ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

python -m src.plms.plm --model-name ohe --data-path $ROOT/datasets/deeploc2.csv --output-path $ROOT/embeddings/ohe/deeploc2
