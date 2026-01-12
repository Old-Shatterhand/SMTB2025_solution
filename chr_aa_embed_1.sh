ROOT="/scratch/SCRATCH_SAS/roman/SMTB"

# Embed sequences (not to be paralellized to not exceed GPU RAM)
for model in esm_t6 esm_t12 esm_t30 esm_t33; do
    python -m src.plm --model-name $model --data-path $ROOT/datasets/binding.csv --output-path $ROOT/aa_embeddings/$model/binding --aa-level
    python -m src.plm --model-name $model --data-path $ROOT/datasets/scope_40_208.csv --output-path $ROOT/aa_embeddings/$model/scope_40_208 --aa-level
done
