NCORES=4

HOME="/scratch/SCRATCH_SAS/roman/SMTB"
MMSEQSDIR="$HOME/datasets/mmseqs"

# mkdir $MMSEQSDIR

# mmseqs createdb /wibicomfs/STBS/roman/uniref90.fasta $MMSEQSDIR/uniref90.db
for dataset in esol fluorescence stability deeploc2; do
    mkdir $HOME/${dataset}_tmp
    mmseqs easy-search $HOME/datasets/${dataset}.fasta $MMSEQSDIR/uniref90.db $HOME/${dataset}.m8 $HOME/${dataset}_tmp --format-output query,target,fident -e inf --max-seqs 100000 --max-seq-id 0 -s 7.5 --threads $NCORES
done
