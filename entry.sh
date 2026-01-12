#!/usr/bin/env bash

# echo "Initialize conda"
# conda init bash
# source $HOME/.bashrc

echo "Go Home"
cd $HOME/PLMBench

# echo "Activate PLM environment"
# conda activate plm

SCRATCH="/scratch/chair_kalinina/s8rojoer/PLM"

echo "Run training"
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_10_3M.txt --output-dir ./models --model-name esm_t6_10_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_09_3M.txt --output-dir ./models --model-name esm_t6_09_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_08_3M.txt --output-dir ./models --model-name esm_t6_08_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_07_3M.txt --output-dir ./models --model-name esm_t6_07_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_06_3M.txt --output-dir ./models --model-name esm_t6_06_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_05_3M.txt --output-dir ./models --model-name esm_t6_05_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_04_3M.txt --output-dir ./models --model-name esm_t6_04_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_03_3M.txt --output-dir ./models --model-name esm_t6_03_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_02_3M.txt --output-dir ./models --model-name esm_t6_02_3M
conda run -n plm --no-capture-output python train.py --data-file $SCRATCH/train_files/train_01_3M.txt --output-dir ./models --model-name esm_t6_01_3M
