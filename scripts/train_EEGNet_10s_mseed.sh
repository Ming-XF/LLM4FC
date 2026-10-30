#!/bin/bash
#
# EEGNet 10s multi-seed training (plain torch, single GPU).
# Edit SEED / DATASET between runs (sequential; 42 协议: batch 24, lr 1e-4).
# Run: sh scripts/train_EEGNet_10s_mseed.sh
#

SEED=44
DATASET="DiseaseTUEP"

cd /root/autodl-tmp/LLM4FC || exit 1

export LLM4FC_DATA_10S=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python main.py \
    --model "EEGNet" \
    --num_epochs 200 \
    --num_repeat 1 \
    --save_steps 25 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_metric "AUC" \
    --dataset "$DATASET" \
    --model_dir "output_dir/output_dir_eegnet_s${SEED}" \
    --append "_s${SEED}" \
    --batch_size 24 \
    --learning_rate 1e-4 \
    --few_shot 0 \
    --frequency 128 \
    --D 2 \
    --num_kernels 8 \
    --p1 4 \
    --p2 8 \
    --dropout 0.5 \
    --do_train \
    --do_evaluate \
    --do_test
