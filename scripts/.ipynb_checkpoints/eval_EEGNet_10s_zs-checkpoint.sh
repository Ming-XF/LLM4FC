#!/bin/bash
#
# EEGNet 10s zero-shot evaluation script (single-GPU, no training)
# Edit TARGET_DATASET / PRETRAIN_PATH / append before each run.
# Run: sh scripts/eval_EEGNet_10s_zs.sh
#

export LLM4FC_DATA_10S=1
export CUDA_VISIBLE_DEVICES=1

TARGET_DATASET="DiseaseTUEP"
PRETRAIN_PATH="./output_dir_eegnet10s/EEGNet_DiseaseTUAB_train"
APPEND="_10s_tuab"

python main.py \
    --model "EEGNet" \
    --dataset "$TARGET_DATASET" \
    --pretrain_path "$PRETRAIN_PATH" \
    --few_shot 0 \
    --batch_size 16 \
    --frequency 128 \
    --D 2 \
    --num_kernels 8 \
    --p1 4 \
    --p2 8 \
    --dropout 0.5 \
    --append "$APPEND" \
    --do_test
