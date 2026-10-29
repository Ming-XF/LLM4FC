#!/bin/bash
#
# EEGNet zero-shot evaluation script (single-GPU, no training)
# No external arguments; edit the variables below.
# Run: sh scripts/eval_EEGNet_zs.sh
#

TARGET_DATASET="DiseaseTUAB"
PRETRAIN_PATH="./output_dir/EEGNet_DiseaseCAUEEG_train"

# 10s 段口径（当前 EEGNet 官方结果口径）
export LLM4FC_DATA_10S=1

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
    --append "_caueeg" \
    --do_test
