#!/bin/bash
#
# EEGNet random-init control: zero-shot TUAB eval with UNTRAINED weights.
# The pretrain path deliberately does not exist, so load_model skips loading
# and the model stays randomly initialized. If the pipeline is leak-free,
# AUC should be ~0.5.
# Run: sh scripts/eval_EEGNet_randinit_control.sh
#

TARGET_DATASET="DiseaseTUAB"
PRETRAIN_PATH="./output_dir/EEGNet_RANDOM_INIT_DOES_NOT_EXIST"

python main.py \
    --model "EEGNet" \
    --dataset "$TARGET_DATASET" \
    --pretrain_path "$PRETRAIN_PATH" \
    --few_shot 0 \
    --append "_randinit" \
    --batch_size 16 \
    --frequency 128 \
    --D 2 \
    --num_kernels 8 \
    --p1 4 \
    --p2 8 \
    --dropout 0.5 \
    --do_test
