#!/bin/bash
#
# EEGNet same-domain metric re-test: TUAB -> TUAB test.
# Reuses the saved checkpoint; writes metrics_*.json via the trainer.
# Run: sh scripts/eval_EEGNet_s44_tuab_test.sh
#

TARGET_DATASET="DiseaseTUAB"
PRETRAIN_PATH="./output_dir/output_dir_eegnet_s44/EEGNet_DiseaseTUAB_train"
APPEND="_s44_tuab_test"

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
    --append "$APPEND" \
    --do_test
