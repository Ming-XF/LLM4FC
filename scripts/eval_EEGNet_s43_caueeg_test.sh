#!/bin/bash
#
# EEGNet same-domain metric re-test: CAUEEG -> CAUEEG test.
# Reuses the saved checkpoint; writes metrics_*.json via the trainer.
# Run: sh scripts/eval_EEGNet_s43_caueeg_test.sh
#

TARGET_DATASET="DiseaseCAUEEG"
PRETRAIN_PATH="./output_dir/output_dir_eegnet_s43/EEGNet_DiseaseCAUEEG_train"
APPEND="_s43_caueeg_test"

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
