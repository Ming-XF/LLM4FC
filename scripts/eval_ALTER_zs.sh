#!/bin/bash
#
# ALTER zero-shot evaluation script (single-GPU, no training)
# No external arguments; edit the variables below.
# Run: sh scripts/eval_ALTER_zs.sh
#

TARGET_DATASET="DiseaseTUAB"
PRETRAIN_PATH="./output_dir/ALTER_DiseaseCAUEEG_train"

python main.py \
    --model "ALTER" \
    --dataset "$TARGET_DATASET" \
    --pretrain_path "$PRETRAIN_PATH" \
    --few_shot 0 \
    --batch_size 16 \
    --num_heads 1 \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --do_test
