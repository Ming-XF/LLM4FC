#!/bin/bash
#
# ALTER training script
# Usage: ./scripts/train_ALTER.sh <DATASET> <EARLY_STOP_METRIC>
#
# Examples:
#   ./scripts/train_ALTER.sh DiseaseBeirut   AUC
#   ./scripts/train_ALTER.sh GenderDS        AUC
#   ./scripts/train_ALTER.sh AgeCAUEEG       Loss
#

DATASET=${1:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC>"}
EARLY_STOP_METRIC=${2:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC>"}

deepspeed --num_gpus=6 main.py \
    --model "ALTER" \
    --num_epochs 200 \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "./output_dir/ALTER_AgeCAUEEG_train" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --num_heads 1 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
