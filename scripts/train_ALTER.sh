#!/bin/bash
#
# ALTER training script
# Usage: ./scripts/train_ALTER.sh <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>
#
# Examples:
#   ./scripts/train_ALTER.sh DiseaseBeirut   AUC   0.001
#   ./scripts/train_ALTER.sh GenderDS        AUC   0.001
#   ./scripts/train_ALTER.sh AgeCAUEEG       MAE   0.1
#

DATASET=${1:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>"}
EARLY_STOP_METRIC=${2:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>"}
EARLY_STOP_MIN_DELTA=${3:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>"}

deepspeed --num_gpus=3 main.py \
    --model "ALTER" \
    --num_epochs 200 \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 5 \
    --few_shot_seed 42 \
    --pretrain_path "./output_dir/ALTER_AgeCAUEEG_train" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --num_heads 1 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
