#!/bin/bash
#
# BNT training script
# Usage: ./scripts/train_BNT.sh <DATASET> <EARLY_STOP_METRIC>
#
# Examples:
#   ./scripts/train_BNT.sh DiseaseBeirut   AUC
#   ./scripts/train_BNT.sh GenderDS        AUC
#   ./scripts/train_BNT.sh AgeTUAB         Loss
#

DATASET=${1:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC>"}
EARLY_STOP_METRIC=${2:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC>"}

deepspeed --num_gpus=6 main.py \
    --model "BNT" \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --num_epochs 200 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
