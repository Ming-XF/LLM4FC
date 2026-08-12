#!/bin/bash
#
# BrainNetCNN training script
# Usage: ./scripts/train_BrainNetCNN.sh <DATASET> <EARLY_STOP_METRIC>
#
# Examples:
#   ./scripts/train_BrainNetCNN.sh DiseaseBeirut   AUC
#   ./scripts/train_BrainNetCNN.sh GenderDS        AUC
#   ./scripts/train_BrainNetCNN.sh AgeTUAB         Loss
#

DATASET=${1:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC>"}
EARLY_STOP_METRIC=${2:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC>"}

deepspeed --num_gpus=6 main.py \
    --model "BrainNetCNN" \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --batch_size 4 \
    --num_epochs 200 \
    --drop_last False \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
