#!/bin/bash
#
# BrainNetCNN training script — no external arguments; edit the variables below.
# Run: sh scripts/train_BrainNetCNN.sh
#

DATASET="AgeTUAB"
EARLY_STOP_METRIC="R2"

deepspeed --num_gpus=2 main.py \
    --model "BrainNetCNN" \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --num_epochs 200 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
