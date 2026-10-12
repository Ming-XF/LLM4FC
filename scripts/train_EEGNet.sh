#!/bin/bash
#
# EEGNet training script — no external arguments; edit the variables below.
# Run: sh scripts/train_EEGNet.sh
#

DATASET="DiseaseCAUEEG"
EARLY_STOP_METRIC="AUC"

deepspeed --num_gpus=2 main.py \
    --model "EEGNet" \
    --num_epochs 200 \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --frequency 128 \
    --D 2 \
    --num_kernels 8 \
    --p1 4 \
    --p2 8 \
    --dropout 0.5 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
