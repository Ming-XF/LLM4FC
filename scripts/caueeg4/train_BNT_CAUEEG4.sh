#!/bin/bash
#
# BNT DeepSpeed ZeRO-2 training on CAUEEG4 (caueeg4)
# GPUs: 4  |  batch_size: 3 per GPU  |  total effective: 12
#

deepspeed --num_gpus=6 main.py \
    --model "BNT" \
    --num_repeat 1 \
    --dataset 'CAUEEG4' \
    --percentage 1 \
    --data_dir "../data/CAUEEG/caueeg4.npz" \
    --batch_size 3 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "F_score" \
    --deepspeed \
    --deepspeed_config scripts/deepspeed/BNT.json \
    --do_train \
    --do_evaluate \
    --do_test
