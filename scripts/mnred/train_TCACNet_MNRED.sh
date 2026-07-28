#!/bin/bash
#
# TCACNet DeepSpeed ZeRO-2 training on MNRED (mnred)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "TCACNet" \
    --num_repeat 1 \
    --dataset 'MNRED' \
    --data_dir "../data/MNRED/mnred.npy" \
    --batch_size 4 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Accuracy" \
    --deepspeed \
    --deepspeed_config scripts/deepspeed/TCACNet.json \
    --do_train \
    --do_evaluate \
    --do_test
