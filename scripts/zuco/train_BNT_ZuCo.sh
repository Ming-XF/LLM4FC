#!/bin/bash
#
# BNT DeepSpeed ZeRO-2 training on ZuCo (zuco)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "BNT" \
    --num_repeat 1 \
    --dataset 'ZuCo' \
    --data_dir "/data/datasets/ZuCo/ZuCo-TSR.npy" \
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
    --deepspeed_config scripts/deepspeed/BNT.json \
    --do_train \
    --do_evaluate \
    --do_test
