#!/bin/bash
#
# GCDGCN DeepSpeed ZeRO-2 training on CAUEEG4 (caueeg4)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "GCDGCN" \
    --num_repeat 1 \
    --dataset 'CAUEEG4' \
    --data_dir "../data/CAUEEG/caueeg4.npz" \
    --batch_size 4 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --deepspeed \
    --deepspeed_config scripts/deepspeed/GCDGCN.json \
    --do_train \
    --do_evaluate \
    --do_test
