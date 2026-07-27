#!/bin/bash
#
# ALTER DeepSpeed ZeRO-2 training on Beirut (beirut)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "ALTER" \
    --num_repeat 1 \
    --dataset 'Beirut' \
    --data_dir "../data/Beirut/Beirut.npy" \
    --batch_size 4 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --num_heads 1 \
    --deepspeed \
    --deepspeed_config scripts/deepspeed/ALTER.json \
    --do_train \
    --do_evaluate \
    --do_test
