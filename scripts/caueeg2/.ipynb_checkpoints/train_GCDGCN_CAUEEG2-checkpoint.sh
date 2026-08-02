#!/bin/bash
#
# GCDGCN DeepSpeed ZeRO-2 training on CAUEEG2 (caueeg2)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "GCDGCN" \
    --num_repeat 1 \
    --dataset 'CAUEEG2' \
    --data_dir "../data/CAUEEG/caueeg2.npz" \
    --batch_size 4 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --eps 1e-8 \
    --early_stop_patience 25 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --deepspeed \
    --deepspeed_config ds_config_zero2.json \
    --do_train \
    --do_evaluate \
    --do_test
