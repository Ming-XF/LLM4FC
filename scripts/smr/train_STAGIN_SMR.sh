#!/bin/bash
#
# STAGIN DeepSpeed ZeRO-2 training on SMR (smr)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "STAGIN" \
    --num_repeat 1 \
    --dataset 'SMR' \
    --data_dir "../data/SMR/SMR128.npy" \
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
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --d_model 64 \
    --window_size 50 \
    --window_stride 3 \
    --dynamic_length 600 \
    --num_layers 2 \
    --deepspeed \
    --deepspeed_config ds_config_zero2.json \
    --do_train \
    --do_evaluate \
    --do_test
