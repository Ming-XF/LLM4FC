#!/bin/bash
#
# FBNetGen DeepSpeed ZeRO-2 training on DS (ds)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "FBNetGen" \
    --num_repeat 1 \
    --dataset 'DS' \
    --data_dir "../data/DS/ds.npz" \
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
    --deepspeed_config scripts/deepspeed/FBNetGen.json \
    --do_train \
    --do_evaluate \
    --do_test
