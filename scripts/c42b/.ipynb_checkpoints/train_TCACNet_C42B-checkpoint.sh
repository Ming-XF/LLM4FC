#!/bin/bash
#
# TCACNet DeepSpeed ZeRO-2 training on C42B (c42b)
# GPUs: 4  |  batch_size: 3 per GPU  |  total effective: 12
#

deepspeed --num_gpus=6 main.py \
    --model "TCACNet" \
    --num_repeat 1 \
    --dataset 'C42B' \
    --data_dir "../data/C42B/C42B128.npy" \
    --batch_size 3 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --deepspeed \
    --deepspeed_config scripts/deepspeed/TCACNet.json \
    --do_train \
    --do_evaluate \
    --do_test
