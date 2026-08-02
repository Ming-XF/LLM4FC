#!/bin/bash
#
# ALTER DeepSpeed ZeRO-2 training on Beirut (beirut)
# GPUs: 4  |  batch_size: 2 per GPU  |  total effective: 12
#

deepspeed --num_gpus=6 main.py \
    --model "ALTER" \
    --num_repeat 1 \
    --dataset 'Beirut' \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --data_dir "../data/Beirut/beirut.npy" \
    --batch_size 2 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.7 \
    --val_set 0.15 \
    --schedule 'cos' \
    --early_stop_patience 25 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --num_heads 1 \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
