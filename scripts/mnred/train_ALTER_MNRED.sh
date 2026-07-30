#!/bin/bash
#
# ALTER DeepSpeed ZeRO-2 training on MNRED (mnred)
# GPUs: 4  |  batch_size: 3 per GPU  |  total effective: 12
#

deepspeed --num_gpus=6 main.py \
    --model "ALTER" \
    --num_repeat 1 \
    --dataset 'MNRED' \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --finetune_epochs 10 \
    --data_dir "../data/MNRED/mnred.npy" \
    --batch_size 3 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --num_heads 1 \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
