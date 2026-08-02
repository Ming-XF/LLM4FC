#!/bin/bash
#
# BrainNetCNN DeepSpeed ZeRO-2 training on TUEP (tuep)
# GPUs: 6  |  batch_size: 2 per GPU  |  total effective: 12
#

deepspeed --num_gpus=6 main.py \
    --model "BrainNetCNN" \
    --num_repeat 1 \
    --dataset 'TUEP' \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --data_dir "../data/TUEP/tuep.npz" \
    --batch_size 2 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 25 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
