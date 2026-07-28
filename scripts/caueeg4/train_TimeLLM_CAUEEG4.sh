#!/bin/bash
#
# TimeLLM DeepSpeed ZeRO-2 training on CAUEEG4 (caueeg4)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset 'CAUEEG4' \
    --data_dir "../data/CAUEEG/caueeg4.npz" \
    --batch_size 4 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Accuracy" \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 500 \
    --num_patches 19 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --dropout 0.1 \
    --save_steps 25 \
    --deepspeed \
    --deepspeed_config scripts/deepspeed/TimeLLM.json \
    --do_train \
    --do_evaluate \
    --do_test
