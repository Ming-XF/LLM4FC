#!/bin/bash
#
# TimeLLM DeepSpeed ZeRO-2 training on CAUEEG2 (caueeg2)
# GPUs: 4  |  batch_size: 4 per GPU  |  total effective: 16
#

deepspeed --num_gpus=4 main.py \
    --model "TimeLLM" \
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
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 500 \
    --num_patches 19 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --dropout 0.1 \
    --save_steps 25 \
    --deepspeed \
    --deepspeed_config ds_config_zero2.json \
    --do_train \
    --do_evaluate \
    --do_test
