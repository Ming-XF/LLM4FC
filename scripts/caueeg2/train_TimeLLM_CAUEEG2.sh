#!/bin/bash
#
# TimeLLM DeepSpeed ZeRO-2 training on CAUEEG2 (caueeg2)
# GPUs: 4  |  batch_size: 3 per GPU  |  total effective: 12
#

deepspeed --num_gpus=4 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset 'CAUEEG2' \
    --data_dir "../data/CAUEEG/caueeg2.npz" \
    --batch_size 3 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 500 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --num_gcn_layers 1 \
    --dropout 0.1 \
    --save_steps 25 \
    --deepspeed \
    --deepspeed_config scripts/deepspeed/TimeLLM.json \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --do_train \
    --do_evaluate \
    --do_test


    --llm_type chatglm \
    --llm_path ./model/chatglm-6b \

    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
