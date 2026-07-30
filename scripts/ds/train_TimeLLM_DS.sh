#!/bin/bash
#
# TimeLLM DeepSpeed ZeRO-2 training on DS (ds)
# GPUs: 4  |  batch_size: 3 per GPU  |  total effective: 12
#

    --llm_type chatglm \
    --llm_path ./model/chatglm-6b \

    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \

deepspeed --num_gpus=6 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset 'DS' \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --data_dir "../data/DS/ds.npz" \
    --batch_size 3 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.7 \
    --val_set 0.15 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 1000 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --num_gcn_layers 1 \
    --dropout 0.1 \
    --save_steps 25 \
    --deepspeed \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --do_train \
    --do_evaluate \
    --do_test


deepspeed --num_gpus=4 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset 'DS' \
    --few_shot 1 \
    --few_shot_seed 42 \
    --pretrain_path "./output_dir/TimeLLM_CAUEEG2_train" \
    --data_dir "../data/DS/ds.npz" \
    --batch_size 2 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.7 \
    --val_set 0.15 \
    --schedule 'cos' \
    --early_stop_patience 25 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 1000 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --num_gcn_layers 1 \
    --dropout 0.1 \
    --save_steps 25 \
    --deepspeed \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --do_train \
    --do_evaluate \
    --do_test



