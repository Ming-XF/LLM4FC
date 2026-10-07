#!/bin/bash
#
# TimeLLMKnowledge (领域知识 Token 编译器) training script
# No external arguments; edit the variables below.
# Run: sh scripts/train_TimeLLMKnowledge.sh
#

DATASET="AgeCAUEEG"
EARLY_STOP_METRIC="R2"

deepspeed --num_gpus=2 main.py \
    --model "TimeLLMKnowledge" \
    --num_epochs 200 \
    --num_repeat 1 \
    --save_steps 25 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --dataset "$DATASET" \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --llm_init pretrained \
    --use_task_prompt \
    --num_prototypes 1000 \
    --readout_layers 4 \
    --deepspeed \
    --deepspeed_config deepspeed/train_patch.json \
    --do_train \
    --do_evaluate \
    --do_test
