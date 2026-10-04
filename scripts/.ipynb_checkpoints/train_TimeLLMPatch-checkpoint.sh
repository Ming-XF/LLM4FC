#!/bin/bash
#
# TimeLLMPatch (raw-EEG patch + reprogramming + frozen LLM) training script
# Usage: ./scripts/train_TimeLLMPatch.sh <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA> [EXTRA_ARGS]
#
# Examples:
#   ./scripts/train_TimeLLMPatch.sh AgeCAUEEG   MAE   0.1
#   ./scripts/train_TimeLLMPatch.sh AgeCAUEEG   MAE   0.1  "--max_steps 2 --num_epochs 1"
#

DATASET=${1:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA> [EXTRA_ARGS]"}
EARLY_STOP_METRIC=${2:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA> [EXTRA_ARGS]"}
EARLY_STOP_MIN_DELTA=${3:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA> [EXTRA_ARGS]"}
EXTRA_ARGS=${4:-}

deepspeed --num_gpus=2 main.py \
    --model "TimeLLMPatch" \
    --num_epochs 200 \
    --num_repeat 1 \
    --save_steps 25 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --dataset "$DATASET" \
    --d_model 128 \
    --num_heads 8 \
    --d_ff 128 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --llm_init pretrained \
    --use_task_prompt \
    --num_prototypes 1000 \
    --patch_len 512 \
    --patch_stride 512 \
    --deepspeed \
    --deepspeed_config deepspeed/train_patch.json \
    --do_train \
    --do_evaluate \
    --do_test \
    $EXTRA_ARGS
