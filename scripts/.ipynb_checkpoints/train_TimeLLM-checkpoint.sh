#!/bin/bash
#
# TimeLLM training script
# Usage: ./scripts/train_TimeLLM.sh <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>
#
# Examples:
#   ./scripts/train_TimeLLM.sh DiseaseBeirut   AUC   0.001
#   ./scripts/train_TimeLLM.sh GenderDS        AUC   0.001
#   ./scripts/train_TimeLLM.sh AgeTUAB         MAE   0.1
#

DATASET=${1:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>"}
EARLY_STOP_METRIC=${2:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>"}
EARLY_STOP_MIN_DELTA=${3:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>"}

deepspeed --num_gpus=6 main.py \
    --model "TimeLLM" \
    --num_epochs 200 \
    --num_repeat 1 \
    --save_steps 25 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --dataset "$DATASET" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --d_model 128 \
    --num_heads 8 \
    --d_ff 128 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --llm_init random \
    --use_task_prompt \
    --num_gcn_layers 1 \
    --num_prototypes 1000 \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
