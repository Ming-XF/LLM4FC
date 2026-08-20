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
    --use_task_prompt \
    --num_gcn_layers 1 \
    --use_cvib \
    --cvib_mode vae \
    --cvib_beta 1e-3 \
    --num_prototypes 1000 \
    --token_order time_first \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --lora_target_modules "v_proj,o_proj" \
    --lora_num_layers 1 \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test

# ── LoRA / GC-LoRA ──
# To enable fine-tuning, add the following flags to the deepspeed command above
# (insert them after --lora_target_modules):
#
#   --use_lora \              # enable standard LoRA
#   --use_gc_lora \           # enable graph-conditioned LoRA (requires --use_lora)
#   --block_causal_mask \
#   --use_dataset_prompt \
#   --use_task_prompt \
#   --token_order node_first \
#   --use_cvib \
#   --cvib_mode vae \
#   --cvib_beta 1e-3 \
        



    