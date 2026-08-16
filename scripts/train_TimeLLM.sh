#!/bin/bash
#
# TimeLLM training script
# Usage: ./scripts/train_TimeLLM.sh <DATASET> <EARLY_STOP_METRIC>
#
# Examples:
#   ./scripts/train_TimeLLM.sh DiseaseBeirut   AUC
#   ./scripts/train_TimeLLM.sh GenderDS        AUC
#   ./scripts/train_TimeLLM.sh AgeTUAB         Loss
#

DATASET=${1:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC>"}
EARLY_STOP_METRIC=${2:?"Usage: $0 <DATASET> <EARLY_STOP_METRIC>"}

deepspeed --num_gpus=6 main.py \
    --model "TimeLLM" \
    --num_epochs 200 \
    --num_repeat 1 \
    --save_steps 25 \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --dataset "$DATASET" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 1000 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --num_gcn_layers 1 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "./output_dir/TimeLLM_AgeCAUEEG_train" \
    --use_task_prompt \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj" \
    --token_domain_lambda 1.0 \
    --num_domains 2 \
    --domain_cls_hidden 256 \
    --domain_grl_weight 1.0 \
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
#   --use_token_domain_grl \      # enable token-level domain classifier + GRL

    



    