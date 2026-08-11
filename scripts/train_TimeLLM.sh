#!/bin/bash
#
# TimeLLM training script
# Usage: ./scripts/train_TimeLLM.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
#
# Examples:
#   ./scripts/train_TimeLLM.sh DiseaseBeirut   ../data/Beirut/beirut_disease.npy     AUC
#   ./scripts/train_TimeLLM.sh GenderDS        ../data/DS/ds_gender.npz              AUC
#   ./scripts/train_TimeLLM.sh AgeTUAB         ../data/TUAB/tuab_age.npz             Loss
#   ./scripts/train_TimeLLM.sh FutureFCTUEP    ../data/TUEP/tuep_futurefc.npz        Loss
#

DATASET=${1:?"Usage: $0 <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>"}
DATA_DIR=${2:?"Usage: $0 <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>"}
EARLY_STOP_METRIC=${3:?"Usage: $0 <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>"}

deepspeed --num_gpus=6 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --futurefc_aux_weight 1 \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 1000 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --num_gcn_layers 1 \
    --dropout 0.1 \
    --save_steps 25 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --batch_size 2 \
    --num_epochs 200 \
    --drop_last False \
    --schedule 'cos' \
    --use_dataset_prompt \
    --use_task_prompt \
    --use_lora \
    --use_gc_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "o_proj" \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
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

    



    