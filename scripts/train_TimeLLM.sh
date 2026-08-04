#!/bin/bash
#
# TimeLLM training script
# Usage: ./scripts/train_TimeLLM.sh <DATASET> <DATA_DIR>
#
# Examples:
#   ./scripts/train_TimeLLM.sh DiseaseBeirut   ../data/Beirut/beirut_disease.npy
#   ./scripts/train_TimeLLM.sh GenderDS        ../data/DS/ds_gender.npz
#   ./scripts/train_TimeLLM.sh AgeTUAB         ../data/TUAB/tuab_age.npz
#   ./scripts/train_TimeLLM.sh FutureFCTUEP    ../data/TUEP/tuep_futurefc.npz
#

DATASET=${1:?"Usage: $0 <DATASET> <DATA_DIR>"}
DATA_DIR=${2:?"Usage: $0 <DATASET> <DATA_DIR>"}

deepspeed --num_gpus=6 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --data_dir "$DATA_DIR" \
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
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 25 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
