#!/bin/bash
#
# OFA (8B Llama) 10s training on DiseaseTUAB (GPU 0, parallel-safe).
# Run: sh scripts/train_llama4ts_tuab_10s.sh
#


cd "$(dirname "$0")/../Classification" || exit 1
mkdir -p experiments_llm4fc

python src/main.py \
    --output_dir experiments_llm4fc \
    --comment "OFA 8B TUAB disease 10s" \
    --name DiseaseTUAB_10s \
    --records_file ofa_records_10s.xls \
    --data_dir ../llm4fc_data_10s_tuab \
    --data_class npz \
    --pattern _train \
    --val_pattern _val \
    --test_pattern _test \
    --epochs 200 \
    --lr 0.0001 \
    --patch_size 16 \
    --stride 16 \
    --optimizer RAdam \
    --d_model 4096 \
    --llm_path /root/autodl-tmp/LLM4FC/model/deepseek-r1-distill-llama-8B \
    --pos_encoding learnable \
    --task classification \
    --key_metric AUROC \
    --early_stop_patience 20 \
    --val_interval 1 \
    --batch_size 32 \
    --num_workers 4 \
    --seed 44 \
    --gpu 0 \
