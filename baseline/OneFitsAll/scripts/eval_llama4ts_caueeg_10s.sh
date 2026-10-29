#!/bin/bash
#
# OFA (8B Llama) DiseaseCAUEEG 10 s — test-set evaluation only
# (loads model_best.pth from the 10 s training run).
# Run: sh scripts/eval_llama4ts_caueeg_10s.sh
#

export CUDA_VISIBLE_DEVICES=1

cd "$(dirname "$0")/../Classification" || exit 1
mkdir -p experiments_llm4fc

python src/main.py \
    --output_dir experiments_llm4fc \
    --comment "OFA 8B CAUEEG 10s test-eval" \
    --name DiseaseCAUEEG_10s \
    --records_file ofa_records_10s.xls \
    --data_dir ../llm4fc_data_10s \
    --data_class npz \
    --pattern _train \
    --val_pattern _val \
    --test_pattern _test \
    --epochs 50 \
    --lr 0.0005 \
    --patch_size 16 \
    --stride 16 \
    --optimizer RAdam \
    --d_model 4096 \
    --llm_path /root/autodl-tmp/LLM4FC/model/deepseek-r1-distill-llama-8B \
    --pos_encoding learnable \
    --task classification \
    --key_metric AUROC \
    --batch_size 32 \
    --num_workers 4 \
    --seed 42 \
    --test_only testset \
    --load_model experiments_llm4fc/DiseaseCAUEEG_10s_2026-10-28_23-05-57_4so/checkpoints/model_best.pth
