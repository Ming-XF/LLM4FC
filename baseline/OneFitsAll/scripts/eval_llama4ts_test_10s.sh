#!/bin/bash
#
# OFA (8B Llama) 10s test-set evaluation on the model's own training domain.
# Edit TARGET_DIR / EXP_NAME / LOAD_MODEL before each run.
# Run: sh scripts/eval_llama4ts_test_10s.sh
#

export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")/../Classification" || exit 1
mkdir -p experiments_llm4fc

TARGET_DIR="../llm4fc_data_10s"
EXP_NAME="DiseaseCAUEEG_10s"
LOAD_MODEL="experiments_llm4fc/DiseaseCAUEEG_10s_2026-10-29_11-16-31_ADc/checkpoints/model_best.pth"

python src/main.py \
    --output_dir experiments_llm4fc \
    --comment "OFA 8B 10s test-eval" \
    --name "$EXP_NAME" \
    --records_file ofa_records_10s.xls \
    --data_dir "$TARGET_DIR" \
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
    --batch_size 32 \
    --num_workers 4 \
    --seed 42 \
    --test_only testset \
    --load_model "$LOAD_MODEL"
