#!/bin/bash
#
# OFA (8B Llama) 10s zero-shot evaluation from the CAUEEG training domain.
# Edit TARGET_DIR / name before each run (TUAB / TUEP).
# Run: sh scripts/eval_llama4ts_zs_10s.sh
#


cd "$(dirname "$0")/../Classification" || exit 1
mkdir -p experiments_llm4fc

TARGET_DIR="../llm4fc_data_10s_tuep"
EXP_NAME="ZS_CAUEEG_TUEP"

python src/main.py \
    --output_dir experiments_llm4fc \
    --comment "OFA 8B CAUEEG zero-shot" \
    --name "$EXP_NAME" \
    --records_file ofa_records_zs.xls \
    --data_dir "$TARGET_DIR" \
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
    --gpu 1 \
    --test_only testset \
    --load_model experiments_llm4fc/DiseaseCAUEEG_10s_2026-10-30_02-33-55_D7y/checkpoints/model_best.pth
