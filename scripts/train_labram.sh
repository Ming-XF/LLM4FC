#!/bin/bash
#
# LaBraM baseline (ICLR 2024) fine-tuning on LLM4FC disease datasets
# No external arguments; edit the variables below.
# Run: sh scripts/train_labram.sh
# Prerequisite: python baseline/LaBraM/dataset_maker/make_LLM4FC.py
#

TRAIN_DATASET="DiseaseCAUEEG"
OUT_DIR="out_llm4fc_caueeg"

cd baseline/LaBraM && python run_class_finetuning.py \
    --dataset "$TRAIN_DATASET" \
    --model labram_base_patch200_200 \
    --finetune checkpoints/labram-base.pth \
    --output_dir "$OUT_DIR" \
    --log_dir "$OUT_DIR/log" \
    --batch_size 64 \
    --lr 1e-4 \
    --warmup_epochs 5 \
    --epochs 200 \
    --early_stop_patience 20 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --seed 42 \
    --num_workers 8
