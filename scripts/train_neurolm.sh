#!/bin/bash
#
# NeuroLM baseline (ICLR 2025) instruction tuning on LLM4FC datasets
# No external arguments; edit the variables below.
# Run: sh scripts/train_neurolm.sh
# Prerequisite: python baseline/NeuroLM/prepare_llm4fc.py --dataset <DS> for each dataset
#

TRAIN_DATASET="TUAB"
EVAL_DATASETS="TUAB CAUEEG TUEP"
EPOCHS=10
BATCH_SIZE=64
LR=5e-4
PATIENCE=3
OUT_DIR="baseline/NeuroLM/out_llm4fc_tuab"

python baseline/NeuroLM/train_instruction_llm4fc.py \
    --train_dataset "$TRAIN_DATASET" \
    --eval_datasets $EVAL_DATASETS \
    --NeuroLM_path baseline/NeuroLM/checkpoints/NeuroLM-B.pt \
    --dataset_dir baseline/NeuroLM/llm4fc_data \
    --out_dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --warmup_epochs 1 \
    --patience "$PATIENCE" \
    --text_max_len 80
