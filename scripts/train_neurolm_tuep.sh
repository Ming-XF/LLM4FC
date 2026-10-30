#!/bin/bash
#
# NeuroLM baseline (ICLR 2025) instruction tuning on LLM4FC datasets
# No external arguments; edit the variables below.
# Run: sh scripts/train_neurolm_tuep.sh
# Prerequisite: python baseline/NeuroLM/prepare_llm4fc.py --dataset <DS> for each dataset
#

TRAIN_DATASET="TUEP"
EVAL_DATASETS="TUEP CAUEEG TUAB"
EPOCHS=200
BATCH_SIZE=64
LR=1e-4
PATIENCE=20
OUT_DIR="baseline/NeuroLM/out_llm4fc_s44_tuep"

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
    --text_max_len 80 \
    --seed 44 \
    --gpu 1
