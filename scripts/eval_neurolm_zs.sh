#!/bin/bash
#
# NeuroLM baseline zero-shot evaluation (official weights, no training)
# No external arguments; edit the variables below.
# Run: sh scripts/eval_neurolm_zs.sh
#

EVAL_DATASETS="TUEP"
NEUROLM_PATH="baseline/NeuroLM/checkpoints/NeuroLM-B.pt"
OUT_DIR="baseline/NeuroLM/out_llm4fc_zeroshot"

python baseline/NeuroLM/train_instruction_llm4fc.py \
    --eval_only \
    --eval_datasets $EVAL_DATASETS \
    --NeuroLM_path "$NEUROLM_PATH" \
    --dataset_dir baseline/NeuroLM/llm4fc_data \
    --out_dir "$OUT_DIR" \
    --text_max_len 80
