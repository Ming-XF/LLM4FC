#!/bin/bash
#
# LaBraM zero-shot cross-domain evaluation (single GPU, no training)
# No external arguments; edit the variables below.
# Run: sh scripts/eval_labram_zs.sh
#

TARGET_DATASET="DiseaseTUEP"
RESUME_DIR="out_llm4fc_tuep/checkpoint-best.pth"

cd baseline/LaBraM && python run_class_finetuning.py \
    --dataset "$TARGET_DATASET" \
    --model labram_base_patch200_200 \
    --resume "$RESUME_DIR" \
    --eval \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --batch_size 64 \
    --num_workers 8
