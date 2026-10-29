#!/bin/bash
#
# EEG-Deformer 10s training on DiseaseTUEP (GPU 0, parallel-safe).
# Unified settings: 200 epochs / lr 1e-4 / patience 20 / val-AUC / seed 42.
# Run: sh scripts/train_deformer_tuep.sh
#

export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")/.." || exit 1

python main_LLM4FC.py \
    --dataset DiseaseTUEP \
    --data-dir llm4fc_data_10s_tuep \
    --gpu 0 \
    --max-epoch 200 \
    --lr 1e-4 \
    --batch-size 64 \
    --patience 20 \
    --seed 42 \
    --save-path ./save_llm4fc
