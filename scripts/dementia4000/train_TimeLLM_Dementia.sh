#!/bin/bash
#
# TimeLLM training on Dementia4000
#
# Architecture (based on Time-LLM ICLR 2024):
#   DFC sliding windows (graph patches) → BrainNetCNN (graph encoder)
#   → ReprogrammingLayer (cross-attention to text prototypes)
#   → Frozen ChatGLM-6B → mean-pool → classifier (4 classes)
#
# Trainable parameters (~76M):
#   - BrainNetCNN (E2E+E2N+N2G): ~200K
#   - Patch projection: 256→64: ~16K
#   - mapping_layer: Linear(150528→500): ~75M
#   - ReprogrammingLayer (Q/K/V/out projections): ~1M
#   - Classification head: Linear(4096→4): ~16K
#
# GPU memory: ~24 GB (single GPU, bf16, batch_size=2)
#   ChatGLM-6B (frozen): ~12 GB
#   Trainable params + activations: ~10-12 GB
#

python main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset 'Dementia4000' \
    --data_dir "../data/Dementia4000/Dementia4000.npz" \
    --batch_size 2 \
    --num_epochs 100 \
    --save_steps 25 \
    --drop_last False \
    --train_set 0.8 \
    --val_set 0.1 \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --eps 1e-8 \
    --warmup_steps 200 \
    --early_stop_patience 15 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Accuracy" \
    --d_model 64 \
    --num_heads 8 \
    --patch_stride 1 \
    --num_prototypes 500 \
    --llm_layers 28 \
    --dropout 0.1 \
    --do_train \
    --do_evaluate \
    --do_test
