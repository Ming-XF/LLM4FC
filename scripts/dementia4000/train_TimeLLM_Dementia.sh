#!/bin/bash
#
# TimeLLM training on Dementia4000
#
# Architecture (TimeLLM v2 — static FC):
#   SFC → GCN node encoder (channel-name embeddings) → Reprogramming
#   (cross-attention to text prototypes) → Frozen ChatGLM-6B
#   → Flatten + classifier (AD/MCI/SCD/NC)
#
# Trainable parameters (~77M):
#   - channel_embed_projection (4096→128): ~0.5M
#   - GCN + node_projection: ~24K
#   - mapping_layer: Linear(vocab→500): ~75M
#   - ReprogrammingLayer (Q/K/V/out): ~1M
#   - node_pos_embed (19×4096): ~155K
#   - output_projection (Flatten + Linear): ~10K
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
    --num_prototypes 500 \
    --num_patches 19 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --dropout 0.1 \
    --do_train \
    --do_evaluate \
    --do_test
