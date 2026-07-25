#!/bin/bash
#
# TimeLLM DeepSpeed ZeRO-2 training on Dementia4000 (4 GPUs)
#
# Architecture (TimeLLM v2 — static FC):
#   SFC → GCN node encoder (channel-name embeddings) → Reprogramming
#   (cross-attention to text prototypes) → Frozen ChatGLM-6B
#   → Flatten + classifier (AD/MCI/SCD/NC)
#
# ZeRO-2 shards optimizer states + gradients across GPUs (model params
# are replicated). ChatGLM-6B (frozen) is 12 GB → not sharded by ZeRO-2,
# but the trainable mapping_layer (~75M, ~150 MB fp32) is tiny.
# ZeRO-2 mainly helps by offloading Adam states for the trainable params.
#
# GPU memory: ~18-20 GB per GPU (bf16, batch_size=2 per GPU)
#   ChatGLM-6B (frozen, bf16): ~12 GB
#   Trainable params (ZeRO-2 sharded): ~150 MB
#   Activations (batch_size=2): ~6-8 GB
#   Total effective batch size = 2 × 4 = 8

deepspeed --num_gpus=4 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset 'Dementia4000' \
    --data_dir "../data/Dementia4000/Dementia4000.npz" \
    --batch_size 4 \
    --num_epochs 10 \
    --save_steps 5 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-4 \
    --eps 1e-8 \
    --weight_decay 0 \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 500 \
    --num_patches 19 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --dropout 0.1 \
    --deepspeed \
    --deepspeed_config ds_config_zero2.json \
    --do_train \
    --do_evaluate \
    --do_test


deepspeed --num_gpus=4 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset 'Dementia4000' \
    --data_dir "../data/Dementia4000/Dementia4000.npz" \
    --batch_size 4 \
    --num_epochs 10 \
    --save_steps 5 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-4 \
    --eps 1e-8 \
    --weight_decay 0 \
    # --warmup_steps 200 \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --d_model 64 \
    --num_heads 8 \
    --num_prototypes 500 \
    --num_patches 19 \
    --d_ff 128 \
    --gcn_hidden 128 \
    --dropout 0.1 \
    --deepspeed \
    --deepspeed_config ds_config_zero2.json \
    --do_train \
    --do_evaluate \
    --do_test
