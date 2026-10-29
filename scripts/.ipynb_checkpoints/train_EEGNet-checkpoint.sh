#!/bin/bash
#
# EEGNet training script — no external arguments; edit the variables below.
# Run: sh scripts/train_EEGNet.sh
#

DATASET="DiseaseTUEP"
EARLY_STOP_METRIC="AUC"

# H800 本机卡间 P2P 直连损坏（NCCL BROADCAST 无限挂起），必须禁用 P2P 走共享内存
export NCCL_P2P_DISABLE=1
# 10s 段口径（当前 EEGNet 官方结果口径）
export LLM4FC_DATA_10S=1

deepspeed --num_gpus=2 main.py \
    --model "EEGNet" \
    --num_epochs 200 \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --frequency 128 \
    --D 2 \
    --num_kernels 8 \
    --p1 4 \
    --p2 8 \
    --dropout 0.5 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
