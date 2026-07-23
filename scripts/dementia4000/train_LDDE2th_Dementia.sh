#!/bin/bash
#
# LDDE2th single training on Dementia4000 (80/10/10 split)
#
# 架构：BNC 特征提取 → ChatGLM-6B (LoRA r=4) 编码 → 分类头
# 训练：LoRA + BNC + 投影层 + 分类头 全参数联合优化
#
# 显存需求：~35 GB (batch_size=4)，需 ≥48 GB GPU
# 预训练依赖：需先训练 DFCBNC（bash scripts/dementia4000/train_DFCBNC_Dementia.sh）

python main.py \
    --model "LDDE2th" \
    --num_repeat 1 \
    --dataset 'Dementia4000' \
    --data_dir "../data/Dementia4000/Dementia4000.npz" \
    --batch_size 4 \
    --num_epochs 200 \
    --save_steps 50 \
    --drop_last False \
    --train_set 0.8 \
    --val_set 0.1 \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --eps 1e-8 \
    --warmup_steps 400 \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Accuracy" \
    --do_train \
    --do_evaluate \
    --do_test
