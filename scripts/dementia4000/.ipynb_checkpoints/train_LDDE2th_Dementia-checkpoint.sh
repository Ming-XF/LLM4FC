#!/bin/bash
#
# LDDE2th 3-fold training on Dementia_MMS (tech.md 最终方案)
#
# 架构：单 BNC + ChatGLM-6B (LoRA r=4) + 语义原型桥 + 知识掩膜反馈 + 跨模态一致性
# 训练：Phase A (1-10) BNC冻结 → Phase B (11-20) BNC解冻0.1×lr → Phase C (21-200) 全参数联合优化
#
# 显存需求：~46 GB (batch_size=2)，需 ≥48 GB GPU
# 预训练依赖：需先训练 DFCBNC（bash scripts/dementia_mms/train_DFCBNC_Dementia.sh）

python main.py \
    --model "LDDE2th" \
    --num_repeat 3 \
    --dataset 'Dementia4000' \
    --data_dir "../data/Dementia4000/Dementia4000.npz" \
    --batch_size 4 \
    --num_epochs 200 \
    --save_steps 50 \
    --drop_last False \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --eps 1e-8 \
    --warmup_steps 400 \
    --do_train \
    --do_evaluate \
    --do_test
