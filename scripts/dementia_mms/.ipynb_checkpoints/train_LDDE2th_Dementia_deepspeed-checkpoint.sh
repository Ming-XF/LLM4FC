#!/bin/bash
#
# LDDE2th 3-fold training on Dementia4000 with DeepSpeed ZeRO-3
#
# 架构：单 BNC + ChatGLM-6B (LoRA r=4) + 语义原型桥 + 知识掩膜反馈 + 跨模态一致性
# 训练：Phase B (1-10) 过渡期 → Phase C (11-200) 全参数联合优化
#
# 显存：ZeRO-3 将 LLM 12GB 参数分摊到 2×48GB GPU，每卡 ~24GB
# 预训练依赖：需先训练 DFCBNC（bash scripts/dementia4000/train_DFCBNC_Dementia.sh）

deepspeed --num_gpus=1 main.py \
    --model "LDDE2th" \
    --num_repeat 1 \
    --dataset 'Dementia_MMS' \
    --data_dir "../data/Dementia_MMS/Dementia_MMS.npz" \
    --batch_size 3 \
    --num_epochs 200 \
    --save_steps 50 \
    --drop_last False \
    --train_set 0.8 \
    --val_set 0.1 \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --eps 1e-8 \
    --warmup_steps 400 \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Accuracy" \
    --deepspeed \
    --deepspeed_config ds_config_zero2.json \
    --do_train \
    --do_evaluate \
    --do_test
