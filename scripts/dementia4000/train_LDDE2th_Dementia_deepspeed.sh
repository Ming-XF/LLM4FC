#!/bin/bash
#
# LDDE2th single training on Dementia4000 with DeepSpeed ZeRO-2
#
# 架构：BNC 特征提取 → ChatGLM-6B (LoRA r=4) 编码 → 分类头
# 训练：LoRA + BNC + 投影层 + 分类头 全参数联合优化
#
# 显存：ZeRO-2 将 LLM 12GB 参数分摊到 4×GPU
# 预训练依赖：需先训练 DFCBNC（bash scripts/dementia4000/train_DFCBNC_Dementia.sh）


deepspeed --num_gpus=4 main.py \
    --model "LDDE2th" \
    --num_repeat 1 \
    --dataset 'Dementia4000' \
    --data_dir "../data/Dementia4000/Dementia4000.npz" \
    --save_steps 50 \
    --batch_size 2 \
    --num_epochs 50 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --eps 1e-8 \
    --warmup_steps 400 \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --deepspeed \
    --deepspeed_config ds_config_zero2.json \
    --do_train \
    --do_evaluate

# deepspeed --num_gpus=4 main.py \
#     --model "LDDE2th" \
#     --num_repeat 1 \
#     --dataset 'Dementia4000' \
#     --data_dir "../data/Dementia4000/Dementia4000.npz" \
#     --max_steps 20 \
#     --save_steps 1 \
#     --batch_size 1 \
#     --num_epochs 3 \
#     --drop_last False \
#     --train_set 0.6 \
#     --val_set 0.2 \
#     --schedule 'cos' \
#     --optimizer 'Adam' \
#     --learning_rate 1e-3 \
#     --weight_decay 1e-4 \
#     --eps 1e-8 \
#     --warmup_steps 400 \
#     --early_stop_patience 20 \
#     --early_stop_min_delta 0.001 \
#     --early_stop_metric "Loss" \
#     --deepspeed \
#     --deepspeed_config ds_config_zero2.json \
#     --do_train \
#     --do_evaluate


python main.py \
      --model "LDDE2th" \
      --dataset 'Dementia4000' \
      --data_dir "../data/Dementia4000/Dementia4000.npz" \
      --batch_size 1 \
      --do_test
