#!/bin/bash
#
# BNT training script
# Usage: ./scripts/train_BNT.sh   (dataset / early-stop 参数在脚本内部配置)
#
# Datasets:  DiseaseBeirut / GenderDS  (AUC, 0.001)
#            AgeCAUEEG / AgeTUAB      (MAE, 0.1)
#

# ── 实验参数（在脚本内部配置，不再从命令行传入）──
DATASET="AgeCAUEEG"              # 数据集: DiseaseBeirut/GenderDS/AgeCAUEEG/AgeTUAB 等
EARLY_STOP_METRIC="MAE"          # 早停指标: AUC(分类) / MAE(回归)

deepspeed --num_gpus=2 main.py \
    --model "BNT" \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --num_epochs 200 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
