#!/bin/bash
#
# 知识银行精确复现抽取 (s42 训练配置, 全量 8B bf16 前向)。
# Run: sh analysis/exp2/run_extract.sh
#

cd /root/autodl-tmp/LLM4FC || exit 1
export CUDA_VISIBLE_DEVICES=0

python analysis/exp2/extract_bank_reproduce.py
