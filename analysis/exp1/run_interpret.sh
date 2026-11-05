#!/bin/bash
#
# KnowledgeLLM 重编程选择性分析 (TimeLLM Fig.5(e) 同款)。
# 单卡推理，产物在 analysis/results/。
# Run: sh analysis/run_interpret.sh
#

cd /root/autodl-tmp/LLM4FC || exit 1
export LLM4FC_DATA_10S=1
export CUDA_VISIBLE_DEVICES=0

python analysis/exp1/interpret_reprogramming.py
