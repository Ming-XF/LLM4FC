#!/bin/bash
#
# 知识 token 语义分析 (试验一): LLM 输入向量的词表余弦 top-5。
# Run: sh analysis/exp3/run_interpret.sh
#

cd /root/autodl-tmp/LLM4FC || exit 1
export LLM4FC_DATA_10S=1
export CUDA_VISIBLE_DEVICES=0

python analysis/exp3/interpret_knowledge_tokens.py
