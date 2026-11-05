#!/bin/bash
#
# KnowledgeLLM 融合 token 语义解码 (分析实验二, 纯 CPU)。
# Run: sh analysis/exp2/run_decode.sh
#

cd /root/autodl-tmp/LLM4FC || exit 1

python analysis/exp2/interpret_fusion_tokens.py
