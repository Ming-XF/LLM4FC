#!/bin/bash
#
# 融合 token x 知识集余弦相似度热图 (纯 CPU)。
# Run: sh analysis/exp2/run_cosine.sh
#

cd /root/autodl-tmp/LLM4FC || exit 1

python analysis/exp2/cosine_heatmap.py
