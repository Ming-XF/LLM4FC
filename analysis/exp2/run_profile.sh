#!/bin/bash
#
# 融合 token 透镜: 知识集归属与判定表 (纯 CPU)。
# Run: sh analysis/exp2/run_profile.sh
#

cd /root/autodl-tmp/LLM4FC || exit 1

python analysis/exp2/fusion_lens_profile.py
