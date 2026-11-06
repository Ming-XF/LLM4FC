#!/bin/bash
#
# KnowledgeLLM top_k=4 零样本评估 driver: CAUEEG -> TUAB, seeds {42,43,44}。
# 单卡顺序评估，产出 metrics_KnowledgeLLM_s{S}_topk4_caueeg_to_tuab_DiseaseTUAB.json。
# Run: sh scripts/run_eval_KnowledgeLLM_topk4_zs.sh
#

for S in 42 43 44; do
  echo "===== zero-shot eval: topk4 seed=$S ====="
  sh scripts/eval_KnowledgeLLM_topk4_zs.sh "$S" \
    || echo "EVAL FAILED: topk4 seed=$S"
done

echo "All zero-shot evals finished."
