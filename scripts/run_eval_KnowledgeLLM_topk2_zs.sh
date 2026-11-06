#!/bin/bash
#
# KnowledgeLLM top_k=2 零样本评估 driver: CAUEEG -> TUAB, seeds {42,43,44}。
# 单卡顺序评估，产出 metrics_KnowledgeLLM_s{S}_topk2_caueeg_to_tuab_DiseaseTUAB.json。
# Run: sh scripts/run_eval_KnowledgeLLM_topk2_zs.sh
#

for S in 42 43 44; do
  echo "===== zero-shot eval: topk2 seed=$S ====="
  sh scripts/eval_KnowledgeLLM_topk2_zs.sh "$S" \
    || echo "EVAL FAILED: topk2 seed=$S"
done

echo "All zero-shot evals finished."
