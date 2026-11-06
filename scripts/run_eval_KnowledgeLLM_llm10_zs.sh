#!/bin/bash
#
# KnowledgeLLM LLM层数=10 零样本评估 driver: CAUEEG -> TUAB, seeds {42,43,44}。
# 单卡顺序评估，产出 metrics_KnowledgeLLM_s{S}_llm10_caueeg_to_tuab_DiseaseTUAB.json。
# Run: sh scripts/run_eval_KnowledgeLLM_llm10_zs.sh
#

for S in 42 43 44; do
  echo "===== zero-shot eval: L10 seed=$S ====="
  sh scripts/eval_KnowledgeLLM_llm10_zs.sh "$S" \
    || echo "EVAL FAILED: L10 seed=$S"
done

echo "All zero-shot evals finished."
