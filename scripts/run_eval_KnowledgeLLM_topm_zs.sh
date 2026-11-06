#!/bin/bash
#
# KnowledgeLLM top_m 零样本评估 driver: CAUEEG -> TUAB,
# top_m in {8,16,128} x seeds {42,43,44}，单卡顺序评估。
# 产出 metrics_KnowledgeLLM_s{S}_topm{M}_caueeg_to_tuab_DiseaseTUAB.json。
# Run: sh scripts/run_eval_KnowledgeLLM_topm_zs.sh
#

for M in 8 16 128; do
  for S in 42 43 44; do
    echo "===== zero-shot eval: topm=$M seed=$S ====="
    sh scripts/eval_KnowledgeLLM_topm_zs.sh "$M" "$S" \
      || echo "EVAL FAILED: topm=$M seed=$S"
  done
done

echo "All zero-shot evals finished."
