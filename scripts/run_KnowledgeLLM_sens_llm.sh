#!/bin/bash
#
# KnowledgeLLM LLM 层数敏感性 driver: {2,4,8,10} x seeds {42,43,44}。
# 层间串行、层内 3 seed 进程并行（每进程双卡 deepspeed，与主方法 3-seed
# 战役同一启动方式，共享 GPU0+GPU1）；端口按 29700+L*100+S 分配防冲突。
# Run: sh scripts/run_KnowledgeLLM_sens_llm.sh
#

for L in 2 4 8 10; do
  echo "===== LLM layers = $L : launching seeds 42/43/44 ====="
  PIDS=""
  for S in 42 43 44; do
    PORT=$((29700 + L * 100 + S))
    sh scripts/train_KnowledgeLLM_sens_llm.sh "$L" "$S" "$PORT" &
    PIDS="$PIDS $!"
  done
  # shellcheck disable=SC2086
  wait $PIDS || echo "layer $L: some seed run failed (check logs)"
  echo "===== LLM layers = $L : done ====="
done

echo "All sensitivity runs finished."
