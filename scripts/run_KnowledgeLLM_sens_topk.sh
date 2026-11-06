#!/bin/bash
#
# KnowledgeLLM top_k 敏感性 driver: {2,4,16,32} x seeds {42,43,44}
# (锚点 8 = 主方法 a3a 已有, 不重跑)。
# 值间串行、值内 3 seed 进程并行（每进程双卡 deepspeed，与主方法 3-seed
# 战役同一启动方式，共享 GPU0+GPU1）；端口按 29800+TOPK*100+S 分配防冲突。
# Run: sh scripts/run_KnowledgeLLM_sens_topk.sh
#

for K in 2 4 16 32; do
  echo "===== top_k = $K : launching seeds 42/43/44 ====="
  PIDS=""
  for S in 42 43 44; do
    PORT=$((29800 + K * 100 + S))
    sh scripts/train_KnowledgeLLM_sens_topk.sh "$K" "$S" "$PORT" &
    PIDS="$PIDS $!"
  done
  # shellcheck disable=SC2086
  wait $PIDS || echo "top_k $K: some seed run failed (check logs)"
  echo "===== top_k = $K : done ====="
done

echo "All top_k sensitivity runs finished."
