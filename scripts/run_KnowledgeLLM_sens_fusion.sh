#!/bin/bash
#
# KnowledgeLLM fusion tokens 敏感性 driver: {4,8,32,64} x seeds {42,43,44}
# (锚点 16 = 主方法 a3a 已有, 不重跑)。
# 值间串行、值内 3 seed 进程并行（每进程双卡 deepspeed，与主方法 3-seed
# 战役同一启动方式，共享 GPU0+GPU1）；端口按 29920+FUSION*100+S 分配防冲突。
# Run: sh scripts/run_KnowledgeLLM_sens_fusion.sh
#

for F in 4 8 32 64; do
  echo "===== fusion_tokens = $F : launching seeds 42/43/44 ====="
  PIDS=""
  for S in 42 43 44; do
    PORT=$((29920 + F * 100 + S))
    sh scripts/train_KnowledgeLLM_sens_fusion.sh "$F" "$S" "$PORT" &
    PIDS="$PIDS $!"
  done
  # shellcheck disable=SC2086
  wait $PIDS || echo "fusion $F: some seed run failed (check logs)"
  echo "===== fusion_tokens = $F : done ====="
done

echo "All fusion_tokens sensitivity runs finished."
