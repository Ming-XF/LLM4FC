#!/bin/bash
#
# KnowledgeLLM top_m 敏感性 driver: {8,16,64,128} x seeds {42,43,44}
# (锚点 32 = 主方法 a3a 已有, 不重跑)。
# 值间串行、值内 3 seed 进程并行（每进程双卡 deepspeed，与主方法 3-seed
# 战役同一启动方式，共享 GPU0+GPU1）；端口按 29850+TOPM*100+S 分配防冲突。
# Run: sh scripts/run_KnowledgeLLM_sens_topm.sh
#

for M in 8 16 64 128; do
  echo "===== top_m = $M : launching seeds 42/43/44 ====="
  PIDS=""
  for S in 42 43 44; do
    PORT=$((29850 + M * 100 + S))
    sh scripts/train_KnowledgeLLM_sens_topm.sh "$M" "$S" "$PORT" &
    PIDS="$PIDS $!"
  done
  # shellcheck disable=SC2086
  wait $PIDS || echo "top_m $M: some seed run failed (check logs)"
  echo "===== top_m = $M : done ====="
done

echo "All top_m sensitivity runs finished."
