# TimeLLMKnowledge v2 — AgeCAUEEG 实验结果（2026-10-07）

## 一、结论速览

v2（全局+局部知识 + patch 编码 + 捷径解除）在 AgeCAUEEG 上：
**Test R² = 0.193，MAE = 7.21 年** —— 本战役（Phase5/Knowledge）中冻结 LLM
路径**首次**学到有效年龄信号（v1 及之前全部 R²≈0）。

| | v1（旧设计） | v2（本次） |
|---|---|---|
| Test R² | -0.008 | **+0.193** |
| Test MAE | 8.16 年 | **7.21 年** |
| Best val R² | 0.0003 | **0.189** |
| Train loss 末值 | 0.031（平台） | 0.0015（真实拟合） |
| 早停轮数 | 34/200 | 43/200（耗时 64 min，78.6s/epoch） |

参照：b8 池化直连（无 LLM）跨域 R² 0.29-0.37 —— v2 域内 0.19，方向正确，
仍有距离。

## 二、关键诊断（v1 失败原因）

- 梯度失衡：mapping_layer/readout/head 梯度范数 10~250；知识路径
  （phrase_projection/value_weights/reprogram.q_proj）仅 1e-5~1e-4，
  小 6-7 个数量级 → 模型走"READOUT→常数预测"捷径，统计量不被训练。
- v2 修复后（bypass 直连监督）：知识路径梯度抬升到 1e-1~1e-2（冒烟验证）。
- 另有 nan 修复：0.3% 全平样本 → spectral_stats 0/0 → 已加 +1e-12 + 模型内
  nan_to_num（见 data/dataset.py 与 TimeLLMKnowledge.py，git diff 可见）。

## 三、训练后 gate 值（checkpoint 读出）

| 参数 | 初始 | 训练后 | 解读 |
|---|---|---|---|
| num_gate（数值分支，softplus） | 1.0 | 1.005 | 两分支保持均衡 |
| raw_gate（patch 分支，softplus） | 1.0 | 1.006 | 未偏向任一分支 |
| bypass_gate（λ，直接乘） | 0.1 | 0.109 | 直连权重基本不变 |
| phrase_beta（β） | 0.25 | 0.252 | phrase 锚点稳定 |

学习主要发生在分支内部权重（编译器/原型/头），gate 均衡，无压死/失控。

## 四、运行配置（复现）

- 脚本：`sh scripts/train_TimeLLMKnowledge.sh`（DATASET=AgeCAUEEG，EARLY_STOP_METRIC=R2）
- deepspeed：`deepspeed/train_patch.json`，**单卡 32 / 全局 64**，lr=1e-4，ZeRO-2，fp32
- 2×97GB 显卡，显存峰值每卡 78.8GB
- 日志：`log_dir/train_TimeLLMKnowledge_AgeCAUEEG.log`（logger 独占，干净）
- checkpoint：`output_dir/TimeLLMKnowledge_AgeCAUEEG_train/`（best + final-epoch-200）
- 注意：v2 序列 240 tokens，**单卡 48 会 OOM**（冻结 LLM 32 层激活 ≈100GB+），
  32 是当前上限附近；梯度检查点方案试过但按指示已回退。

## 五、已实现的 v2 设计要点（改动文件）

- `model/TimeLLMKnowledge/local_stats.py`（新）：10s 段频谱/形态统计量 torch 实现
  （与 numpy 版同公式；4s 窗 50% overlap；先聚合 band power 再算比值）
- `model/TimeLLMKnowledge/knowledge_compiler.py`：q = g_num·RMSNorm(W_i·s_i)
  + g_raw·RMSNorm(R_i·Stem(EEG块))；phrase 移出 q（冻结 phrase_bank）
- `model/TimeLLMKnowledge/TimeLLMKnowledge.py`：PatchStem、6×10s 局部知识、
  phrase 4096 维 gated 融合、捷径解除全套（READOUT 固定 " age" token、层权重
  固定均匀、head bias 冻结 0.5 先验、bypass 头 λ=0.1）
- `config.py`：6 个 BooleanOptionalAction 开关（默认全开），与 --readout_layers 同模式

## 六、待做消融（未跑，下次开机继续）

| 消融 | sh 脚本加 | 目的 |
|---|---|---|
| 关 raw 分支 | `--no-use_raw_branch` | patch 分支贡献归因 |
| 关数值分支 | `--no-use_numeric_branch` | 统计量分支贡献归因 |
| 关局部知识 | `--no-use_local_knowledge` | 局部 6×10s 增益归因 |
| 关 bypass | `--no-use_knowledge_bypass` | 直连监督是否必要 |
| 放开 bias/层权重 | `--no-freeze_head_bias` / `--learn_layer_weights` | 捷径回归诊断 |

每个约 40-60 分钟。之后按 codex 实验顺序：尺度消融 → phrase 融合对比 →
局部逐步加（FC→频谱→形态）→ 跨域零样本/few-shot。

验收判据（codex）：shuffle/置零知识必须显著退化；std(pred)/std(label) 摆脱≈0；
线性 probe 能从 q 或 stem 预测年龄。
