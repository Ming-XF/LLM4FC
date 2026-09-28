# WORK.md — 年龄预测跨域零样本泛化研究进展

> 目标：基于 TimeLLM 架构改进，利用 LLM 泛化能力提升 EEG 年龄预测的
> 跨数据集零样本泛化（CAUEEG → TUAB 为主测试对）。每次实验后更新本文档
> 并 git 提交阶段性成果。

## 基线（当前最优与历史对照）

| 方法 | 源域测试 MAE | 源域测试 PearsonR | 零样本 MAE | 零样本 PearsonR |
|---|---|---|---|---|
| TimeLLM flatten（历史，d_model 64） | 7.55 | 0.342 | 16.70 | 0.032 |
| ALTER（历史） | — | — | 17.04 | 0.148 |
| TimeLLM answer_query（epoch 12 欠训练） | 8.27 | 0.022 | 16.86 | -0.051 |
| **TimeLLM answer_query（epoch 64 完整训练，当前基线）** | **7.70** | **0.309** | **17.98** | **0.039** |

结论：所有变体零样本均≈随机（PearsonR ≈ 0），失败不是源域欠拟合，
而是 GCN soft token 与原型映射缺乏语言语义锚点，特征本身跨域不可迁移。

## 研究假设

跨域失效的根因在**特征空间**：重编程层的 learnable 原型与 GCN 输出都
由源域监督形成，LLM 没有获得语言语义可解释的输入，其预训练知识未被利用。
修复方向（按 codex_output 路线 + 领域知识排序）：

| 编号 | 方法 | 作用层面 | 预期 |
|---|---|---|---|
| E1 | FC 秩归一化（--fc_rank_norm，每窗口边秩 → [-1,1]） | 输入（域不变表示） | 中 |
| E2 | 语义原型库（固定 LLM 编码的神经连接短语 + 小残差） | 特征空间（语言锚点） | 高 |
| E3 | 有序年龄区间原型头（年龄描述 embedding + 余弦 + 期望年龄 + EMD） | 输出空间（语言锚点） | 高 |
| E4 | 多视图一致性（原始 FC 与秩视图预测一致正则） | 正则（源域-only） | 中 |

## 实验协议

- **筛选**：AgeCAUEEG 训练 30 epoch（其余与完整协议相同），
  零样本评估 AgeTUAB；判定标准：PearsonR > 0.10 或 MAE < 16.5 为有信号
- **完整验证**：早停 patience 20、无 delta，训练至收敛后零样本 + 5-shot
- 每个方法的 checkpoint 独立备份至 `output_dir/TimeLLM_AgeCAUEEG_train_<编号>`

## 提交日志

### #1 — E1：FC 秩归一化输入（结果：负）
- 新增 `--fc_rank_norm` 参数；TimeLLMTrainer 对 DFC 每个窗口做边秩变换 → [-1,1]
- 筛选结果（30 epoch，早停于 28）：源域 best test MAE 8.14 / **PearsonR -0.0035**
  （基线 7.70 / 0.309）——秩变换破坏 GCN 对相关强度的利用，源域即损坏
- 零样本：MAE 17.20 / PearsonR 0.0755（基线 17.98 / 0.039），仍≈随机，无信号
- checkpoint: output_dir/TimeLLM_AgeCAUEEG_train_E1_rank/
- 结论：E1 单独无效，放弃（秩视图仅保留给 E4 作一致性正则的辅助视图）

### #2 — E2：语义原型库（结果：负）
- 新增 `--prototype_mode semantic`：45 个神经连接短语的冻结 LLM embedding
  作为 cross-attention 键值 + 可训练小残差；mapping_layer(vocab→1000) 不再创建
- 筛选结果（30 epoch）：源域 best test MAE 7.98 / PearsonR 0.195
  （基线同阶段 ~0.31，略弱但能学；显存 64.7GB，训练 62s/epoch）
- 零样本：MAE 17.60 / PearsonR -0.046（基线 17.98/0.039）——无信号，判负
- checkpoint: output_dir/TimeLLM_AgeCAUEEG_train_E2_semantic/
- 启示：固定语义键值使源域略降，但跨域未改善——特征空间锚定还不够，
  输出侧与输入侧仍需联合探索

### #3 — E3：有序年龄原型头（待跑）
- `--age_head ordinal --emd_lambda 0.1`：20 个年龄区间描述原型，
  余弦匹配 + 期望年龄 + EMD 有序正则
- 开始 30-epoch 筛选训练
