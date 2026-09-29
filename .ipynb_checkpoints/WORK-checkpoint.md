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

### #3 — E3：有序年龄原型头（v1 结果：失败，已修复重跑）
- `--age_head ordinal --emd_lambda 0.1`：20 个年龄区间描述原型，
  点积匹配 + 期望年龄 + EMD 有序正则
- v1 筛选（30 epoch）：源域 MAE 11.46 / PearsonR -0.03，几乎没学——
  原因：h 与原型双归一化后 4096 维余弦量级 ~0.02，softmax 梯度极弱
- 修复：h_answer 不归一化（原型仍单位化），logits 量级 ~1，
  与线性头同尺度；dtype 修复（头计算显式 fp32）
- v2(点积+EMD)/v3(白化)/v4(归一化)/v5(投影)/v6(19维瓶颈读出) 全部失败：
  软最大化饱和或瓶颈容量不足，30 epoch 内源域无法拟合
- v7（线性头 + h_answer 对齐 4096 维 bin 短语原型，λ=0.01）：
  源域 best test MAE 7.96 / PearsonR 0.242（略低于基线 0.31，正则代价合理）
  零样本：**MAE 16.51 / PearsonR 0.021**——MAE 迄今最低（标定改善），
  但相关性仍≈0，无泛化信号，判负
- checkpoint: output_dir/TimeLLM_AgeCAUEEG_train_E3v7_align/
- 启示：单模块（输入/特征/输出侧）都无法撼动 PearsonR≈0；
  后续转向组合与更强的方法级设计

### #4 — E4：多视图一致性（结果：负）
- `--mv_consistency --mv_lambda 0.1`：原始 FC 与秩视图经共享 GCN+重编程层
  的 soft token 语义一致（源域-only，不经过 LLM 二次前向）
- 筛选结果（30 epoch）：源域 best test MAE 8.00 / PearsonR 0.153
- 零样本：MAE 17.66 / PearsonR -0.031——无信号，判负
- checkpoint: output_dir/TimeLLM_AgeCAUEEG_train_E4_mv/

**单模块阶段小结（E1-E4 全部判负）**：零样本 PearsonR 全部在 ±0.08 内，
单点修复（输入/特征/输出/正则任一层面）无法撼动跨域失效。

### #5 — E5：Graph-to-Text 确定性图证据（待跑）
- `--graph_evidence`：每样本 5 条确定性图统计证据（总强度/半球间耦合/
  额叶连通/时间变异性/变化趋势）量化成自然语言，以冻结 LLM embedding
  文本 token 形式插在 DFC patches 与 <answer> 之间
- 证据生成完全无源域监督参数（源域-free），语言锚定，构造上跨域稳定
- 槽位词均为 llama 单 token（定长证据序列，无需掩码）

### 关键诊断（2026-09-29 晨）——证据信号与失效机制

1. **确定性统计量在目标域有信号**：6 统计量 ridge（目标域内拟合）
   TUAB test r=+0.184 / CAUEEG test r=+0.174
2. **跨域固定映射**（源域拟合 ridge → 目标域）：**old-def（子矩阵均值含
   对角锚点）r=+0.074~0.083**；new-def（严格边掩码）r≈0.006~0.042 →
   **对角=1 是重要的跨站点锚点**；样本内排序特征无益（0.004）
3. **3 级量化文本毁掉信号**：连续 0.083 → 量化 0.017（E5/E6/E7 失效根因）
4. **LLM 证据状态跨域 ridge r≈0.01**：冻结 LLM 对量化文本的上下文化
   不产生额外可迁移信号（源域有 0.25 但不可迁移）
5. **4096 维头过拟合**淹没 6 维统计量信号（E8 的 [states⊕stats] 头
   在目标域输出大噪声，r=-0.04）
6. **6 维头优化不足**：Adam lr=1e-4 下需 ~65 epoch 才收敛
   （30 epoch loss 0.052 vs ridge 最优 0.008）

### #5b — E8b：stats_head（运行中，100 epoch）
- 统计量改回 old-def（含对角锚点）；头只读 6 维连续统计量
- 预期：源域 r≈0.17、零样本 r≈+0.07~0.08（确定性迁移下限）
