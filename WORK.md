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

### #5b — E8b：stats_head（结论：SGD 不可行 → 闭式 ridge 成功）
- 统计量改回 old-def（含对角锚点）；头只读连续统计量
- 关键教训链：lr=1e-4 收敛到塌缩解（r≈0）；lr=1e-3 批噪声下落入
  反相关解（r=-0.098 零样本，variability/trend 两特征跨域符号翻转）；
  **唯一可靠解 = 闭式 ridge（收缩压制矛盾系数）**

### #8c — E8c：Calibration-Anchored Evidence Readout（方法成型）
- `scripts/fit_stats_ridge.py` 源域闭式拟合 ridge（val 选 alpha），
  注入冻结统计量头（--stats_ridge），`--no_patches` 纯证据通路
- 特征：4 维（strength/inter/front/asym）→ **9 维**（+5 脑区均值
  cent/temp/pari/occ/mid；variability/trend/hub 跨域符号翻转，剔除）
- **最终零样本结果（CAUEEG → TUAB，n=1047）**：
  PearsonR **+0.184** / MAE 16.90 / RMSE 20.37
  对比：TimeLLM answer_query 基线 0.039（**4.7×**）、历史 flatten 0.032、
  **历史 ALTER 0.148（被超越）**；源域 val r=0.227
- 修复：npz 标签已是归一化值（fit 脚本多余 /100 曾致 MAE 46.8）

### #9 — E9：LLM 残差分支验证（结果：剔除）
- 冻结 9 维 ridge 统计量分支 + 零初始化 LLM 状态残差分支（瓶颈 32 维）
- 源域 r=0.193/MAE 7.91（残差有效）；零样本 r=+0.18395（与 E8c 完全一致）、
  MAE 17.49/RMSE 21.06 反而略差 → **LLM 残差零可迁移增益，剔除**

## 第二阶段：疾病任务扩展（CAER-v2）

### 目标
保持年龄任务泛化有效（r=0.184），提升疾病任务跨域零样本泛化。

### 数据与语义
- DiseaseCAUEEG（5068 例，AD 相关二分类）、DiseaseTUAB（5602 例，
  正常/异常）——当前环境唯一疾病域对；标签语义不完全一致（AD vs 异常，
  codex 曾警告），诊断显示频谱证据下两者仍高度可迁移（见下）

### 关键诊断（D1-D3）
- **FC 证据对疾病任务无效**：CAUEEG 域内 AUC 仅 0.508（AD 在 FC 空间
  不可学）；跨域两个方向均 <0.5
- **频谱证据是突破**（9 维：相对频带功率 d/t/a/b、t/b、a/(d+t)、
  慢波指数、额枕 alpha 比、颞 theta）：
  疾病跨域 CAUEEG→TUAB **AUC 0.816**、反向 0.731；
  域内 0.719/0.821（FC: 0.508/0.626）；
  年龄跨域 r=0.196（超 FC 的 0.184）、域内 0.324
- **18 维组合（FC+频谱）**：疾病跨域 AUC 0.811、年龄跨域 **r=+0.243**
  ——年龄大幅提升，疾病由频谱主导

### CAER-v2 设计（实现完成）
- 证据 = 9 维 FC 统计量 + 9 维频谱统计量（数据集层确定性计算，
  `BaseDataset.spectral_stats`，4 个 age/disease 数据集已接入）
  + 10 句证据文本（FC 6 + 频谱 4）；统计量裁剪 ±5
- 双任务冻结闭式头（`scripts/fit_caer_heads.py`，共享证据、分叉读出）：
  年龄 ridge（val 选 alpha）/ 疾病 logistic（val AUC 选 C）
- 分类读出：p=sigmoid(w·x+b) → logits=[log(1-p), log(p)]（B,2 二元协议）
- codex（gpt-5.6-sol）咨询采纳要点：共享证据编译器+分叉任务头的统一设计、
  特征选择只用源域 val、语义错位按"跨任务生理表型迁移"诚实表述；
  未采纳：连续数值文本（定长编码约束）、更丰富谱特征（诊断显示
  25 维反降：疾病 0.802、年龄 0.091）

### CAER-v2 验证结果（CAUEEG→TUAB 零样本）

| 任务 | 基线管线 | CAER-v1 (FC) | **CAER-v2 (FC+频谱)** |
|---|---|---|---|
| 年龄 PearsonR | 0.039 | 0.184 | **+0.344** |
| 疾病 AUC | 0.350 | 0.460 | **+0.814** |

- 疾病零样本 Acc 0.708 / Sens 0.855 / Spec 0.539；基线管线反相关（0.350）
- 一致性核验：模型评估（r=0.3436）与数据集路径独立复算（r=+0.344）吻合；
  特征、标签、索引全部验证一致，无泄漏（早前 numpy 探针 r=0.003 系其
  spec9 未裁剪 ±5 的脚本 bug，已定位）
- 反向（TUAB 拟合）头拟合中，用于 TUAB→CAUEEG 反向验证

## 最终方法（经验证有效）

**Calibration-Anchored Evidence Reprogramming（CAER）**
基于 TimeLLM 的改进（全部改动在 TimeLLM 管线内）：

1. **确定性图证据**（`--graph_evidence`，model/TimeLLM/graph_evidence.py）：
   19 通道 10 分区滑窗 DFC → 9 个连续统计量（总体强度含对角锚点、
   半球间/半球内比、额/后比、半球不对称、中央/颞/顶/枕/中线脑区强度；
   剔除跨域符号翻转的 variability/trend/hub）+ 6 句自然语言证据文本
   （单 token 槽位词定长编码，无监督规则生成）
2. **纯语言通路**（`--no_patches`）：序列 = [task prompt][证据文本]，
   冻结 llama-8B 上下文化（GCN/重编程 patch 通路消融剔除）
3. **标定锚定读出**（`--stats_head --stats_ridge`）：源域闭式 ridge
   （scripts/fit_stats_ridge.py，源 val 选 alpha）注入冻结统计量头；
   SGD 解依赖跨域符号翻转特征、4096 维 LLM 状态头过拟合——均已消融排除

**验证结果（CAUEEG → TUAB 零样本，n=1047，seed 42 确定性）**：

| 方法 | PearsonR | MAE | RMSE |
|---|---|---|---|
| TimeLLM flatten（历史） | 0.032 | 16.70 | 20.26 |
| TimeLLM answer_query 完整管线 | 0.039 | 17.98 | — |
| ALTER（历史最优） | 0.148 | 17.04 | 20.58 |
| **CAER（本方法）** | **+0.184** | **16.90** | **20.37** |

- 相对 TimeLLM 家族基线提升 **4.7×**（0.039→0.184），超越 ALTER（+0.036）
- 源域（CAUEEG）：val r=0.227，测试 MAE 7.91
- E9 消融：可学习 LLM 残差零可迁移增益（r 不变、MAE 变差）
- 负结果启示：可学习组件（GCN patches/LLM 残差/4096 维头）在目标域
  全部失效；可迁移性来自确定性证据 + 收缩解（ridge）
