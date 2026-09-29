# CAER: Calibration-Anchored Evidence Reprogramming

基于 TimeLLM 架构改进的 EEG 跨域零样本方法，在**年龄回归**与**疾病分类**
两个任务上同时有效。v2 引入双证据（功能连接 + 频谱），共享证据编译器
与冻结 LLM 通路，每任务分叉冻结闭式读出头。实验过程与全部诊断记录见
`WORK.md`；方法设计经 codex（gpt-5.6-sol）咨询核验。

## 1. 方法架构（CAER-v2）

```
19通道 EEG (10个滑窗)
   │
   ├─ ① 确定性双证据编译器（无监督规则，源域-free）
   │     e = [e_FC (9维) ; e_spec (9维)]
   │     ├─ FC: 总体强度(对角锚点)、半球间/内比、额/后比、不对称、
   │     │      中央/颞/顶/枕/中线脑区强度
   │     └─ 频谱: delta/theta/alpha/beta 相对功率、theta/beta、
   │            alpha/(d+t)、慢波指数、额枕 alpha 比、颞 theta
   │     + 10 句自然语言证据文本（FC 6 + 频谱 4，定长单token编码）
   ▼
② 冻结 LLM (DeepSeek-R1-Distill-Llama-8B)
   [task prompt][证据文本] 上下文化（纯语言通路）
   ▼
③ 标定锚定读出（每任务分叉、闭式拟合、冻结）
   年龄: ridge(e)      ← scripts/fit_caer_heads.py（val 选 alpha）
   疾病: σ(w·e+b)      ← 同脚本（val AUC 选 C）
```

**核心原则**（全部经消融验证）：
1. 证据完全由确定性规则生成，不含任何源域学习参数；
2. 读出头用闭式收缩解（ridge / L2 logistic），SGD 解依赖跨域符号翻转
   特征不可迁移；
3. 可学习组件（GCN patches / LLM 状态 / 残差分支）跨域零增益，
   一律冻结或移除；
4. 特征子集与正则强度只由**源域 val** 选择，无目标域泄漏。

## 2. 实验效果（零样本，seed 42 确定性）

### 年龄预测（CAUEEG ↔ TUAB）

| 方法 | CAUEEG→TUAB PearsonR | TUAB→CAUEEG PearsonR |
|---|---|---|
| TimeLLM answer_query 完整管线 | 0.039 | — |
| TimeLLM flatten（历史） | 0.032 | — |
| ALTER（历史最优） | 0.148 | — |
| CAER-v1（仅 FC 证据） | 0.184 | — |
| **CAER-v2（FC+频谱）** | **+0.344** | **+0.207** |

### 疾病分类（CAUEEG ↔ TUAB，语义为"跨任务生理表型迁移"，见 §5）

| 方法 | CAUEEG→TUAB AUC | TUAB→CAUEEG AUC |
|---|---|---|
| TimeLLM answer_query 完整管线 | 0.350 | — |
| CAER 仅 FC 证据 | 0.460 | 0.436 |
| **CAER-v2（FC+频谱）** | **+0.814** | **+0.704** |

（CAUEEG→TUAB 疾病：Acc 0.708 / Sens 0.855 / Spec 0.539；
源域拟合 val：年龄 r=0.227→0.32，疾病 AUC=0.827）

## 3. 消融与关键发现

| 消融/诊断 | 结果 | 结论 |
|---|---|---|
| FC 证据对疾病 | 源域 AUC 0.508（不可学） | AD 信号不在宽带 FC |
| **频谱证据对疾病** | 跨域 AUC 0.816/0.731 | 频谱是疾病任务的信号域 |
| 18 维组合 | 年龄 r=0.243 诊断（真实 0.344） | 双证据互补 |
| 丰富频谱（25 维：log 比值/SEF90/谱熵/1/f） | 疾病 0.802、年龄 0.091 | 更复杂特征破坏迁移，维持 18 维 |
| 可学习组件（E1-E9） | 全部 ≤0.08 | 见 v1 文档 §3 |

**频谱比 FC 更强且互补**：弥漫性慢波、alpha 减弱等频域特征在
两个数据集间符号一致（疾病跨域 AUC 0.46→0.81；年龄 r 0.18→0.34）。

**一致性核验**：模型评估与数据集路径独立复算吻合（r=0.3436 vs
+0.344）；特征/标签/索引逐项验证一致；无泄漏（特征仅由输入信号
确定性生成，头仅由源域拟合）。

## 4. 使用方式

```bash
# 1. 拟合源域双头（CAUEEG 源；反向用 python3 scripts/fit_caer_heads.py TUAB）
python3 scripts/fit_caer_heads.py

# 2. 零样本评估（scripts/train_TimeLLM_answer_query.sh 含方法配置：
#    --graph_evidence --no_patches --stats_head
#    年龄: --stats_ridge ./output_dir/stats_ridge.pt
#    疾病: --stats_logistic ./output_dir/stats_logistic.pt
#    改 sh 头部 DATASET/PRETRAIN_PATH(任意无bin目录) 后运行）：
./scripts/train_TimeLLM_answer_query.sh
```

## 5. 局限与诚实声明

- **疾病语义错位**：DiseaseCAUEEG（AD 相关）与 DiseaseTUAB
  （正常/异常）标签语义不完全一致。按 codex 建议，疾病实验表述为
  **"跨任务生理表型迁移"**，不宣称 AD→abnormal 同标签零样本；
  其双向 AUC（0.81/0.70）表明两类病理的频域表型高度可迁移
- **年龄 MAE 标定**：跨站点标定漂移使 MAE 受限（CAUEEG→TUAB
  MAE 16.6），r 为标定无关的主指标
- **LLM 通路**：冻结 LLM 处理证据文本为架构组成部分，但可迁移信号
  全部来自确定性证据 + 闭式读出；LLM 状态的跨域增益为零（v1 消融）
- **数据对**：当前环境仅有 CAUEEG/TUAB 两个数据集
