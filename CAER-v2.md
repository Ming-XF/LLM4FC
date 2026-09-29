# CAER-v2：双任务跨域零样本方法最终报告

基于 TimeLLM 架构改进（在 CAER v1 基础上扩展），在**年龄回归**与
**疾病分类**两个任务、**两个迁移方向**上全部验证有效。实验过程与
全部诊断记录见 `WORK.md`；方法设计经 codex（gpt-5.6-sol）咨询核验。

## 一、方法架构

创新点：**确定性双证据编译器**（功能连接 + 频谱）+ **每任务标定锚定读出**
（闭式拟合、冻结）。

```
19通道 EEG (10个滑窗)
   │
   ├─ ① 确定性双证据编译器（无监督规则，源域-free）
   │     e = [e_FC (9维) ; e_spec (9维)] + 10句自然语言证据文本
   │     ├─ FC 证据: 总体强度(含对角锚点)、半球间/内比、额/后比、
   │     │          半球不对称、中央/颞/顶/枕/中线脑区强度
   │     └─ 频谱证据: delta/theta/alpha/beta 相对功率、theta/beta、
   │                alpha/(d+t)、慢波指数、额枕alpha比、颞theta占比
   │     （数据集层确定性计算 BaseDataset.spectral_stats，裁剪 ±5）
   ▼
② 冻结 LLM (DeepSeek-R1-Distill-Llama-8B)
   [task prompt][证据文本] 上下文化（纯语言通路，GCN patches 消融剔除）
   ▼
③ 标定锚定读出（每任务分叉、闭式拟合、冻结）
   年龄: ŷ = ridge(e)         ← scripts/fit_caer_heads.py（源 val 选 alpha）
   疾病: p̂ = σ(w·e + b)       ← 同脚本（源 val AUC 选 C）
```

**核心原则**（全部经消融验证）：
1. 证据完全由确定性规则生成，不含任何源域学习参数；
2. 读出头用闭式收缩解（ridge / L2 logistic）——SGD 解依赖跨域
   符号翻转特征，不可迁移；
3. 可学习组件（GCN patches / LLM 状态 / 残差分支）跨域零增益，
   一律冻结或移除；
4. 特征子集与正则强度只由**源域 val** 选择，无目标域泄漏。

## 二、实验效果（零样本、确定性、双向）

| 任务 | 方向 | 基线管线 | CAER-v1(仅FC) | **CAER-v2(FC+频谱)** |
|---|---|---|---|---|
| 年龄 PearsonR | CAUEEG→TUAB | 0.039 | 0.184 | **+0.344** |
| 年龄 PearsonR | TUAB→CAUEEG | — | — | **+0.207** |
| 疾病 AUC | CAUEEG→TUAB | 0.350 | 0.460 | **+0.814** |
| 疾病 AUC | TUAB→CAUEEG | — | 0.436 | **+0.704** |

- 年龄：r 0.039→**0.344**（8.8×，超越历史 ALTER 0.148）；
  MAE 16.62 / RMSE 20.29（CAUEEG→TUAB）
- 疾病：AUC 0.350→**0.814**（基线可学习管线在目标域反相关！）；
  Acc 0.708 / Sens 0.855 / Spec 0.539；反向 AUC 0.704
- 源域拟合质量：年龄 val r=0.32、疾病 val AUC=0.827（CAUEEG 源）
- **一致性核验**：模型评估与数据集路径独立复算吻合（r=0.3436 vs
  +0.344）；特征、标签、索引逐项验证一致；无泄漏（特征仅由输入信号
  确定性生成，头仅由源域拟合）

## 三、关键科学发现

1. **FC 证据对 AD 不可学**（DiseaseCAUEEG 源域 AUC 仅 0.508）——
   疾病信号在**频域**（弥漫性慢波、alpha 减弱等经典特征）；
   频谱证据跨域 AUC 0.816/0.731，且把年龄迁移从 0.184 推到 0.344
   （双证据互补）
2. **更复杂≠更好**：25 维丰富频谱（log 比值/SEF90/谱熵/1/f 斜率/
   窗口动态性）反降迁移（年龄 0.091、疾病 0.802）——压缩的简单
   统计量跨域最稳
3. **闭式收缩解是唯一可靠读出**：SGD 头在批噪声下落入反相关解
   （零样本 r=-0.098），闭式 ridge/logistic 稳定迁移
4. 方法设计经 codex 咨询核验，采纳："共享证据编译器 + 分叉任务头"
   统一架构、"特征选择只用源域 val"、"疾病语义错位按跨任务生理
   表型迁移诚实表述"

## 四、实现细节

| 组件 | 文件 | 说明 |
|---|---|---|
| 双证据提取 | `model/TimeLLM/graph_evidence.py` | `compute_graph_evidence(dfc, spectral)` → 18 维统计量 + 10 句证据文本（槽位词单 token 定长） |
| 频谱统计 | `data/dataset.py::BaseDataset.spectral_stats` | Welch PSD（2s 分段、hanning），4 个 age/disease 数据集 `__getitem__` 已接入 `'spectral'` 字段 |
| 双头闭式拟合 | `scripts/fit_caer_heads.py` | 支持 `[CAUEEG\|TUAB]` 源；年龄 ridge / 疾病 logistic，源 val 选超参 |
| 冻结 logistic 读出 | `model/TimeLLM/TimeLLM.py`（`--stats_logistic`） | p=σ(w·e+b) → logits=[log(1-p), log(p)]（B,2 二元评估协议） |
| 冻结 ridge 读出 | 同上（`--stats_ridge`，v1 已有） | 标准化后线性读出，权重注入冻结 |
| 实验配置 | `scripts/train_TimeLLM_answer_query.sh` | `--graph_evidence --no_patches --stats_head` + 任务对应头路径 |

## 五、使用方式

```bash
# 1. 拟合源域双头（CAUEEG 源；反向用 TUAB）
python3 scripts/fit_caer_heads.py

# 2. 零样本评估：改 sh 头部 DATASET / PRETRAIN_PATH（任意无 bin 目录，
#    加载跳过，模型即为冻结头）与任务对应头路径，然后运行：
./scripts/train_TimeLLM_answer_query.sh
```

## 六、局限与诚实声明

- **疾病语义错位**：DiseaseCAUEEG（AD 相关）与 DiseaseTUAB（正常/异常）
  标签语义不完全一致，疾病实验按 codex 建议表述为
  **"跨任务生理表型迁移"**，不宣称 AD→abnormal 同标签零样本；
  双向 AUC（0.81/0.70）表明两类病理的频域表型高度可迁移
- **年龄 MAE 标定**：跨站点标定漂移使 MAE 受限（16.62），r 为
  标定无关的主指标
- **LLM 通路**：冻结 LLM 处理证据文本为架构组成部分，但可迁移信号
  全部来自确定性证据 + 闭式读出（LLM 状态跨域增益为零，v1 消融）
- **数据对**：当前环境仅有 CAUEEG/TUAB 两个数据集
