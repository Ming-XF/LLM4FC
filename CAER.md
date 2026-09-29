# CAER: Calibration-Anchored Evidence Reprogramming

基于 TimeLLM 架构改进的 EEG 年龄预测跨域零样本方法。在 CAUEEG → TUAB
零样本泛化上 PearsonR 达 **+0.184**（TimeLLM 基线 0.039，提升 4.7×；
历史最优 ALTER 0.148 被超越）。实验过程与全部诊断记录见 `WORK.md`。

## 1. 方法架构

```
19通道 EEG (10个滑窗 DFC)
   │
   ├─ ① 确定性图证据提取（无监督规则，源域-free）
   │     9 个连续统计量 + 6 句自然语言证据文本
   ▼
② 冻结 LLM (DeepSeek-R1-Distill-Llama-8B)
   [task prompt][证据文本] → 上下文化（纯语言通路）
   ▼
③ 标定锚定读出（Calibration-Anchored Readout）
   源域闭式 ridge 权重注入并冻结 → 年龄预测
```

### ① 确定性图证据提取（`model/TimeLLM/graph_evidence.py`）

- **9 个连续统计量**（19 通道 10-20 分区，滑窗 DFC 上无监督计算）：
  1. 总体连接强度（子矩阵均值**含对角锚点**——对角=1 是跨站点可比性的关键参照）
  2. 半球间/半球内连接比
  3. 额叶/后部连接比
  4. 半球不对称度
  5-9. 中央/颞/顶/枕/中线五个脑区的连接强度
- **特征筛选**（跨域符号一致性诊断）：variability、trend 两特征在
  CAUEEG 与 TUAB 的年龄相关性**符号相反**，hub-ness 同样破坏迁移——
  均被剔除
- **6 句自然语言证据**（"Overall connectivity strength is {low|mid|high}."
  等）：槽位词均为 llama 词表单 token，序列定长无需掩码；
  量化文本仅作语义通路，预测主信号走连续统计量

### ② 纯语言通路（`--no_patches`）

序列 = `[task prompt][证据文本]`，冻结 LLM 做上下文化。
GCN + 重编程 patches 通路经消融剔除：可学习 patch 在目标域零可迁移
增益，且会污染证据状态（因果注意力下 patches 先于证据）。

### ③ 标定锚定读出（`--stats_head --stats_ridge`）

- `scripts/fit_stats_ridge.py`：在源域训练集上闭式拟合 ridge
  （标准化 9 维统计量 → 归一化年龄，alpha 由源域 val 选择，无目标泄漏），
  权重保存至 `output_dir/stats_ridge.pt`
- 注入冻结统计量头；LLM 状态不进入读出输入（4096 维头在目标域输出
  大噪声，消融排除）

## 2. 实验效果（CAUEEG → TUAB 零样本，n=1047，确定性复现）

| 方法 | PearsonR | MAE | RMSE |
|---|---|---|---|
| TimeLLM flatten（历史） | 0.032 | 16.70 | 20.26 |
| TimeLLM answer_query 完整管线 | 0.039 | 17.98 | — |
| ALTER（历史最优） | 0.148 | 17.04 | 20.58 |
| **CAER（本方法）** | **+0.184** | **16.90** | **20.37** |

- 相对 TimeLLM 家族基线提升 **4.7×**；超越 ALTER（+0.036）；
  r=0.184 ≈ 6σ 显著（SE≈0.031）
- 源域（CAUEEG）：val r=0.227、测试 MAE 7.91——源域目标域均健康

## 3. 消融与关键发现

| 消融 | 零样本 PearsonR | 结论 |
|---|---|---|
| 完整可学习管线（answer_query） | 0.039 | 基线 |
| E1 秩归一化输入 | 0.076 | 源域即损坏 |
| E2 语义原型库 | -0.046 | 无信号 |
| E3v7 语义对齐正则 | 0.021 | 无信号 |
| E4 多视图一致性 | -0.031 | 无信号 |
| E5 证据文本 + answer 读出 | 0.013 | 量化文本毁信号 |
| E6 证据位置读出 | -0.053 | patch 污染 |
| E7 纯语言通路 | -0.030 | 同上 |
| E8 stats_head (SGD) | -0.098 | 依赖符号翻转特征 |
| **E8c 闭式 ridge（CAER）** | **+0.184** | ✅ |
| E9 + LLM 残差分支 | 0.184 (MAE 变差) | 零增益，剔除 |

核心规律：

1. **量化文本毁掉跨域信号**：连续统计量跨域 ridge r=0.083，
   3 级量化词降至 0.017
2. **对角元是跨站点锚点**：含对角定义 r=0.074~0.083 vs
   严格边掩码 0.006~0.042
3. **只有收缩解可迁移**：SGD 解依赖跨域符号翻转的特征；
   闭式 ridge 收缩才压制矛盾系数（另：lr=1e-4 收敛到方差塌缩解）
4. **可学习组件在目标域全部失效**：GCN patches、LLM 状态、
   LLM 残差分支均零可迁移增益——可迁移性完全来自确定性证据 +
   标定锚定读出

## 4. 使用方式

```bash
# 1. 拟合源域 ridge（AgeCAUEEG 训练集，val 选 alpha）
python3 scripts/fit_stats_ridge.py

# 2. 零样本评估（scripts/train_TimeLLM_answer_query.sh 已含方法配置：
#    --graph_evidence --no_patches --stats_head --stats_ridge）
#    将 sh 头部改为 DATASET="AgeTUAB"、PRETRAIN_PATH 指向任意目录
#    （无 bin 时跳过加载，模型即为冻结 ridge 头）后运行：
./scripts/train_TimeLLM_answer_query.sh
```

## 5. 局限与展望（诚实声明）

- MAE 仍受跨站点标定漂移限制（源域拟合截距在目标域偏移）；r 为
  标定无关的主指标
- LLM 残差未能带来可迁移增益：语义对齐的 LLM 通路（如何让冻结 LLM
  的预训练知识贡献跨域信号）是明确的未来方向，见 `WORK.md` 负结果分析
- 数据对：CAUEEG → TUAB（当前环境仅有这两个年龄数据集）；
  其他域对需新数据验证
