# TimeLLMKnowledge v2 数据重训与零样本评估结果

日期：2026-10-09（seed 42，8B DeepSeek-R1-Distill-Llama 冻结，配置与 `scripts/train_TimeLLMKnowledge.sh` / `eval_TimeLLMKnowledge_zs.sh` 一致）

## 背景：预处理修复（v2 数据）

此前分析文档 `TimeLLMKnowledge_Age零样本泛化分析与建议.md` 确认了两个关键问题：

1. **年龄标签坐标不统一**：CAUEEG 用 23–96、TUAB 用 0–89 各自 min-max 归一化，训练/评估代码却假定统一 `age/100` 坐标 → 零样本指标被坐标错位偶然校准，不可信。
2. **EEG 预处理不一致**：TUAB 有固定 19 导联顺序 + average reference + 200Hz 重采样，CAUEEG 直接取前 19 通道无参考处理 → FC 分布系统性偏移（TUAB 非对角 FC 绝对均值比 CAUEEG 高约 41%）。

修复内容（`data/preprocess.py` 等）：

- 年龄统一 `labels_raw / 100`，强制 `label_min=0, label_max=100`；无 `labels_raw` 的旧文件按记录的边界反解并校验一致性。
- 统一 19 导联 10-20 顺序（`EEG_19_CHANNEL_ORDER`，含 T7→T3 等别名归一）、average reference、200 Hz 重采样（`prepare_common_19_channel_raw`）。
- CAUEEG/TUAB 多进程预处理改为确定性顺序（`pool.imap`），避免重复预处理选中不同窗口。
- NPZ 原子写入（`save_npz_atomic`）+ v2 元数据：`preprocessing_version`（年龄 `age_eeg_v2_common19_avgref_200hz_fixedage`，疾病 `eeg_v2_common19_avgref_200hz`）、`channel_names`、`reference`、`unit`、标签边界。
- knowledge 特征 sidecar 保存 `source_*` 元数据（`data/knowledge_preprocess.py`）；`data/dataset.py` 加载时严格校验特征与源 NPZ 元数据一致。

新数据（`/root/autodl-tmp/data/`）：CAUEEG/TUAB 的 age 与 disease 四组 NPZ + knowledge 特征，全部通过元数据校验。

---

## 年龄任务（AgeCAUEEG 同域 + CAUEEG→TUAB 零样本）

### 同域 AgeCAUEEG

| 指标 | 旧数据 | 新 v2 数据 |
|---|---:|---:|
| test R² | 0.218 | **0.204** |
| test MAE (y) | 7.09 | **7.14** |
| test Pearson r | — | **0.455** |

早停于 epoch 56，best val R²=0.206。**同域性能持平**（波动范围内），标签修复不影响同域（同域坐标本就一致）。

### 零样本 CAUEEG→TUAB

| 指标 | 旧数据（坐标错） | 新 v2 数据 | 常数 50 岁基线 |
|---|---:|---:|---:|
| MAE (y) | 17.59 | **23.47** | 15.09 |
| R² | -0.408 | **-1.360** | -0.022 |
| Pearson r | 0.144 | **0.210** | 0 |

新 TUAB test 划分分布不变（n=1047，mean 47.3y，std 18.0y）。

**解读：**

1. MAE/R² 变差是"诚实化"而非退步：旧数字是坐标错位偶然校准的假象（CAUEEG 的 23–96 坐标与 TUAB 的 0–89 坐标部分抵消了域偏移）。统一坐标后，模型在老人源域训练（均值 ~71y），对年轻目标域（均值 ~47y）系统性高估，真实误差暴露。
2. Pearson r 0.144→0.210 是本次修复的真实收益（Pearson 对坐标变换不变，提升只能来自 EEG 预处理统一的排序信号增强）。信号仍弱。
3. **年龄零样本仍判失败**（MAE 差于常数基线、R² 为负），与分析文档预判一致："标签 BUG 不是全部原因"。

---

## 疾病任务（DiseaseCAUEEG 同域 + CAUEEG→TUAB 零样本）

疾病标签本就是 0/1，不受年龄坐标问题影响；EEG 预处理统一（导联/参考/重采样）同样生效。

数据：DiseaseCAUEEG n=5068（AD vs Normal，正类率 0.49）；DiseaseTUAB n=5602（abnormal vs normal，正类率 0.515）。

### 结果

| 指标 | 同域 DiseaseCAUEEG | 零样本 CAUEEG→TUAB |
|---|---:|---:|
| AUC | **0.763** | **0.678** |
| Accuracy | 0.713 | 0.640 |
| Sensitivity | 0.584 | 0.681 |
| Specificity | 0.813 | 0.591 |
| Precision | 0.707 | 0.658 |

同域早停于 epoch 52，best val AUC=0.780。

**解读：**

1. **疾病零样本跨域存在真实信号**（AUC 0.678，显著高于机会水平 0.5），与年龄任务零样本失败形成鲜明对比。合理原因：异常/正常 EEG 模式比精细年龄回归特征更具跨域不变性；分类不受标签坐标/尺度影响。
2. 这是 TimeLLMKnowledge 在疾病任务上的首次完整结果（旧数据无同配置基线可比）。

---

## 期间修复的框架 bug

- **`utils/trainer.py:524`（已修，经用户批准）**：binary_evaluate 非分布式分支 `preds_local.numpy()` 缺 `.cpu()`，单卡零样本评估直接崩溃。deepspeed 训练时走分布式分支（503 行有 `.cpu().numpy()`）所以此前未暴露。已改为 `.cpu().numpy()`，与分布式分支行为一致。
- **`utils/trainer.py:658`（未修，潜在同类 bug）**：multiple_evaluate 的非分布式分支同样缺 `.cpu()`。多分类单卡评估时会触发，需要时一并修。

---

## 下次会话注意事项

- **脚本当前是疾病配置**：`scripts/train_TimeLLMKnowledge.sh`（DATASET=DiseaseCAUEEG, EARLY_STOP_METRIC=AUC）、`scripts/eval_TimeLLMKnowledge_zs.sh`（TARGET_DATASET=DiseaseTUAB）。重跑年龄实验前需改回 AgeCAUEEG / AgeTUAB / R2。
- 新 checkpoint：`output_dir/TimeLLMKnowledge_AgeCAUEEG_train/`（v2 年龄）、`output_dir/TimeLLMKnowledge_DiseaseCAUEEG_train/`（v2 疾病）。
- 日志：`log_dir/train_TimeLLMKnowledge_{Age,Disease}CAUEEG.log`、`log_dir/test_TimeLLMKnowledge_zero-shot_{Age,Disease}TUAB.log`。
- 运行环境注意：`OMP_NUM_THREADS=0` 会导致 libgomp 警告，启动前用 `env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` 覆盖。
- **下一步待用户确认**：TimeLLMKnowledge 多任务化方案（见 `TimeLLMKnowledge多任务方案.md`，方案 A 数据 + 双 READOUT + 双任务头 + Huber+0.01·CE）。
- 年龄零样本若继续攻坚，参考分析文档建议：多源训练（CAUEEG+TUEP → TUAB 留出）、被试级指标、多 seed、恢复显式统计残差分支、Huber delta 调至 0.05–0.1、域平衡 batch 等。
