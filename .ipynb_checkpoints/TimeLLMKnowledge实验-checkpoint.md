# TimeLLMKnowledge 实验运行说明

领域知识 Token 编译器（纯知识版，无 patch 路径）。**当前机器没有 GPU，本文件记录运行方式，待显卡恢复后执行。**

## 一、模型一句话

整分钟 EEG → 静态 FC + 频谱 9 + 形态 14 → 32 个 display stats → 每统计量独立编译
（短语 mean 向量 + 数值线性嵌入在 128 维求和）→ 知识 soft prompt → 冻结 llama 8B
→ 最后 K 层 READOUT 位置聚合读出 → 线性头。

序列布局：`[start_tag][task prompt][end_tag][knowledge 1..32][READOUT]`。

## 二、运行前提

1. GPU 恢复（2×96GB；deepspeed JSON 是 **batch size / LR 的唯一来源**，脚本不覆盖）。
2. 数据在 `/root/autodl-tmp/data/`（已预处理的 .npy 无需重新预处理——spectral/morph
   统计量在 `Dataset.__getitem__` 里由 `spectral_stats`/`morph_stats` 运行时现算，
   **首轮数据加载会比以往稍慢**，属正常现象）。
3. LLM 权重在 `model/deepseek-r1-distill-llama-8B/`。

## 三、训练

```bash
sh scripts/train_TimeLLMKnowledge.sh
```

改脚本顶部变量即可换任务：

| 变量 | 取值 |
|---|---|
| `DATASET` | `AgeCAUEEG`（默认）/ `AgeTUAB` / `AgeTUEP` / `DiseaseCAUEEG` / `DiseaseTUAB` / `DiseaseTUEP` 等 |
| `EARLY_STOP_METRIC` | 回归用 `R2`（默认）；分类用 `AUC` 或 `Accuracy` |

关键参数（脚本内已设好，如无必要不改）：

- `--readout_layers 4`：多层读出的层数 K（第 29~32 层）；可试 8。
- `--num_prototypes 1000`、`--d_model 128`、`--num_heads 8`。
- `--llm_type llama` + `--llm_init pretrained`：8B 冻结；消融可换 `--llm_init random`。
- `--use_task_prompt`：任务 prompt 默认开启。
- batch/LR 见 `deepspeed/train_patch.json`（当前 batch=32、lr=1e-4、ZeRO-2、fp32）。

输出位置：

- 训练日志：`log_dir/train_TimeLLMKnowledge_<DATASET>.log`
- checkpoint：`output_dir/TimeLLMKnowledge_<DATASET>_train{,_best}/`（含 `config.json`）
- 测试结果：`output_dir/results.json`

## 四、训练中监控

模型 `forward` 的 `ModelOutputs.hidden_state` 返回：

- `q` (B,32,128)：知识 query 向量；
- `knowledge` (B,32,4096)：reprogram 后 soft prompt；
- `readout_agg` (B,4096)：K 层聚合后的读出向量；
- `layer_weights` (K,)：各层聚合权重 α（应保持最终层权重 ≥ β=0.4 的锚定下限）。

重点观察：

1. 损失是否正常下降（首 1~2 epoch 不降属于正常，数值/短语分支在预热）；
2. `layer_weights` 不塌缩到某一层（锚定 β=0.4 防止全塌到最终层）；
3. 若训练完全不收敛：检查 `P`/`value_weights`/`mapping_layer` 的梯度范数是否
   持续接近 0——若是，说明冻结 LLM 长梯度路径问题真实存在，按设计加
   gated pre-LLM bypass（`u=MeanPool(knowledge) → RMSNorm → Linear(4096,out)`，
   `y = y_llm + g·y_bypass`，g 初始 0.05~0.2，当前未实现，需要时再加）。

## 五、评价与迁移

- 训练脚本自带 `--do_evaluate --do_test`（60/20/20 划分，seed 42）。
- 零样本/few-shot 迁移脚本可仿照 `scripts/eval_TimeLLMPatch_zs.sh`：
  `--pretrain_path ./output_dir/TimeLLMKnowledge_AgeCAUEEG_train` +
  `--few_shot 0`（zero-shot）或 `>0`（few-shot）+ `--do_evaluate`。
- 迁移时目标数据集必须同样返回 spectral/morph（现有 6 个数据集均已支持），
  统计量 schema 统一为 32 维（分类任务也含形态族）。

## 六、实现文件索引

- `model/TimeLLMKnowledge/TimeLLMKnowledge.py`：Model / Config / ReprogrammingLayer / RMSNorm
- `model/TimeLLMKnowledge/knowledge_compiler.py`：32 短语清单 + phrase bank + P + W_i
- `model/TimeLLMKnowledge/prompts.py`：任务 prompt（与 TimeLLMPatch 相同）
- `model/TimeLLM/graph_evidence.py`：`compute_graph_evidence`（已支持 W=1 静态 FC）
- `data/dataset.py`：`spectral_stats`/`morph_stats`（整分钟 15×4s 版）
- `trainers.py`：`TimeLLMKnowledgeTrainer`；`config.py`：模型分支 + `--readout_layers`
