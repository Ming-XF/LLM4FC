结论先说：这次结果“略好”不能归因于把计算移到预处理阶段；当前零样本结果确实基本无效，而且我确认了一个会直接破坏年龄跨域迁移的标签归一化问题。修复后，MAE/R² 很可能明显变化，但 Pearson r=0.1439 表明模型仍只有很弱的跨域排序信号，不能仅靠校准解决。

### 1. 这次为什么略有提升

三次 AgeCAUEEG 结果是：

| 版本 | MAE | R² | 最佳验证 R² |
|---|---:|---:|---:|
| 原 v2stats | 7.2106 | 0.1929 | 0.1892 |
| 在线 PSD 版本 | 7.0984 | 0.2161 | 0.2212 |
| 离线预处理版本 | 7.0906 | 0.2181 | 0.2147 |

真正从“在线 PSD”改成“离线 PSD”后，只改善了 0.0079 岁 MAE 和 0.0021 R²，验证集反而略低。这远小于单次训练、早停点和浮点实现带来的波动，不能说明离线预处理提升了模型能力。

而且这次提交并不是等价搬运：

- 旧的显式数值统计分支被删除，只保留 raw/patch token。
- 局部 FC 从“整分钟标准化后切片”变成了“每个 10 秒片段重新标准化后计算相关性”，后者更接近真正的局部 Pearson FC。
- 输入接口、语义原型、patch embedding、bypass、bias 和训练参数也发生过变化。
- 旧 checkpoint 有 24 个保存键，新版本有 36 个，属于不同模型版本，不能把零样本改善只归因于预计算。

因此，合理解释是：局部 FC 计算更规范、路径简化以及训练波动共同带来了很小的改善；离线化本身主要提高速度和确定性。

### 2. 已确认的关键 BUG：跨域标签坐标不一致

当前源码要求年龄统一用 `[0,100]`，例如 [data/preprocess.py](/root/autodl-tmp/LLM4FC/data/preprocess.py:5) 和 [age_caueeg.py](/root/autodl-tmp/LLM4FC/data/caueeg/age_caueeg.py:247)。

但磁盘中的实际 NPZ 是：

- CAUEEG：`label_min=23, label_max=96`
- TUAB：`label_min=0, label_max=89`
- TUEP：`label_min=0, label_max=100`

并且我验证了 `labels_raw`：CAUEEG、TUAB 都确实使用各自数据集 min-max 归一化，不是元数据写错。

因此模型训练输出 \(p\) 的含义是：

```text
CAUEEG: age = 23 + 73p
TUAB:   age =  0 + 89p
```

当前零样本评估却直接把 CAUEEG checkpoint 的输出 `p` 与 TUAB 的标签 `p` 比较。比如 `p=0.5` 在源域表示 59.5 岁，在目标域标签中表示 44.5 岁。这使当前 MAE/R² 不是合法的统一尺度迁移指标。

这也解释了为什么同域结果可以正常，而跨域结果严重失准。评估代码根据目标数据的 span 换算 MAE，见 [trainer.py](/root/autodl-tmp/LLM4FC/utils/trainer.py:808)，却没有将源 checkpoint 的输出坐标转换成目标坐标。

需要注意：正仿射换算不会改变 Pearson 相关，所以 `PearsonR=0.14385` 仍然有效地说明跨域信号很弱。标签 BUG 不是全部原因。

### 3. 当前零样本到底有多差

最新 CAUEEG → TUAB：

- MAE：17.5905 岁
- R²：-0.4083
- Pearson r：0.1439

TUAB 测试集直接预测 50 岁的简单常数基线是：

- MAE：约 15.09 岁
- R²：约 -0.022

也就是说，当前模型不仅 R² 为负，MAE 还差于不看 EEG、恒定预测 50 岁的基线。现阶段应明确报告为“零样本泛化失败”。

此外，源域和目标域年龄分布明显不同：

- CAUEEG 训练集：均值约 70.9 岁，标准差约 9.7 岁。
- TUAB 测试集：均值约 47.3 岁，标准差约 18.0 岁。

模型是在较窄、明显偏老的源域上训练，然后预测更年轻、更宽的目标人群，这会造成严重的标签分布偏移。

### 4. 另一个高优先级问题：两域 EEG 预处理不一致

TUAB 明确执行：

- 固定 19 通道顺序；
- average reference；
- 重采样至 200 Hz。

见 [age_tuab.py](/root/autodl-tmp/LLM4FC/data/tuab/age_tuab.py:199)。

CAUEEG 当前只是直接取 EDF 前 19 通道，没有：

- 根据通道名称重排；
- average reference；
- 显式 resample。

见 [age_caueeg.py](/root/autodl-tmp/LLM4FC/data/caueeg/age_caueeg.py:139)。

这对 FC 特别致命。实际离线特征中，TUAB 的 FC 非对角绝对均值比 CAUEEG 高约 41%，标准差高约 43%。逐通道 z-score 消除不了参考方式导致的相关结构差异。

### 5. 代码中暂未发现的 BUG

零样本控制流本身是正确的：加载源 checkpoint 后直接测试，没有目标域训练，见 [main.py](/root/autodl-tmp/LLM4FC/main.py:172)。

日志中的：

```text
583 keys not in checkpoint
35/36 keys loaded
```

也不表示 LLM 没加载。checkpoint 设计上只保存可训练参数和 buffer，冻结 LLM 从本地 pretrained 权重重新初始化；唯一跳过的键是目标任务 prompt。当前 AgeCAUEEG 和 AgeTUAB 的任务 prompt 文本相同，所以这不是失败原因。不过加载逻辑应增加“所有可训练键必须匹配”的严格校验。

### 6. 建议执行顺序

第一优先级：先修正确性，再谈调参。

1. 在年龄 loader 中优先读取 `labels_raw`，统一计算 `labels=labels_raw/100`，并强制 `label_min=0, label_max=100`。
2. 用旧 checkpoint 做一次取证评估时，应先按源域公式把预测还原成年龄；正式实验必须在统一标签后重新训练源模型。
3. 给数据加载增加断言：任何 Age 数据的 `label_min/max` 不是 `0/100` 就直接报错。
4. 统一 CAUEEG、TUAB、TUEP 的通道顺序、参考、采样率、滤波和单位。
5. 为在线/离线 FC、PSD 添加逐样本一致性测试。特别明确局部 FC 是否采用片段内 z-score。
6. checkpoint 中保存并校验 Git commit、特征版本、标签归一化、通道表、参考方式和模型开关。

第二优先级：建立可信评估。

- 同时报告窗口级和被试级指标；被试级先平均一个人的多个窗口预测。
- 报告 MAE、RMSE、R²、Pearson、Spearman、CCC、预测均值/标准差以及校准斜率/截距。
- 必须包含常数 50 岁、源域均值、线性 Ridge/MLP 等基线。
- 至少跑 3–5 个训练种子；当前每个版本只有一次运行。
- 保存逐样本预测，方便区分“有排序信号但校准错”与“完全无信号”。
- 零样本参数选择不能使用 TUAB 验证标签；应在源域验证集或 CAUEEG/TUEP 的 leave-one-domain-out 设置中完成。

第三优先级：不大改架构时最值得尝试的改进。

- 恢复显式、可解释的 FC/频带/形态统计残差分支，但先做稳定归一化；当前 `FCEmbed`、`SpecEmbed` 经过全局平均池化后再生成 9/13 个 token，这些 token 实际并不分别对应“额区连接”“alpha 功率”等语义，见 [patch_embeds.py](/root/autodl-tmp/LLM4FC/model/TimeLLMKnowledge/patch_embeds.py:34)。
- 给 global 与 local 分支设置独立可学习 gate。当前 192 个局部 token 对 32 个全局 token，占比过大，还缺少显式 segment identity。
- 对六个局部段先做 mean/std 或轻量 attention 聚合，降低片段噪声和源域过拟合。
- 在 FC/PSD encoder 前加入固定的 feature normalization 或 LayerNorm，尤其要处理 TUAB 与 CAUEEG 的 FC 尺度偏移。
- 将 Huber `delta` 从归一化空间的 `1.0` 改为 `0.05–0.1`；现在的 `delta=1.0` 相当于约 100 岁范围，实际几乎退化成 MSE。
- 小网格测试学习率 `1e-5 / 3e-5 / 1e-4`、gradient clipping `1.0`、weight decay `1e-3`，并尝试 EMA/SWA 或最佳 3–5 个 checkpoint ensemble。
- 使用 subject-balanced sampler，避免多个窗口使某些被试权重更大。
- 做源域增强：reference augmentation、channel dropout、FC edge dropout、频带 mask、轻度噪声和时间裁剪。
- 最有希望的架构不变方案是多源训练：用 CAUEEG+TUEP 训练、TUAB 严格留出测试，并采用域平衡 batch。单一偏老源域很难学到真正稳定的年龄映射。

不建议第一步就加 LoRA 或解冻 LLM。当前决定迁移效果的主要是 EEG 特征、reprogramming 和输出头；语言预训练并不会自动提供跨设备、跨参考的 EEG 不变性。

我没有修改代码，也没有启动训练，等待你的下一步指令。
