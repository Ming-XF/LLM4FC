# TimeLLMKnowledge 的 EEG 分类 Baseline 调研

检索截止：2026-10-12。候选首先通过 OpenAlex、Crossref 和 Semantic Scholar 发现，再用 DOI、会议/期刊页面、OpenReview 或作者仓库交叉核验。年份较新不自动等于适合复现；下文把 venue 层级、是否直接做 EEG 分类、代码/权重可得性以及与本项目输入的匹配程度分开说明。

## 最终结论：按三类组织

TimeLLMKnowledge 使用 19 导、200 Hz、60 秒 EEG，并同时接收 raw EEG、global/local FC 和 PSD。基线建议分为以下三类；FC/PSD-only 仅作为输入控制，不冒充三类中的文献模型。

| 类别 | 核心推荐 | 近期补充 | 最终用途 |
|---|---|---|---|
| **传统小模型**（从头训练，不使用 LLM/大规模 EEG 预训练） | EEGNet；Shallow/DeepConvNet；LMDA-Net | EEG Conformer；EEG-Deformer（JBHI 2024）；HiSAM-MambaNet（ICASSP 2026） | EEGNet/ShallowConvNet/LMDA-Net 是经典或轻量基线；其余是较新的从头训练架构，不一定参数小。所有模型需在 CAUEEG/DS 上统一训练，不能直接引用原论文 MI/BCI 数字。 |
| **基于 LLM**（通用语言模型或 time-series reprogramming，不是 EEG foundation pretraining） | One Fits All（NeurIPS 2023，含时间序列分类）；EEG-GPT（NER 2025） | Time-LLM adaptation（ICLR 2024）；Tokenized EEG + LLM（Information Fusion 2026）；Modality reprogramming（Neurocomputing 2026） | 检验“通用 LLM/reprogramming”是否带来增益；必须明确冻结/微调 LLM、可训练参数量和输入 token 化方式。 |
| **Foundation model**（在大量 EEG 上预训练；继续预训练 LLM 的 NeuroLM 也归此类） | LaBraM；EEGPT；CBraMod；NeuroLM；EEGMamba | Gram（TAFFC 2026）；FEMBA（TBME 2026）；Neuro-GPT（ISBI 2024）；BIOT/EEG2Rep | 预训练 EEG 表征对照；分别报告 frozen linear probe 和 full fine-tuning，并审计预训练数据与目标集重合。 |

### 三类完整候选配置（扩展方案）

如果需要覆盖较多模型、且计算资源允许，可按以下扩展配置组织：

1. **传统小模型**：EEGNet + ShallowConvNet + LMDA-Net 为经典/轻量代表；EEG Conformer + EEG-Deformer 为较新、较强的从头训练网络；HiSAM-MambaNet 是 2026 扩展。
2. **基于 LLM**：One Fits All 的 EEG 分类适配 + EEG-GPT；Time-LLM-style classifier 可作为与 TimeLLMKnowledge 架构接近的补充。Information Fusion 2026 论文在拿到全文并确认代码/协议后再加入。
3. **Foundation**：LaBraM + EEGPT + CBraMod + NeuroLM + EEGMamba；资源允许且完成数据审计后优先加入 Gram、再评估 FEMBA。

这样同时覆盖经典基线、近期 raw-EEG 架构、通用 LLM 改造和 EEG 专用预训练模型；提示、reprogramming 和 agent 只作为 LLM 实现细节，不再单列成第四类。

### 六个初步 baseline：建议先实际跑这一组

下面是从完整候选集中收缩出的六个主 baseline。这个集合优先考虑“方法代表性 + venue 层次 + 时间新旧 + 能否与 TimeLLMKnowledge 形成清晰对照”，不是要求每个类别机械地放两个模型。

| # | 类别 | 推荐 baseline | venue / 年份 | 选择理由与执行说明 |
|---:|---|---|---|---|
| 1 | 传统小模型 | **EEGNet** | *Journal of Neural Engineering*, 2018. [DOI](https://doi.org/10.1088/1741-2552/aace8c) | 经典、轻量、官方实现成熟；作为 raw-EEG 分类的下限和可复现锚点。 |
| 2 | 传统小模型 | **EEG-Deformer** | *IEEE JBHI*, 2024. [DOI](https://doi.org/10.1109/JBHI.2024.3504604) | 近期领域强刊的 dense convolution + Transformer 架构；比只保留经典 CNN 更能反映当前 EEG 架构水平。适配 19 导、200 Hz 和 60 秒输入后从头训练。 |
| 3 | 基于通用 LLM | **One Fits All (OFA)** | **NeurIPS 2023 Spotlight**. [DOI](https://doi.org/10.52202/075280-1877) | 综合 AI 顶会、官方代码包含时间序列 classification；将多通道 EEG 作为多变量时间序列适配分类头。原论文不是 EEG，因此结果应标为 adapted baseline。 |
| 4 | 基于 LLM | **Tokenized EEG signals with large language models for epilepsy detection via multimodal information fusion** | **Information Fusion, 2026**. [DOI](https://doi.org/10.1016/j.inffus.2026.104128) | 六个候选中最新且直接做 EEG+LLM 分类/检测的高水平期刊论文。已核实 DOI 和 2026-07 出版信息；但文章非开放获取，当前未确认公开代码，正式实验前必须获得全文/许可材料，核实 token 化、LLM 训练策略、数据划分和实现。 |
| 5 | Foundation | **LaBraM** | **ICLR 2024**. [OpenReview](https://openreview.net/forum?id=QzTpTRVtrP) | EEG foundation 的经典顶会代表，预训练规模和官方下游流程较成熟；作为非语言式 EEG foundation 锚点。 |
| 6 | Foundation | **NeuroLM** | **ICLR 2025**. [OpenReview](https://openreview.net/forum?id=Io9yFt7XH7) | 最新一代 EEG+语言模型 foundation 代表，在 EEG 上继续预训练/离散 token 化；正好检验 TimeLLMKnowledge 与 EEG foundation 的差异。 |

**第 4 项的可复现替代**：如果 Information Fusion 2026 的全文、代码或权重在实验周期内无法获得，用 **EEG-GPT（IEEE EMBS NER 2025，DOI: [10.1109/NER61569.2025.11589079](https://doi.org/10.1109/NER61569.2025.11589079)）**替代。EEG-GPT 直接做 TUH normal/abnormal EEG 分类，但 NER 是专业会议，venue 层级低于 Information Fusion；因此它是复现优先的 fallback，而不是把两篇同时计入六个主 baseline。

**不放入六个主 baseline 的近期候选**：CBraMod、EEGPT、EEGMamba、Gram、FEMBA、Time-LLM adaptation 和 HiSAM-MambaNet 仍保留在完整推荐表中。若后续显存、代码和预训练数据审计允许，foundation 类优先扩展 EEGMamba/CBraMod，传统类扩展 HiSAM-MambaNet；但初步实验不需要同时堆入。

## Venue 如何表述

- **ICLR、NeurIPS、KDD**：综合机器学习/数据挖掘顶会。
- **ICASSP、ISBI、EMBC、NER**：IEEE/EMBS 领域重要会议；在信号处理、生物医学成像或神经工程领域有认可度，但不能与 ICLR/NeurIPS 直接等同。
- **IEEE JBHI、IEEE TNSRE、IEEE TBME、IEEE TAFFC、NeuroImage、Information Fusion**：期刊。它们在生物医学信息学、神经工程、生物医学工程、情感计算或信息融合领域属于强刊，部分学科年度分区为 Q1；应称“领域强刊/高水平期刊”，不要笼统写成所有计算机领域公认的顶刊。

## 1. 传统小模型：raw-EEG 分类

| 模型 | venue / 年份 | 为什么保留 | 对本项目的建议 |
|---|---|---|---|
| **EEGNet** | *Journal of Neural Engineering*, 2018. [DOI](https://doi.org/10.1088/1741-2552/aace8c), [代码](https://github.com/vlawhern/arl-eegmodels) | EEG/BCI 最经典的轻量 CNN，参数少、复现成熟；虽不新，但应作为基本盘。 | 直接输入 raw EEG；统一 19 导顺序、窗口和 subject-level split。 |
| **ShallowConvNet / DeepConvNet** | *Human Brain Mapping*, 2017. [DOI](https://doi.org/10.1002/hbm.23730), [实现](https://github.com/braindecode/braindecode) | EEG 解码经典浅层/深层 CNN，覆盖紧凑浅层与较深卷积两种传统归纳偏置。 | 选 ShallowConvNet 作轻量经典补充即可；与 EEGNet/LMDA-Net 的重复度较高，计算预算紧时不必全跑。 |
| **EEG Conformer** | *IEEE TNSRE*, 2023（online 2022）. [DOI](https://doi.org/10.1109/TNSRE.2022.3230250), [代码](https://github.com/eeyhsong/EEG-Conformer) | EEG 专用 CNN + Transformer，引用和公开实现较好，是比 EEGNet 更强的成熟架构。 | 作为传统组的必选主 baseline；原论文主要是 MI/BCI，疾病分类需重新训练。 |
| **LMDA-Net** | *NeuroImage*, 2023. [DOI](https://doi.org/10.1016/j.neuroimage.2023.120209), [代码](https://github.com/MiaoZhengQing/LMDA-Code) | 轻量多维注意力模型，代表非 Transformer-only 的 EEG 架构。 | 作为可选补充；若篇幅有限，可放附录或消融表。 |
| **EEG-Deformer** | *IEEE JBHI*, 2024. [DOI](https://doi.org/10.1109/JBHI.2024.3504604), [代码](https://github.com/yi-ding-cs/EEG-Deformer) | 近期 dense convolution + Transformer 的 EEG 模型，领域强刊发表，时间和方法新于 EEG Conformer。 | 传统小模型主表优先加入；检查其代码的输入采样率、通道数和窗口后适配 19 导。 |
| **HiSAM-MambaNet** | *IEEE ICASSP*, 2026. [DOI](https://doi.org/10.1109/icassp55912.2026.11463277) | 2026 年 raw-EEG 架构候选，结合 Mamba、Transformer 和层次化时空注意力。 | 作为最新补充；原论文是 BCI Competition IV 运动想象，不是临床疾病分类。取得作者代码/权重后，在 CAUEEG/DS 上从头训练。 |

**传统组最终排序**：`EEGNet + LMDA-Net（狭义轻量） → EEG Conformer（成熟架构） → EEG-Deformer（近期强刊） → HiSAM-MambaNet（2026 新候选）`。后面三者是从头训练的 raw-EEG 网络，但不都属于狭义“小参数量”。

EEG-DG（*IEEE JBHI*, 2024）和 2026 TBME 的 CKD 是 domain generalization/domain adaptation 方法，不是最合适的通用疾病分类器；只有在单独研究跨域机制时才加入。若方法使用目标域样本或在线适配，必须与 `few_shot=0` 的严格 zero-shot 结果分开。

## 2. 基于 LLM 的 EEG 方法

这里的 LLM 组指“使用已有通用语言模型，或将 EEG 映射到通用 LLM 表示空间后进行分类”，不包括在大量 EEG 上继续预训练形成的 foundation model；后者统一放第三组。

| 方法/论文 | venue / 年份 | 实际设定 | 推荐等级与注意事项 |
|---|---|---|---|
| **One Fits All (OFA)** | **NeurIPS 2023 Spotlight**. [DOI](https://doi.org/10.52202/075280-1877), [官方代码](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) | 使用预训练 GPT 等模型处理多类时间序列任务，官方代码明确含 classification；这是通用 LLM 路线中较贴近分类的顶会基线。 | **优先纳入的通用 LLM 适配基线**：将多通道 EEG 视为多变量序列，适配分类头/patch 输入，在相同 subject split 上训练。原论文不是 EEG。 |
| **Time-LLM**（改造成 EEG 分类器） | **ICLR 2024**. [OpenReview](https://openreview.net/forum?id=Unb5CVPtae), [代码](https://github.com/KimMeen/Time-LLM) | 将数值时间序列 reprogram 到冻结 LLM 表征空间；原论文是时间序列预测，不是 EEG 分类。 | 可作为与 TimeLLMKnowledge 同范式的架构控制；不是现成 EEG 分类结果。改造成分类器时标注 adapted baseline。 |
| **EEG-GPT** | IEEE EMBS NER 2025（arXiv 2401.18006；[DOI](https://doi.org/10.1109/ner61569.2025.11589079)） | 在 TUH EEG 上微调公开 GPT/LLM，做 normal/abnormal 分类；摘要报告 2% 训练数据 AUROC 0.86。 | **直接 EEG+LLM 参照**，但 NER 是神经工程专业会议，不是综合 AI 顶会；公开可复现训练代码需再核验。不能把它称为 soft-prompt 方法，除非代码证明冻结 LLM 且训练连续 prefix。 |
| **Tokenized EEG signals with large language models for epilepsy detection via multimodal information fusion** | **Information Fusion, 2026**. [DOI](https://doi.org/10.1016/j.inffus.2026.104128) | 题目明确是 tokenized EEG + LLM 的癫痫检测；属于最新直接相关的高水平期刊工作。 | **直接 EEG+LLM 方法的首要审计对象**：若全文、任务设定和实现可获得，优先于非 EEG 的时间序列方法；当前公开元数据不足以确认其 token 化、LLM 冻结策略、数据划分和代码。 |
| **Modality reprogramming: Adapting frozen LLMs for multi-channel EEG classification** | *Neurocomputing*, 2026（Crossref 记录卷期为 2026；[DOI](https://doi.org/10.1016/j.neucom.2025.132407)） | 标题直接对应冻结 LLM、多通道 EEG 分类和 modality reprogramming。 | **最接近当前方法的外部 LLM 对照**，但不是 ICLR/NeurIPS/顶级综合期刊；应在全文确认 prompt/adapter、输入任务和公开代码后再纳入主表。 |
| **EEG Agent** | **AAAI 2026**. [DOI](https://doi.org/10.1609/aaai.v40i21.38867) | LLM 调度 EEG 工具完成多任务分析、事件检测和报告生成。 | AAAI venue 很强，但它是 agent/tool-use 框架，不是单一端到端疾病分类器；放 related work，不作为主分类 baseline。 |

### LLM 组的最终执行方式

- 主结果优先审计 **Information Fusion 2026** 的直接 EEG+LLM 方法；若论文/实现暂不可获得，则运行 **One Fits All 的 EEG 分类适配 + EEG-GPT（若代码可得）**。另可运行 Time-LLM-style classifier，作为与 TimeLLMKnowledge 更接近的范式控制。
- Neurocomputing 2026 先做全文/代码审计；若可复现，作为另一直接外部 LLM 对照。
- 固定并报告：LLM backbone、是否冻结、可训练参数量、EEG 到 token 的映射、上下文长度、分类头和训练数据比例。
- 记录提示、reprogramming、agent/tool-use 和 LLM 生成标签等实现细节，但不把它们单独列成 baseline 类别。

## 3. EEG Foundation Models

Foundation 组包括在大规模 EEG 上自监督/多任务预训练的模型，也包括以 LLM 为骨干并在 EEG 上继续预训练或对齐的模型。因此 **NeuroLM 放在 foundation 组是正确的**，不再单列为提示调优或普通 LLM。

| 模型 | venue / 年份 | 预训练/模型特点 | 推荐等级与适配注意事项 |
|---|---|---|---|
| **LaBraM** | **ICLR 2024**. [OpenReview](https://openreview.net/forum?id=QzTpTRVtrP), [代码](https://github.com/935963004/LaBraM) | 约 2,500 小时、约 20 个 EEG 数据集的通用脑信号预训练；提供 TUAB/TUEV 下游示例。 | **必做经典 foundation baseline**；核对 19 导映射、200 Hz 预处理和预训练语料是否包含目标数据。 |
| **EEGPT** | **NeurIPS 2024**. [论文记录](https://doi.org/10.52202/079017-1239), [代码](https://github.com/BINE022/EEGPT) | EEG 专用预训练 Transformer，强调跨任务/跨被试泛化。 | **必做顶会 foundation baseline**；官方示例常见 58 导、256 Hz、4 秒输入，需要重采样和通道映射审计。 |
| **CBraMod** | **ICLR 2025**. [OpenReview](https://openreview.net/forum?id=NPNUHgHF2w), [代码](https://github.com/wjq-learning/CBraMod), [权重](https://huggingface.co/weighting666/CBraMod) | Criss-cross brain foundation model，通道 × 时间 patch 建模，提供下游分类流程。 | **必做近期顶会 baseline**；保持通道身份和顺序，说明 19 导适配是重映射、插值还是丢弃通道。 |
| **NeuroLM** | **ICLR 2025**. [OpenReview](https://openreview.net/forum?id=Io9yFt7XH7), [代码](https://github.com/935963004/NeuroLM), [权重](https://huggingface.co/Weibang/NeuroLM) | 在大规模 EEG 上进行离散 token/语言式多任务预训练；使用 LLM 作为语言-EEG 桥梁。 | **必做的 LLM-based foundation baseline**；使用较小 checkpoint 起步，审计 VQ tokenizer、指令格式、下游微调方式和目标数据重合。 |
| **EEGMamba** | *Neural Networks*, 2025. [DOI](https://doi.org/10.1016/j.neunet.2025.107816), [官方代码](https://github.com/wjq-learning/EEGMamba) | 使用 Mamba backbone 和 masked EEG reconstruction，在五个数据集上约 16,724 小时 EEG 预训练；在六项 BCI 任务评估。 | **近期且可复现的 foundation 候选，建议纳入主表**；检查其预训练数据与目标域重叠，及通道/采样率协议。 |
| **Neuro-GPT** | **IEEE ISBI 2024**. [DOI](https://doi.org/10.1109/ISBI56570.2024.10635453), [代码](https://github.com/wenhui0206/NeuroGPT) | EEG encoder + GPT 架构，通过大规模 EEG masked-segment 自监督预训练；论文在少样本 MI 上验证。 | 作为 NeuroLM 之外的早期 LLM-backed EEG foundation 对照；规模和任务覆盖较小，可选，不必与 NeuroLM 两者都跑。 |
| **Gram: A Large General EEG Model for Raw Data Classification and Restoration** | **IEEE TAFFC, 2026**. [DOI](https://doi.org/10.1109/TAFFC.2025.3638592) | 约 7,000 小时 EEG 预训练，联合 temporal/spectral masked objectives，并报告多种 cross-subject 分类。 | **最新高水平候选**；当前未确认稳定官方代码/权重，获得实现后优先尝试。它是 EEG foundation model，不是通用文本 LLM。 |
| **FEMBA on the Edge** | **IEEE TBME, 2026**. [DOI](https://doi.org/10.1109/TBME.2026.3683482) | 约 21,000 小时 EEG 预训练的双向 Mamba foundation model；摘要报告 TUAB 异常分类结果。 | **最新临床候选**；先核对代码/权重、采样率/导联、TUAB 是否进入预训练，再决定是否纳入严格 unseen-target 表。 |
| **BIOT** | **NeurIPS 2023**. [OpenReview](https://openreview.net/forum?id=c2LZyTyddi), [代码](https://github.com/ycq091044/BIOT) | 面向跨数据 biosignal 学习，提供临床 EEG 预训练和 TUAB/TUEV 示例。 | 作为工程上易落地的补充；若 checkpoint 已包含目标数据，不能宣称严格 zero-shot。 |
| **EEG2Rep** | **ACM KDD 2024**. [DOI](https://doi.org/10.1145/3637528.3671600), [代码](https://github.com/Navidfoumani/EEG2Rep) | 在 latent representation 上做 masked self-prediction，跨六类 EEG 任务验证。 | 可作为自监督表征补充；若主表已经有四个 foundation，可放附录，避免与大型通用模型重复。 |

### Foundation 组的最终执行方式

- **核心五个**：LaBraM、EEGPT、CBraMod、NeuroLM、EEGMamba。若算力有限，优先 LaBraM、CBraMod、EEGMamba，再加 NeuroLM；EEGPT 依输入适配成本决定。
- **2026 扩展**：将 Gram 作为近期优先项；若官方实现、权重和预训练数据清晰，再加入 FEMBA。
- 每个模型至少报告两种协议：`frozen encoder + linear probe` 与 `full/fine-tuning`。不要把随机初始化的同结构模型写成 foundation baseline。
- 预训练数据包含 CAUEEG、DS、TUAB、TUEP 或其同源记录时，单独标注“目标数据参与预训练”，不能与严格 zero-shot 结果混报。

## 与 TimeLLMKnowledge 的输入和公平性控制

1. **raw-EEG 统一**：所有传统小模型和 LLM baseline 使用相同 19 导、200 Hz、60 秒切片及 subject-level split；同一被试的窗口不得跨 train/validation/test。
2. **FC/PSD 控制**：增加一个只输入 `global_fc`、`local_fc`、`global_psd`、`local_psd` 的 FC/PSD-MLP，作为输入特征控制。它不是顶会文献模型，不替代三类主 baseline。
3. **不纳入仓库现有连接模型**：BrainNetCNN、BNT、GCDGCN 等是连接矩阵/脑网络建模模型，不作为 raw-EEG 传统小模型；若要研究 DFC 表示，另设“FC/DFC 补充组”，不要与三类主表混列。
4. **任务标签不能混淆**：CAUEEG/DS 是 AD vs control；TUAB 是 normal vs abnormal；TUEP 是 epilepsy/no-epilepsy。不同标签语义不能直接当作同一个跨域分类任务。
5. **跨域协议分开**：source→target、`few_shot=0` zero-shot、few-shot 和 full-data fine-tuning 分开报告。使用目标域无标签统计、伪标签或在线适配的方法也应单列为 adaptation，而不是严格 zero-shot。
6. **指标和随机性**：至少 3 个 few-shot seeds，报告 subject-level 均值和标准差；分类任务报告 AUROC、balanced accuracy、F1，并说明类别不平衡处理。

## 一句话最终推荐

论文主表采用 **EEGNet / ShallowConvNet / LMDA-Net / EEG Conformer / EEG-Deformer**（从头训练，覆盖经典轻量与近期架构），**Information Fusion 2026（若可复现）/ One Fits All + EEG-GPT**（基于通用 LLM），以及 **LaBraM / EEGPT / CBraMod / NeuroLM / EEGMamba**（EEG foundation）；将 **Time-LLM adaptation、HiSAM-MambaNet、Gram、FEMBA** 作为对齐研究重点或完成代码/数据审计后的扩展项。FC/PSD-MLP只作输入控制，不把 BrainNetCNN、BNT 等连接模型混入传统 EEG 分类组。

## 可复跑检索

`search_eeg_baselines.py` 按 traditional、LLM-based、foundation 三组查询 OpenAlex，并导出候选 JSON/CSV。OpenAlex 结果只用于候选发现；正式引用以 DOI、会议/期刊页面和作者官方仓库为准。API key 仍由现有代码/环境配置管理，不在报告中展示密钥值。
