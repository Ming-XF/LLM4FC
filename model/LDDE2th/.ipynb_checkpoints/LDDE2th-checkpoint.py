import torch
import torch.nn.functional as F
from torch import nn
from ..base import BaseConfig, ModelOutputs
from .LDDE2thLayers import *


from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, get_peft_model

import os
import json
import logging

_log = logging.getLogger(__name__)


# ── L_text 类别名称与目标模板 ────────────────────────────────────
# 训练时从标签构造短诊断结论文本，作为因果 LM 目标，
# 教会 lm_head 从 [系统 | 原型 | 任务 | 数据] 的 hidden state 解码出中文诊断文本。


def build_system_prompt():
    """构建 LLM system prompt，说明输入结构和任务目标。"""
    return (
        "你是一位神经影像专家。接下来你会依次收到两组 Soft Prompt，"
        "它们由脑功能网络的专业知识文本嵌入经患者特异性激活加权生成："
        "(1) 跨时间窗的整体脑功能网络激活画像，"
        "(2) 各个时间窗口的动态脑网络激活快照。"
        "请基于这些信息，诊断该患者属于以下四类之一："
        "阿尔茨海默病、轻度认知障碍、主观认知下降或正常认知。"
    )



# ═══════════════════════════════════════════════════════════════════════════════
# Pretrained BNC Loading (P1.3) — warm-start BNC from DFCBNC checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

def load_pretrained_bnc(model, task_id, base_dir="output_dir/DFCBNC", verbose=True):
    """从 DFCBNC checkpoint 按折号加载 BNC 权重到 LDDE2th。

    对应 tech.md §6.1 + todo.md §0.3 折号对应：
      - 3 折交叉验证，第 k 折加载 DFCBNC-k.bin
      - DFCBNC 的 BNC 与 LDDE2th 的 BNC 是同一 BrainNetCNN 实例，参数名天然对齐
      - 仅加载 bnc 子模块权重，不加载 DFCBNC 的分类头（cls.*）

    文件缺失则直接报错退出 — 预训练 BNC 是训练的必要前提。

    Args:
        model:    LDDE2th 实例（需已有 self.bnc）
        task_id:  int — 折号 0/1/2，加载 DFCBNC-{task_id}.bin
        base_dir: str — DFCBNC checkpoint 目录
        verbose:  bool — 是否打印加载日志

    Raises:
        FileNotFoundError: checkpoint 文件不存在
    """
    import os
    path = os.path.join(base_dir, f"DFCBNC-{task_id}.bin")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DFCBNC checkpoint not found: {path}\n"
            f"  Train DFCBNC first: bash scripts/dementia_mms/train_DFCBNC_Dementia.sh"
        )

    dfcbnc = torch.load(path, map_location='cpu', weights_only=False)
    model.bnc.load_state_dict(dfcbnc.bnc.state_dict())

    n_params = len(dfcbnc.bnc.state_dict())
    if verbose:
        print(f"[load_pretrained_bnc] Loaded {n_params} BNC params from {path}")

    return {'bnc_params_loaded': n_params}


class LDDE2thConfig(BaseConfig):
    def __init__(self, node_size, num_classes, dfc_stride=5):
        super(LDDE2thConfig, self).__init__(node_size=node_size,
                                            num_classes=num_classes)
        self.dfc_stride = dfc_stride

        # ── 模块 A: 语义原型 ──
        self.num_prototypes = 16
        self.proto_dim = 128
        self.proto_temperature = 1.0
        self.use_soft_prompt = True           # 16 个连续 soft token 替代离散 proto text

        # ── LoRA ──
        self.lora_r = 4
        self.lora_alpha = 8                    # scaling = α/r = 2.0

        # ── 内部文本生成（tech3.md §3）──
        self.rich_text_top_k_ab = 3            # 异常模式最多纳入的原型数
        self.rich_text_top_k_pr = 3            # 保留/代偿模式最多纳入的原型数

        # ── 损失权重 ──
        self.lambda_div = 1.0
        self.lambda_entropy = 1.0
        self.lambda_cla_llm = 0             # LLM 直接分类监督（与 L_cla_bnc 同级，驱动 M 样本特异性）
        self.lambda_text = 1               # 因果 LM 文本生成损失（教 lm_head 解码诊断文本）

        # ── COT target ──
        self.cot_target_mode = "cot"             # "cot" = COT（当前）, "template" = 固定模版

        # ── 自回归生成 ──
        self.max_new_tokens = 200               # LLM 推理输出最大 token 数

        # ── 日志 ──
        self.self_train_log_steps = 200          # debug 日志写入间隔（由 trainer 用 --save_steps 覆盖）


class LDDE2th(nn.Module):
    """LDDE2th — 单 BNC + LLM 知识反馈闭环。

    Architecture (tech.md §2.1)
    ────────────────────────────
    1. BNC 编码 DFC 逐时间窗 → F_per_window (B*L, 256)
    2. 语义原型预测头 → text_summary（中文脑网络描述）
    3. LLM encode: [SystemPrompt | proto_text | TaskPrompt | DataTokens] → HL
    4. 知识掩膜：llm_pooled → M (B, C)，per-channel 调制 H_ch → H_ch_masked
    5. BNC 诊断头：H_cls = mean_pool(H_ch_fused, dim=C) → cls_bnc → logits（唯一诊断输出）
    + LLM 直接分类头：llm_pooled → cls_llm → L_cla_llm（辅助监督）
    """

    def __init__(self, config: LDDE2thConfig):
        super().__init__()
        self.config = config
        C = config.node_size
        self.dfc_stride = config.dfc_stride

        # ── 1. BNC — DFC spatial encoder (Module 2, tech.md §3.3) ──
        self.bnc = BrainNetCNN(C)

        # ── 2. Tokenizer + LLM (tech.md §3.2) ──
        tokenizer = AutoTokenizer.from_pretrained(
            "./model/chatglm-6b", trust_remote_code=True
        )
        base_llm = AutoModel.from_pretrained(
            "./model/chatglm-6b", trust_remote_code=True
        ).bfloat16()
        # gradient_checkpointing OFF — saves ~30% forward compute at the cost of
        # extra GPU memory for storing intermediate activations.
        base_llm.transformer.gradient_checkpointing = False

        self.tokenizer = tokenizer
        wte = base_llm.transformer.word_embeddings

        # 3a. System Prompt → frozen text embeddings (tech.md §3.2 表第一行)
        system_text = build_system_prompt()
        system_ids = tokenizer.encode(system_text)
        with torch.no_grad():
            system_embeds = wte(torch.tensor(system_ids))
        self.register_buffer("system_prompt_embeds", system_embeds)
        self._system_prompt_text = system_text

        # 3b. LoRA r=4 (tech.md §3.2)
        # Device placement is handled by the Trainer (DeepSpeed or .to(device))
        self.llm = base_llm
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["query_key_value", "dense",
                            "dense_h_to_4h", "dense_4h_to_h"],
            lora_dropout=0.05,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, peft_config)

        # ── 4. Cross-Attention Knowledge Mask Head ──
        self.mask_head = CrossAttentionMaskHead(
            llm_dim=4096,
            bnc_dim=128,
            num_channels=C,
        )

        # ── 5. BNC 诊断预测头 (Module 5, tech.md §3.6) ──
        #     per-channel 聚合后特征维度为 128 (E2N 输出通道数)
        self.cls_bnc = nn.Linear(128, config.num_classes)

        # ── 6. LLM 直接分类头 (tech.md §5.1) ──
        #     从 llm_pooled 直接预测疾病类别，为 LLM 提供直接的梯度信号，
        #     防止通过 mask_head 间接回传的梯度过弱导致 LLM 训练坍缩。
        self.cls_llm = nn.Linear(4096, config.num_classes)

        # ── 7. Semantic Prototype Layer (Module 4, tech.md §3.5) ──
        self.semantic_proto = SemanticPrototypeLayer(
            feature_dim=256,
            proto_dim=config.proto_dim,
            num_prototypes=config.num_prototypes,
            llm_dim=4096,
            temperature=config.proto_temperature,
            tokenizer=tokenizer,
            wte=wte,  # LLM 词嵌入表，用于原型文本初始化（tech3.md §1）
        )


        # ── 9. EOS token ID（tech.md §7.1）──
        # ChatGLM-6B 使用 icetk（非原生 SentencePiece），eos_token 为 <eop> (130005)。
        # 之前误用 convert_tokens_to_ids('</s>') 返回 2（SP 内部 ID，不实际出现），
        # 导致 autoregressive 生成永不检测到 EOS，始终跑到 max_length 才停。
        self.eos_token_id = tokenizer.eos_token_id               # 130005 = <eop>
        self._eop_token_id = tokenizer.eos_token_id             # alias for target text

        self._top_k_ab = config.rich_text_top_k_ab
        self._top_k_pr = config.rich_text_top_k_pr

        # ── COT target 文本片段（字符串，运行时拼接后一次性 tokenize）──
        self._cot_prefix = "脑网络关键发现："
        self._cot_interpretation_prefix = "病理生理解读："

        # Proto labels — full phrases with direction suffixes (strings only)
        self._proto_items = {}   # idx → {'abnormal': str, 'preserved': str}
        for p in PROTOTYPE_LABELS:
            zh = p['zh']
            has_integrity = "完整性" in zh
            self._proto_items[p['idx']] = {
                'abnormal':  zh + ("减弱" if has_integrity else "异常降低"),
                'preserved': zh + ("保留" if has_integrity else "代偿性增强"),
            }


        # ── 加载 DeepSeek 生成的推理文本 (per-sample，训练时随机混入 COT target) ──
        ds_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                               'Dementia4000', 'reasoning_texts.json')
        self._ds_sentences = {}   # {int(sample_idx): [str, ...]}  raw strings for runtime tokenization
        try:
            with open(ds_path, 'r') as f:
                ds_texts = json.load(f)
            for idx_str, text_list in ds_texts.items():
                sentences = []
                for text in text_list:
                    for sent in text.split('。'):
                        sent = sent.strip()
                        if not sent or '诊断结论' in sent:
                            continue
                        sentences.append(sent + '。')
                if sentences:
                    self._ds_sentences[int(idx_str)] = sentences
            _log.info(f"Loaded DeepSeek reasoning texts for {len(self._ds_sentences)} samples")
        except FileNotFoundError:
            _log.warning(f"DeepSeek reasoning texts not found at {ds_path}")

        # ── 通用训练状态 ──
        self._debug_step = 0                      # 全局 debug step counter
        self._st_last_log_epoch = -1              # epoch 切换检测（通用日志用）
        self._st_m_epoch_sum = 0.0                # per-epoch M 均值累加
        self._st_m_epoch_count = 0                # per-epoch M 样本数

        # ── 窗口级日志累加器（每 log_steps 输出一次）──
        self._st_win_reset()
        self._st_cur_text_type = None          # 当前 step 文本类型
        self._st_cur_text_preview = ""
        self._st_cur_text_len = 0

    # ── Forward (tech.md §4.1) ────────────────────────────────────
    def forward(self, time_series, DFC, labels,
                gender=None, age=None, education=None,
                sample_indices=None, epoch=0, inference_mode=None):
        """
        Args:
            time_series: (B, C, Ts)   EEG time series（当前不使用）
            DFC:         (B, L_full, C, C)   dynamic FC matrices
            gender, age, education: (B, 1)   demographics（可选，当前不使用）
            labels:      (B,)          ground-truth class indices
            epoch:       int           当前 epoch（用于一致性权重退火）
            inference_mode: None | "fast" | "full"
                None  — 训练路径（默认）
                "fast" — 快速推理：LLM encode → M → 分类，不生成文本
                "full" — 完整推理：LLM encode → M → 分类 + 诊断文本生成
        """
        B, L_full, C, _ = DFC.shape

        # ── Step 0: 时序降采样 (tech.md §4.1) ──
        DFC = DFC[:, ::self.dfc_stride, :, :]           # (B, L, C, C)
        L = DFC.shape[1]

        # ════════════════════════════════════════════════════════════
        # Step 1: BNC 编码 → HB + per-channel 特征 (tech.md §4.1 Step 1)
        # ════════════════════════════════════════════════════════════
        DFC_flat = DFC.reshape(B * L, 1, C, C)           # (B*L, 1, C, C)
        # 将输入转为与 BNC 参数一致的 dtype（DeepSpeed fp16 模式需要）
        DFC_flat = DFC_flat.to(dtype=next(self.bnc.parameters()).dtype)
        F_per_window, F_ch = self.bnc(DFC_flat, return_channel_features=True)
        # F_per_window: (B*L, 256) — N2G 压缩后的全局特征
        # F_ch:          (B*L, 128, C) — E2N 层 per-channel 特征（供掩膜调制）
        HB = F_per_window.reshape(B, L, 256).mean(dim=1)       # (B, 256)  保留兼容
        H_ch = F_ch.reshape(B, L, 128, C).mean(dim=1)           # (B, 128, C) per-channel time-mean pool

        # ════════════════════════════════════════════════════════════
        # Step 2: 语义原型 → 文本描述 + L_div (tech.md §3.5 + §4.1 Step 2)
        # ════════════════════════════════════════════════════════════
        proto_out = self.semantic_proto(F_per_window, B, L)
        activations_mean = proto_out['activations_mean']   # (B, 16)
        activations_2d = proto_out['activations_2d']        # (B, L, 16)
        L_div = prototype_diversity_loss(self.semantic_proto.prototypes)

        # ════════════════════════════════════════════════════════════
        # Step 3: LLM 输入组装 + 因果编码 → HL (tech.md §4.1 Step 3)
        # ════════════════════════════════════════════════════════════

        # 3a. 组装 LLM 输入嵌入（4 组件 prefix，不含 target）
        # System Prompt embedding
        sys_embeds = self.system_prompt_embeds.unsqueeze(0) \
                        .expand(B, -1, -1)               # (B, S, 4096)

        # 语义原型 soft prompt — 16 个连续 token
        # soft_prompt[b, i, :] = activations_mean[b, i] × proto_to_llm(prototypes[i])
        # 每个原型的激活值决定该 token 在 LLM embedding space 中的方向和强度
        wte = self.llm.base_model.model.transformer.word_embeddings
        if self.config.use_soft_prompt:
            soft_proto_embeds = proto_out['soft_prompt'].to(
                dtype=sys_embeds.dtype, device=HB.device)   # (B, K+L, 4096)
            P_proto = self.semantic_proto.num_prototypes + L  # K + L 个 token
        else:
            # 保留原离散路径（use_soft_prompt=False）
            proto_ids_padded = proto_out['proto_token_ids'].to(HB.device)
            soft_proto_embeds = wte(proto_ids_padded)        # (B, P_proto, 4096)
            P_proto = soft_proto_embeds.shape[1]

        # 拼接 prefix: [System | SoftProto(K+L)]
        prefix_embeds = torch.cat(
            [sys_embeds, soft_proto_embeds], dim=1)
        T_prefix = prefix_embeds.shape[1]                  # S + P_proto + P

        # 序列长度安全检查（prefix）
        if T_prefix > 2048:
            raise RuntimeError(
                f"LLM prefix too long: T_prefix={T_prefix} > 2048. "
                f"Reduce proto text or time windows."
            )

        is_training = self.training and inference_mode is None

        # ════════════════════════════════════════════════════════════
        # 3b. 训练时：构造 COT target text + embed → 拼接 prefix
        #     COT target 仅依赖 proto activations 和 DeepSeek 文本，
        #     不依赖 M（M 在 Step 4 中有梯度计算），无需两遍 Pass。
        # ════════════════════════════════════════════════════════════
        if is_training:
            self._debug_step += 1

            # ── Per-epoch 监控日志 ──
            if epoch != self._st_last_log_epoch:
                if self._st_last_log_epoch >= 0:
                    self._log_epoch_summary()
                self._st_last_log_epoch = epoch
                self._st_m_epoch_sum = 0.0
                self._st_m_epoch_count = 0

            # ── 构造 COT target（无需 M/L）──
            target_ids = self._build_cot_target_ids(
                B,
                activations_mean=activations_mean,
                sample_indices=sample_indices,
                mode=self.config.cot_target_mode,
            ).to(HB.device)

            # 序列总长度安全截断
            T_target = target_ids.shape[1]
            max_allowed = 2048 - T_prefix
            if T_target > max_allowed:
                target_ids = target_ids[:, -max_allowed:]
                T_target = max_allowed

            target_embeds = wte(target_ids.clamp(min=0))
            inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

            # [LOG] 捕获当前 step 文本信息 (B=1)
            _ids = target_ids[0].tolist()
            _ids_valid = [t for t in _ids if t >= 0]
            self._st_cur_text_type = 'template'
            self._st_cur_text_preview = self.tokenizer.decode(_ids_valid, skip_special_tokens=False)
            self._st_cur_text_len = len(_ids_valid)
        else:
            inputs_embeds = prefix_embeds
            target_ids = None

        T_total = inputs_embeds.shape[1]

        # ════════════════════════════════════════════════════════════
        # 3c. 因果 mask + 2D position_ids（ChatGLM-6B 格式）
        # ════════════════════════════════════════════════════════════
        seq_pos = torch.arange(T_total, device=inputs_embeds.device,
                               dtype=torch.long).unsqueeze(0).expand(B, -1)
        position_ids = torch.stack([seq_pos, seq_pos.clone()], dim=1)
        # 因果 mask：True = 不可见（上三角），False = 可 attend（下三角 + 对角线）
        causal = torch.triu(torch.ones(T_total, T_total, device=inputs_embeds.device,
                                       dtype=torch.bool), diagonal=1)
        attn_mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

        # 3d. 调用 transformer（LoRA 层注入在子模块中）
        use_cache = (inference_mode == "full")
        output = self.llm.model.transformer(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attn_mask,
            output_hidden_states=False,
            use_cache=use_cache,
            return_dict=True,
        )
        HL = output.last_hidden_state.transpose(0, 1)     # (B, T_total, 4096)

        # ════════════════════════════════════════════════════════════
        # Step 3e: 文本生成（仅完整推理模式 — 提前到 llm_pooled 之前，
        #          使生成文本的 hidden states 能参与知识掩膜和分类）
        # ════════════════════════════════════════════════════════════
        generated_text = None
        if inference_mode == "full":
            with torch.no_grad():
                try:
                    prefix_last_hidden = output.last_hidden_state[-1:]
                    generated_text, gen_ids, gen_hidden, gen_mask = \
                        self._autoregressive_generate(
                            prefix_last_hidden=prefix_last_hidden,
                            past_key_values=output.past_key_values,
                            T=T_prefix, B=B,
                            temperature=0.7,
                            top_p=0.9,
                            return_hidden_states=True,
                        )
                    gen_hidden = gen_hidden.to(dtype=HL.dtype)
                    gen_mask = gen_mask.to(device=HL.device)
                except Exception as e:
                    _log.warning(f"Text generation failed: {e}")
                    import traceback
                    _log.warning(traceback.format_exc())
                    generated_text = [f"[Generation error: {e}]"] * B
                    gen_hidden = None

        # 3f. llm_pooled
        if inference_mode == "full" and gen_hidden is not None \
                and gen_hidden.shape[1] > 0:
            # Full inference: SoftProto + generated text hidden states
            # 生成文本的 hidden state 包含诊断描述语义，增强掩膜判别力
            soft_proto_hidden = HL[:, T_prefix - P_proto : T_prefix, :]
            gen_hidden_valid = gen_hidden * gen_mask.unsqueeze(-1).to(
                gen_hidden.dtype)
            pool_hidden = torch.cat([soft_proto_hidden, gen_hidden_valid],
                                    dim=1)  # (B, P_proto + num_gen, 4096)
            pool_mask = torch.cat([
                torch.ones(B, P_proto, device=HL.device,
                           dtype=gen_mask.dtype),
                gen_mask
            ], dim=1)
            llm_pooled = (
                pool_hidden * pool_mask.unsqueeze(-1).to(pool_hidden.dtype)
            ).sum(dim=1) / pool_mask.sum(dim=1, keepdim=True).clamp(
                min=1).to(pool_hidden.dtype)
        elif is_training:
            # 训练时：SoftProto + Target mean pool
            # Causal attention 确保 target token 不可见未来位置，无标签泄漏风险。
            llm_pooled = HL[:, T_prefix - P_proto : T_total, :].mean(dim=1)
        else:
            # Fast inference: SoftProto only
            llm_pooled = HL[:, T_prefix - P_proto : T_prefix, :].mean(dim=1)

        # ════════════════════════════════════════════════════════════
        # Step 4: 知识掩膜 + per-channel 调制 + 分类 (tech.md §4.1 Step 4-5)
        # ════════════════════════════════════════════════════════════
        M = self.mask_head(llm_pooled, H_ch)               # (B, C), [0.05, 2.0]

        # ── M 监控：记录 per-step mask 均值（per-epoch 汇总）──
        if is_training:
            self._st_m_epoch_sum += M.detach().mean().item()
            self._st_m_epoch_count += 1

        # Per-channel 调制: M (B, C) → 广播到每个特征维度 (B, 128, C)
        H_ch_masked = H_ch * M.unsqueeze(1)                # (B, 128, C)  per-channel scale
        H_ch_fused = H_ch + H_ch_masked                    # (B, 128, C)  残差融合

        # 聚合为分类特征：mean-pool over channels
        H_cls = H_ch_fused.mean(dim=-1)                    # (B, 128)

        logits_bnc = self.cls_bnc(H_cls)                  # (B, num_classes)
        L_cla_bnc = F.cross_entropy(logits_bnc, labels)
        L_entropy = mask_entropy_loss(M)

        # ════════════════════════════════════════════════════════════
        # Step 5: LLM 直接分类监督 (tech.md §4.1 Step 5)
        # ════════════════════════════════════════════════════════════

        # LLM 直接分类监督 (tech.md §5.1)
        # 从 llm_pooled 直接预测疾病类别，为 LLM 提供直接的梯度信号，
        # 防止通过 mask_head 间接回传的梯度过弱导致 LLM 训练坍缩。
        L_cla_llm = F.cross_entropy(self.cls_llm(llm_pooled), labels)

        # 5b. 因果 LM 文本生成损失（训练时）
        # L_text: 教 lm_head 从 prefix hidden state 解码 COT 诊断分析文本
        if is_training:
            lm_head = self.llm.model.lm_head
            ln_f = self.llm.model.transformer.final_layernorm
            # HL at positions [T_prefix-1, T_total-1) predict tokens at [T_prefix, T_total)
            HL_for_text = HL[:, T_prefix - 1 : T_total - 1, :]  # (B, T_target, 4096)
            logits_text = lm_head(ln_f(HL_for_text))
            L_text = F.cross_entropy(
                logits_text.reshape(-1, logits_text.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

        else:
            L_text = torch.tensor(0.0, device=HB.device)

        L_total = (L_cla_bnc
                   + self.config.lambda_cla_llm * L_cla_llm
                   + self.config.lambda_text * L_text
                   + self.config.lambda_div * L_div
                   + self.config.lambda_entropy * L_entropy)

        # ════════════════════════════════════════════════════════════════
        # 窗口累积 + 日志输出（每 log_steps 步）
        # ════════════════════════════════════════════════════════════════
        if is_training:
            self._st_win_count += 1
            w = self._st_win_losses
            w[0] += L_cla_bnc.item();         w[1] += L_div.item()
            w[2] += L_entropy.item();         w[3] += L_cla_llm.item()
            w[4] += L_text.item();            w[5] += L_total.item()

            if self._debug_step % self.config.self_train_log_steps == 0:
                import torch.distributed as _dist
                _is_rank0 = (not _dist.is_initialized() or _dist.get_rank() == 0)
                if _is_rank0:
                    n = max(self._st_win_count, 1)
                    # ── 行1: 窗口平均 loss ──
                    _log.info("[Step %d] train avg_loss(cla=%.3f llm=%.3f text=%.3f "
                              "div=%.4f ent=%.4f → total=%.3f)",
                              self._debug_step,
                              w[0]/n, w[3]/n, w[4]/n, w[1]/n, w[2]/n, w[5]/n)

                    # ── 行2: 当前 step LLM 完整输入文本 (B=1) ──
                    _sys = (self._system_prompt_text or "")
                    if self._st_cur_text_type is not None:
                        _tgt = (self._st_cur_text_preview or "")
                        _log.info("[Step %d] LLM_INPUT: [SYS]\"%s\" "
                                  "[PROTO]<K+L soft tokens> "
                                  "[TARGET]\"%s\"",
                                  self._debug_step, _sys, _tgt)
                    else:
                        _log.info("[Step %d] LLM_INPUT: [SYS]\"%s\" "
                                  "[PROTO]<K+L soft tokens> "
                                  "[TARGET]<none>",
                                  self._debug_step, _sys)

                    # ── 行3: 当前 step 原型激活 + M 掩膜 ──
                    _proto_vals = " ".join(f"{activations_mean[0, i].item():.2f}" for i in range(activations_mean.shape[1]))
                    _m_vals = " ".join(f"{M[0, i].item():.2f}" for i in range(M.shape[1]))
                    _log.info("[Step %d] Proto=[%s] div=%.4f | M=[%s]",
                              self._debug_step, _proto_vals, L_div.item(), _m_vals)

                self._st_win_reset()

        return ModelOutputs(
            logits=logits_bnc,
            loss=(L_cla_bnc, L_div, L_entropy, L_cla_llm, L_text),
            hidden_state={
                'HB': HB,
                'H_ch': H_ch,
                'H_ch_masked': H_ch_masked,
                'H_ch_fused': H_ch_fused,
                'H_cls': H_cls,
                'llm_pooled': llm_pooled,
                'M': M,
                'proto_activations': activations_mean,
                'generated_text': generated_text,
            }
        )

    # ── 自回归生成 (tech.md §7.1) ────────────────────────────────────
    # 使用 past_key_values 手动逐 token 生成，避免依赖
    # ChatGLMForConditionalGeneration.generate()（不支持 inputs_embeds）。
    def _autoregressive_generate(self, prefix_last_hidden, past_key_values,
                                  T, B, max_new_tokens=None,
                                  temperature=0.7, top_p=0.9,
                                  return_hidden_states=False):
        device = prefix_last_hidden.device
        transformer = self.llm.model.transformer
        lm_head = self.llm.model.lm_head
        wte = transformer.word_embeddings
        eos_token_id = self.eos_token_id

        # 硬编码最大生成长度，平衡诊断完整性与推理速度
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        # Step 1: 从 prefix 末位 hidden state 采样第一个生成 token
        # prefix_last_hidden: (1, B, 4096)
        logits = lm_head(transformer.final_layernorm(prefix_last_hidden))
        logits = logits.squeeze(0)                           # (B, vocab)

        # Top-p (nucleus) 过滤 — 与后续 token 采样保持一致
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        indices_to_remove = remove.scatter(1, sorted_indices, remove)
        logits[indices_to_remove] = -float('Inf')

        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)             # (B, 1)
        generated_ids = [next_token]

        past_kv = past_key_values
        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        # Hidden state collection for "full" inference mode
        gen_hidden_list = []
        gen_lengths = torch.full((B,), max_new_tokens, dtype=torch.long,
                                 device=device)
        eos_seen = torch.zeros(B, dtype=torch.bool, device=device)

        # Step 2: 自回归循环
        for step in range(max_new_tokens - 1):
            token_embed = wte(next_token)                    # (B, 1, 4096)

            # 2D position_ids: (B, 2, seq_len)，沿用 prefix 的单调递增约定
            gen_pos = T + step
            gen_pos_ids = torch.full((B, 2, 1), gen_pos,
                                     device=device, dtype=torch.long)

            # 单 token 的 trivial attention mask (0 = attend)
            attn_mask_step = torch.zeros(B, 1, 1, 1,
                                         device=device, dtype=torch.bool)

            step_out = transformer(
                inputs_embeds=token_embed,
                position_ids=gen_pos_ids,
                attention_mask=attn_mask_step,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = step_out.past_key_values

            # 取本步 logits + hidden state（对应刚嵌入的 token 的上下文表示）
            step_hidden = step_out.last_hidden_state[-1:]    # (1, B, 4096)

            if return_hidden_states:
                step_hidden_t = step_hidden.transpose(0, 1)  # (B, 1, 4096)
                gen_hidden_list.append(step_hidden_t)

            step_logits = lm_head(transformer.final_layernorm(step_hidden))
            step_logits = step_logits.squeeze(0)              # (B, vocab)

            # Top-p (nucleus) 采样
            sorted_logits, sorted_indices = torch.sort(
                step_logits, descending=True)
            cum_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            indices_to_remove = remove.scatter(
                1, sorted_indices, remove)
            step_logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(step_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)         # (B, 1)

            # 已结束的序列用 eos 填充
            next_token = (next_token * unfinished.unsqueeze(1)
                          + eos_token_id * (~unfinished).unsqueeze(1))
            generated_ids.append(next_token)

            # 记录每个序列首次出现 EOS 的位置（用于 hidden state mask）
            is_first_eos = (next_token.squeeze(1) == eos_token_id) & ~eos_seen
            gen_lengths[is_first_eos] = step + 1  # step+1 = 该 token 在 generated_ids 中的索引
            eos_seen = eos_seen | is_first_eos

            unfinished = unfinished & (next_token.squeeze(1) != eos_token_id)
            if not unfinished.any():
                break

        # Step 3: 解码
        gen_ids = torch.cat(generated_ids, dim=1)            # (B, num_gen)
        if return_hidden_states:
            num_gen = len(gen_hidden_list)
            if num_gen > 0:
                gen_hidden = torch.cat(gen_hidden_list, dim=1)  # (B, num_gen, 4096)
                gen_mask = torch.arange(num_gen, device=device).unsqueeze(0) \
                           < gen_lengths.unsqueeze(1)
            else:
                gen_hidden = torch.zeros(B, 0, 4096, device=device)
                gen_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
            return (self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True),
                    gen_ids, gen_hidden, gen_mask)
        return (self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True),
                gen_ids)

    # ── 构造 COT L_text target（证据摘要）─────────────────────────
    def _build_cot_target_ids(self, B,
                              activations_mean=None,
                              sample_indices=None, mode="cot"):
        """构造 per-sample COT target token IDs，基于语义原型激活值。

        先拼接完整字符串再一次性 tokenize，避免 tokenizer 在中文标点
        周围插入多余空格。

        Args:
            B:                int              batch size
            activations_mean: (B, 16)          proto 激活值（时间平均）
            sample_indices:   (B,) or None     per-sample 原始数据索引（仅 cot 模式）
            mode:             str              "cot" or "template"

        Returns:
            (B, max_len) padded LongTensor, pad 位 = -100 (CE ignore_index)
        """
        import random

        if activations_mean is None:
            raise ValueError("_build_cot_target_ids: activations_mean is required")

        K = activations_mean.shape[1]
        k_ab = min(self._top_k_ab, K)
        k_pr = min(self._top_k_pr, K)

        _, asc_idx_all = activations_mean.topk(k_ab, dim=1, largest=False)
        _, desc_idx_all = activations_mean.topk(k_pr, dim=1)

        encode = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        all_ids = []

        for b in range(B):
            # ── 组装 evidence items ──
            items = []
            for idx_t in asc_idx_all[b, :3]:
                items.append(self._proto_items[idx_t.item()]['abnormal'])
            for idx_t in desc_idx_all[b, :3]:
                items.append(self._proto_items[idx_t.item()]['preserved'])

            # ── 完整文本拼接后一次性 tokenize ──
            text = self._cot_prefix + "、".join(items)
            if mode == "cot":
                text += self._cot_interpretation_prefix
                if sample_indices is not None:
                    ds_sentences = self._ds_sentences.get(
                        int(sample_indices[b].item()), [])
                    if ds_sentences:
                        text += random.choice(ds_sentences)

            ids = encode(text)
            ids.append(self._eop_token_id)
            all_ids.append(ids)

        # ── Pad ──
        max_len = max(len(seq) for seq in all_ids)
        padded = torch.full((B, max_len), -100, dtype=torch.long)
        for i, seq in enumerate(all_ids):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        return padded

    # ── Per-epoch 监控日志 (tech2.md) ───────────────────────────────

    def _log_epoch_summary(self):
        """Per-epoch 监控汇总：M 趋势 + 原型坍缩检测。仅 rank 0 输出。"""
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        epoch = self._st_last_log_epoch

        # ── M 趋势 ──
        if self._st_m_epoch_count > 0:
            m_avg = self._st_m_epoch_sum / self._st_m_epoch_count
        else:
            m_avg = float('nan')
        m_str = f"M mean={m_avg:.4f}"

        # ── 原型坍缩检测 ──
        protos = F.normalize(self.semantic_proto.prototypes, dim=-1)   # (16, D)
        cos_mat = protos @ protos.T                                    # (16, 16)
        mask = ~torch.eye(16, dtype=torch.bool, device=cos_mat.device)
        cos_max = cos_mat[mask].max().item()
        proto_str = f"Proto cos_max={cos_max:.4f}"

        _log.info(f"[Epoch {epoch + 1}] {m_str} | {proto_str}")

    def flush_epoch_summary(self):
        """训练结束时输出最后一个 epoch 的汇总日志。"""
        if self._st_last_log_epoch >= 0:
            self._log_epoch_summary()
            self._st_last_log_epoch = -1  # 防止重复调用

    # ── 窗口级日志累加器 (per log_steps) ────────────────────────────────

    def _st_win_reset(self):
        """重置窗口累加器（每次日志输出后调用）。"""
        self._st_win_count = 0
        self._st_win_losses = [0.0] * 6   # cla_bnc, div, entropy, cla_llm, text, total
