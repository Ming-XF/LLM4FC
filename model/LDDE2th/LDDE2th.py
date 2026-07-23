import torch
import torch.nn.functional as F
from torch import nn
from ..base import BaseConfig, ModelOutputs
from .LDDE2thLayers import BrainNetCNN, BNCFeatureProjector

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, get_peft_model

import logging

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# System prompt — 说明输入结构和诊断任务
# ═══════════════════════════════════════════════════════════════════════════════


def build_system_prompt():
    """构建 LLM system prompt，说明输入结构和任务目标。"""
    return (
        "你是一位神经影像专家。接下来你会收到一组脑功能网络特征token，"
        "它们由动态功能连接矩阵经神经网络编码提取。"
        "请基于这些特征，诊断该患者属于以下四类之一："
        "阿尔茨海默病、轻度认知障碍、主观认知下降或正常认知。"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


class LDDE2thConfig(BaseConfig):
    def __init__(self, node_size, num_classes, dfc_stride=5):
        super(LDDE2thConfig, self).__init__(node_size=node_size,
                                            num_classes=num_classes)
        self.dfc_stride = dfc_stride

        # ── BNC feature projector ──
        self.num_feature_tokens = 4           # BNC 特征投影为几个 LLM token

        # ── LoRA ──
        self.lora_r = 4
        self.lora_alpha = 8                    # scaling = α/r = 2.0

        # ── 日志 ──
        self.self_train_log_steps = 200


# ═══════════════════════════════════════════════════════════════════════════════
# LDDE2th — Simplified: BNC feature extractor → LLM → classification
# ═══════════════════════════════════════════════════════════════════════════════


class LDDE2th(nn.Module):
    """LDDE2th — 简化版：BNC 特征提取 → LLM 编码 → 分类。

    Architecture
    ────────────
    1. BNC 编码 DFC 逐时间窗 → F_per_window (B*L, 256)
    2. 时间平均 → HB (B, 256)
    3. BNCFeatureProjector → feature_tokens (B, num_tokens, 4096)
    4. LLM encode: [SystemPrompt | FeatureTokens] → HL
    5. llm_pooled = mean_pool(feature_token hidden states)
    6. cls_head → logits
    """

    def __init__(self, config: LDDE2thConfig):
        super().__init__()
        self.config = config
        C = config.node_size
        self.dfc_stride = config.dfc_stride

        # ── 1. BNC — DFC spatial encoder (feature extractor only) ──
        self.bnc = BrainNetCNN(C)

        # ── 2. BNC → LLM feature projector ──
        self.bnc_proj = BNCFeatureProjector(
            bnc_dim=256,
            llm_dim=4096,
            num_tokens=config.num_feature_tokens,
        )
        self.num_feature_tokens = config.num_feature_tokens

        # ── 3. Tokenizer + LLM ──
        tokenizer = AutoTokenizer.from_pretrained(
            "./model/chatglm-6b", trust_remote_code=True
        )
        base_llm = AutoModel.from_pretrained(
            "./model/chatglm-6b", trust_remote_code=True
        ).bfloat16()
        base_llm.transformer.gradient_checkpointing = False

        self.tokenizer = tokenizer
        wte = base_llm.transformer.word_embeddings

        # System Prompt → frozen text embeddings
        system_text = build_system_prompt()
        system_ids = tokenizer.encode(system_text)
        with torch.no_grad():
            system_embeds = wte(torch.tensor(system_ids))
        self.register_buffer("system_prompt_embeds", system_embeds)

        # LoRA r=4
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

        # ── 4. Classification head — llm_pooled → logits ──
        self.cls_head = nn.Linear(4096, config.num_classes)

    # ── Forward ────────────────────────────────────────────────────
    def forward(self, time_series, DFC, labels,
                gender=None, age=None, education=None):
        """
        Args:
            time_series: (B, C, Ts)   EEG time series（当前不使用）
            DFC:         (B, L_full, C, C)   dynamic FC matrices
            labels:      (B,)          ground-truth class indices
            gender, age, education: 人口学特征（当前不使用，保留兼容）
        """
        B, L_full, C, _ = DFC.shape

        # ── Step 0: 时序降采样 ──
        DFC = DFC[:, ::self.dfc_stride, :, :]           # (B, L, C, C)
        L = DFC.shape[1]

        # ════════════════════════════════════════════════════════════
        # Step 1: BNC 编码 → per-window features
        # ════════════════════════════════════════════════════════════
        DFC_flat = DFC.reshape(B * L, 1, C, C)           # (B*L, 1, C, C)
        DFC_flat = DFC_flat.to(dtype=next(self.bnc.parameters()).dtype)
        F_per_window = self.bnc(DFC_flat)                # (B*L, 256)
        HB = F_per_window.reshape(B, L, 256).mean(dim=1)  # (B, 256) time-mean pool

        # ════════════════════════════════════════════════════════════
        # Step 2: BNC features → LLM prompt tokens
        # ════════════════════════════════════════════════════════════
        feature_tokens = self.bnc_proj(HB)               # (B, num_tokens, 4096)

        # ════════════════════════════════════════════════════════════
        # Step 3: 组装 LLM 输入 [System | FeatureTokens]
        # ════════════════════════════════════════════════════════════
        sys_embeds = self.system_prompt_embeds.unsqueeze(0) \
                        .expand(B, -1, -1)               # (B, S, 4096)

        inputs_embeds = torch.cat(
            [sys_embeds, feature_tokens.to(dtype=sys_embeds.dtype,
                                           device=HB.device)],
            dim=1,
        )
        T_total = inputs_embeds.shape[1]
        T_prefix = T_total
        P_feat = self.num_feature_tokens

        # ════════════════════════════════════════════════════════════
        # Step 4: LLM 因果编码
        # ════════════════════════════════════════════════════════════
        seq_pos = torch.arange(T_total, device=inputs_embeds.device,
                               dtype=torch.long).unsqueeze(0).expand(B, -1)
        position_ids = torch.stack([seq_pos, seq_pos.clone()], dim=1)

        causal = torch.triu(torch.ones(T_total, T_total, device=inputs_embeds.device,
                                       dtype=torch.bool), diagonal=1)
        attn_mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

        output = self.llm.model.transformer(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attn_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        HL = output.last_hidden_state.transpose(0, 1)     # (B, T_total, 4096)

        # ════════════════════════════════════════════════════════════
        # Step 5: Pool feature token hidden states → classify
        # ════════════════════════════════════════════════════════════
        llm_pooled = HL[:, T_prefix - P_feat:, :].mean(dim=1)  # (B, 4096)
        logits = self.cls_head(llm_pooled)                     # (B, num_classes)
        L_cla = F.cross_entropy(logits, labels)

        return ModelOutputs(
            logits=logits,
            loss=L_cla,
            hidden_state={
                'HB': HB,
                'llm_pooled': llm_pooled,
            }
        )
