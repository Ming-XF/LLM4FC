"""KnowledgeLLM: OFA-style EEG encoders with activation-derived knowledge.

Inputs are one 10-second EEG segment together with dynamic FC and PSD sidecar
features.  Each modality keeps its natural patch count; only the complete
knowledge-token sequence is mean pooled after the first six Llama layers.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..base import BaseConfig, ModelOutputs
from .knowledge_bank import (
    ASPECTS,
    DEFAULT_KNOWLEDGE_TEXTS,
    extract_knowledge_bank,
    load_knowledge_bank,
)


_EPS = 1e-8


class OFAPatchEncoder(nn.Module):
    """Patch and value embedding used by the OFA baseline.

    The sequence length is determined by the input and patch parameters.  No
    adaptive pooling is applied, so different modalities retain their natural
    number of tokens.  No position embedding is added here: the frozen Llama
    backbone applies RoPE at its input stage, and a fixed positional term
    inside the encoder would dominate the conv output at initialization.
    """

    def __init__(self, in_channels, patch_size, stride, d_model, dropout=0.1):
        super().__init__()
        if in_channels <= 0 or patch_size <= 0 or stride <= 0:
            raise ValueError("OFA encoder dimensions and stride must be positive")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.value_embedding = nn.Conv1d(
            in_channels * patch_size,
            d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.value_embedding.weight,
                                mode="fan_in", nonlinearity="leaky_relu")
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"OFA encoder expects (B, {self.in_channels}, L), "
                f"got {tuple(x.shape)}")
        length = x.shape[-1]
        if length < self.patch_size:
            x = F.pad(x, (0, self.patch_size - length), mode="replicate")
            length = x.shape[-1]
        # Match OFA's replication padding, which preserves a final partial
        # patch without forcing every modality to a common output length.
        x = F.pad(x, (0, self.stride), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_size,
                           step=self.stride)
        # (B, C, N, P) -> (B, N, C*P)
        patches = patches.permute(0, 2, 1, 3).reshape(
            x.shape[0], -1, self.in_channels * self.patch_size)
        h = self.value_embedding(patches.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        return self.dropout(h)


class ReprogrammingLayer(nn.Module):
    """Cross-attention from OFA feature tokens to one aspect knowledge bank."""

    def __init__(self, d_model, n_heads, d_llm, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_llm, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, source, value):
        # query (B,L,d_model), source/value (P,d_llm)
        b, length, _ = query.shape
        q = self.query_projection(query).view(
            b, length, self.n_heads, self.head_dim)
        k = self.key_projection(source).view(
            source.shape[0], self.n_heads, self.head_dim)
        # key 侧单位化（scaled-cosine）：词表嵌入尺度远小于 query 尺度，
        # 不归一化时 key 间差异在 score 域被压到 ~1e-4，attention 退化为均匀。
        k = F.normalize(k, dim=-1)
        scores = torch.einsum("blhe,she->bhls", q, k)
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        attention = self.dropout(torch.softmax(scores, dim=-1))
        return torch.einsum("bhls,sd->blhd", attention, value).mean(dim=2)


class KnowledgeMapper(nn.Module):
    """Map the extracted knowledge tokens to P trainable fusion tokens.

    Initialization follows the TimeLLMKnowledge semantic bank
    (``_group_average_init``): each output row averages a disjoint contiguous
    group of bank tokens with uniform weights and no perturbation, so the P
    fusion tokens are distinct from the start.
    """

    def __init__(self, token_ids, num_fusion_tokens):
        super().__init__()
        token_ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
        if token_ids.numel() == 0:
            raise ValueError("Each KnowledgeLLM aspect needs at least one token id")
        if token_ids.numel() < num_fusion_tokens:
            raise ValueError(
                f"Knowledge bank has {token_ids.numel()} tokens but "
                f"{num_fusion_tokens} fusion tokens were requested")
        self.register_buffer("token_ids", token_ids)
        self.mapping = nn.Linear(token_ids.numel(), num_fusion_tokens,
                                 bias=False)
        # 分段平均初始化（对齐 TimeLLMKnowledge 语义银行）：第 i 个输出
        # 平均第 i 段 bank token，P 个输出出生即互不相同。
        with torch.no_grad():
            self.mapping.weight.zero_()
            for i in range(num_fusion_tokens):
                start = (i * token_ids.numel()) // num_fusion_tokens
                end = ((i + 1) * token_ids.numel()) // num_fusion_tokens
                end = max(end, start + 1)
                self.mapping.weight[i, start:end] = 1.0 / float(end - start)

    def forward(self, word_embedding):
        source = word_embedding[self.token_ids].float()
        # Linear acts over the vocabulary-token dimension, yielding P vectors.
        return self.mapping(source.transpose(0, 1)).transpose(0, 1)


class SupervisedContrastiveLoss(nn.Module):
    """Supervised InfoNCE, robust to batches without a positive pair."""

    def __init__(self, temperature=0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("contrastive temperature must be positive")
        self.temperature = temperature

    def forward(self, features, labels):
        # features (B,A,D), labels (B,)
        if features.shape[0] < 2:
            return features.new_zeros(())
        z = F.normalize(features, dim=-1)
        logits = torch.einsum("bad,cad->bac", z, z) / self.temperature
        b = features.shape[0]
        eye = torch.eye(b, device=features.device, dtype=torch.bool)
        positives = labels[:, None].eq(labels[None, :]) & ~eye
        valid = positives.any(dim=-1)
        if not bool(valid.any()):
            return features.new_zeros(())
        logits = logits.masked_fill(eye[:, None, :], -float("inf"))
        log_denominator = torch.logsumexp(logits, dim=-1)
        log_numerator = torch.logsumexp(
            logits.masked_fill(~positives[:, None, :], -float("inf")), dim=-1)
        return (log_denominator - log_numerator)[valid].mean()


class KnowledgeLLMConfig(BaseConfig):
    def __init__(self, node_size, time_series_size=2000, d_model=128,
                 n_heads=8, num_fusion_tokens=8, dropout=0.1,
                 encoder_dim=None, sample_rate=200, eeg_seconds=10,
                 raw_patch_size=200, raw_stride=200, psd_patch_size=8,
                 psd_stride=4, fc_patch_size=1, fc_stride=1,
                 contrast_lambda=0.1, contrast_temperature=0.1,
                 knowledge_bank_path="", knowledge_top_k=8,
                 knowledge_top_m=32, knowledge_temperature=1.0,
                 llm_layers=6, task_type="classification", output_dim=2,
                 llm_type="llama", llm_path="./model/deepseek-r1-distill-llama-8B",
                 llm_init="pretrained"):
        super().__init__(node_size=node_size, output_dim=output_dim,
                         d_model=d_model, dropout=dropout)
        if llm_type != "llama":
            raise ValueError("KnowledgeLLM currently supports only llm_type='llama'")
        self.time_series_size = time_series_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_fusion_tokens = num_fusion_tokens
        self.encoder_dim = encoder_dim or d_model
        self.dropout = dropout
        self.sample_rate = sample_rate
        self.eeg_seconds = eeg_seconds
        self.raw_patch_size = raw_patch_size
        self.raw_stride = raw_stride
        self.psd_patch_size = psd_patch_size
        self.psd_stride = psd_stride
        self.fc_patch_size = fc_patch_size
        self.fc_stride = fc_stride
        self.contrast_lambda = contrast_lambda
        self.contrast_temperature = contrast_temperature
        self.knowledge_bank_path = knowledge_bank_path
        self.knowledge_top_k = knowledge_top_k
        self.knowledge_top_m = knowledge_top_m
        self.knowledge_temperature = knowledge_temperature
        self.llm_layers = llm_layers
        self.task_type = task_type
        self.output_dim = output_dim
        self.llm_type = llm_type
        self.llm_path = llm_path
        self.llm_init = llm_init


class Model(nn.Module):
    def __init__(self, config: KnowledgeLLMConfig):
        super().__init__()
        self.config = config
        self.task_type = config.task_type
        self._load_llm(config)

        channels = config.node_size
        if channels <= 1:
            raise ValueError("KnowledgeLLM requires at least two EEG channels")
        edge_count = channels * (channels - 1) // 2
        triu = torch.triu_indices(channels, channels, offset=1)
        self.register_buffer("fc_triu_indices", triu, persistent=False)

        d_encoder = config.encoder_dim
        self.raw_encoder = OFAPatchEncoder(
            channels, config.raw_patch_size, config.raw_stride, d_encoder,
            config.dropout)
        self.psd_encoder = OFAPatchEncoder(
            channels, config.psd_patch_size, config.psd_stride, d_encoder,
            config.dropout)
        self.fc_encoder = OFAPatchEncoder(
            edge_count, config.fc_patch_size, config.fc_stride, d_encoder,
            config.dropout)
        self.query_projections = nn.ModuleList(
            [nn.Linear(d_encoder, config.d_model) for _ in ASPECTS])

        banks = self._load_or_extract_knowledge(config)
        # DeepSpeed fp32 配置会把可训练 RMSNorm 提升为 fp32，与冻结的 bf16
        # q/k/v/o 权重混排导致前向 dtype 不匹配；将保留的浅层栈整体转 fp32
        # 保持骨干 dtype 一致（知识银行提取仍在截断前的 bf16 完整 8B 上进行）。
        self.llm = self.llm.float()
        self.knowledge_mappers = nn.ModuleList([
            KnowledgeMapper(
                banks[aspect]["token_ids"], config.num_fusion_tokens)
            for aspect in ASPECTS
        ])
        self.reprogrammers = nn.ModuleList([
            ReprogrammingLayer(
                config.d_model, config.n_heads, self.d_llm, config.dropout)
            for _ in ASPECTS
        ])
        # 推理用融合 token 快照：训练保存 checkpoint 前由 capture_knowledge_state
        # 写入；加载后 on_state_loaded 置位，前向直接使用该缓冲作为
        # source/value，不依赖重新提取的知识银行 token 集合。
        self.register_buffer(
            "knowledge_fusion_tokens",
            torch.zeros(len(ASPECTS), config.num_fusion_tokens, self.d_llm))
        self.use_captured_fusion = False

        self.contrast_projection = nn.Linear(d_encoder, 128)
        self.contrastive_loss = SupervisedContrastiveLoss(
            config.contrast_temperature)
        self.output_norm = nn.LayerNorm(self.d_llm)
        self.output_dropout = nn.Dropout(config.dropout)
        self.output_head = nn.Linear(self.d_llm, config.output_dim)

        # All backbone weights remain frozen; only RMSNorm parameters in the
        # retained six-layer stack are trainable, matching the OFA setting.
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        for name, parameter in self.llm.named_parameters():
            if "norm" in name.lower():
                parameter.requires_grad = True
        self.llm.eval()

    def _load_llm(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_path, trust_remote_code=True)
        if config.llm_init == "random":
            llm_config = AutoConfig.from_pretrained(
                config.llm_path, trust_remote_code=True)
            causal = AutoModelForCausalLM.from_config(
                llm_config, trust_remote_code=True)
            causal = causal.to(dtype=torch.bfloat16)
        else:
            causal = AutoModelForCausalLM.from_pretrained(
                config.llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        if not hasattr(causal, "model") or not hasattr(causal, "lm_head"):
            raise ValueError("KnowledgeLLM expects a Llama-style causal model")
        full_backbone = causal.model
        if not hasattr(full_backbone, "layers"):
            raise ValueError("LLM backbone does not expose a transformer layer list")
        if config.llm_layers <= 0 or config.llm_layers > len(full_backbone.layers):
            raise ValueError(
                f"llm_layers={config.llm_layers} exceeds available "
                f"layers={len(full_backbone.layers)}")
        # Keep the complete causal model until the offline bank is extracted;
        # the bank definition uses the original final hidden layer.
        self._full_causal_for_extraction = causal
        self.d_llm = int(full_backbone.config.hidden_size)
        self.llm = full_backbone
        self._full_causal_for_extraction.eval()
        if hasattr(self.llm, "gradient_checkpointing"):
            self.llm.gradient_checkpointing = False

    def _load_or_extract_knowledge(self, config):
        if config.knowledge_bank_path:
            banks = load_knowledge_bank(config.knowledge_bank_path)
        else:
            banks = extract_knowledge_bank(
                self._full_causal_for_extraction,
                self.tokenizer,
                texts=DEFAULT_KNOWLEDGE_TEXTS,
                top_k_positions=config.knowledge_top_k,
                top_m_tokens=config.knowledge_top_m,
                temperature=config.knowledge_temperature,
            )
        vocab_size = self.word_embeddings.shape[0]
        for aspect in ASPECTS:
            ids = banks[aspect]["token_ids"]
            if bool((ids >= vocab_size).any()):
                raise ValueError(
                    f"Knowledge bank {aspect} contains token id outside vocabulary")
        # Only after extraction retain the shallow OFA-style Llama stack.
        full_backbone = self._full_causal_for_extraction.model
        full_backbone.layers = nn.ModuleList(
            list(full_backbone.layers[:config.llm_layers]))
        self.llm = full_backbone
        del self._full_causal_for_extraction
        return banks

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "llm"):
            self.llm.eval()
        return self

    @property
    def word_embeddings(self):
        """Frozen LLM input embedding matrix without duplicate registration."""
        return self.llm.embed_tokens.weight

    def _encode_modalities(self, x, window_fc, window_psd):
        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(
                    "KnowledgeLLM uses one 10-second EEG segment per sample")
            x = x[:, 0]
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B,C,T), got {tuple(x.shape)}")
        expected_time = int(self.config.sample_rate * self.config.eeg_seconds)
        if x.shape[-1] != expected_time:
            raise ValueError(
                f"KnowledgeLLM expects a {self.config.eeg_seconds}s EEG segment "
                f"({expected_time} samples), got {x.shape[-1]}")
        # OFA 数据管线同款 per-sample 标准化（逐通道）：本仓库 raw EEG 存储
        # 尺度极小（~1e-5），不标准化则随机卷积输出被位置嵌入淹没。
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        x = (x - x_mean) / (x_std + _EPS)

        if window_fc.dim() == 3:
            window_fc = window_fc.unsqueeze(1)
        if window_fc.dim() != 4 or window_fc.shape[0] != x.shape[0]:
            raise ValueError(
                "window_fc must have shape (B,W,C,C), "
                f"got {tuple(window_fc.shape)}")
        if window_psd.dim() == 3:
            window_psd = window_psd.unsqueeze(1)
        if window_psd.dim() != 4 or window_psd.shape[0] != x.shape[0]:
            raise ValueError(
                "window_psd must have shape (B,W,C,F), "
                f"got {tuple(window_psd.shape)}")
        if window_fc.shape[2:] != (x.shape[1], x.shape[1]):
            raise ValueError("window_fc channel dimensions do not match x")
        if window_psd.shape[2] != x.shape[1]:
            raise ValueError("window_psd channel dimension does not match x")

        # Dynamic FC: each window is a point on the edge sequence.  The
        # upper-triangular vector avoids duplicating symmetric correlations.
        fc_edges = window_fc[:, :, self.fc_triu_indices[0],
                             self.fc_triu_indices[1]]
        fc_edges = fc_edges.transpose(1, 2).contiguous()  # (B,E,W)
        fc_tokens = self.fc_encoder(fc_edges)

        b, windows, channels, frequencies = window_psd.shape
        psd = window_psd.reshape(b * windows, channels, frequencies)
        psd_tokens = self.psd_encoder(psd)
        psd_tokens = psd_tokens.reshape(b, windows * psd_tokens.shape[1],
                                        psd_tokens.shape[-1])

        raw_tokens = self.raw_encoder(x)
        return fc_tokens, psd_tokens, raw_tokens

    def capture_knowledge_state(self):
        """Snapshot the current fusion tokens into the persistent buffer.

        Called by the trainer before saving a checkpoint so the trained
        mapper output travels with the checkpoint and inference can use it
        directly without relying on a re-extracted bank token set.
        """
        with torch.no_grad():
            for i, mapper in enumerate(self.knowledge_mappers):
                self.knowledge_fusion_tokens[i] = mapper(self.word_embeddings)

    def on_state_loaded(self, use_frozen_fusion):
        """Hook invoked by the trainer after loading a checkpoint."""
        self.use_captured_fusion = (
            use_frozen_fusion
            and bool(self.knowledge_fusion_tokens.abs().sum() > 0))

    def _knowledge_tokens(self, feature_sequences):
        tokens = []
        for i, (features, query_projection, mapper, reprogrammer) in enumerate(
                zip(feature_sequences, self.query_projections,
                    self.knowledge_mappers, self.reprogrammers)):
            query = query_projection(features)
            if self.use_captured_fusion:
                # 评估模式：直接使用 checkpoint 里的融合 token 快照，
                # 与实时 mapper 计算数学等价（同一权重、同一冻结词表）。
                source = self.knowledge_fusion_tokens[i]
            else:
                source = mapper(self.word_embeddings)
            tokens.append(reprogrammer(query, source, source))
        return tokens

    def _labels_for_loss(self, labels):
        if self.task_type == "regression":
            return labels.float().reshape(-1)
        if labels.dim() > 1 and labels.shape[-1] > 1:
            labels = labels.argmax(dim=-1)
        return labels.long().reshape(-1)

    def forward(self, x, window_fc, window_psd, window_mask=None,
                labels=None):
        if window_mask is not None:
            if window_mask.dim() == 1:
                window_mask = window_mask.unsqueeze(1)
            if window_mask.dim() != 2:
                raise ValueError(
                    f"window_mask must have shape (B,W), got {tuple(window_mask.shape)}")
            if window_fc.dim() >= 4 and window_mask.shape != window_fc.shape[:2]:
                raise ValueError(
                    "window_mask shape must match the FC batch/window dimensions")
            if not bool(window_mask.bool().all()):
                raise ValueError("KnowledgeLLM requires valid FC/PSD windows")
        features = self._encode_modalities(x, window_fc, window_psd)
        knowledge = self._knowledge_tokens(features)
        sequence = torch.cat(knowledge, dim=1)
        position_ids = torch.arange(
            sequence.shape[1], device=sequence.device).unsqueeze(0).expand(
                sequence.shape[0], -1)
        hidden = self.llm(
            inputs_embeds=sequence.to(dtype=self.word_embeddings.dtype),
            position_ids=position_ids,
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.float()
        pooled = hidden.mean(dim=1)
        logits = self.output_head(
            self.output_dropout(F.gelu(self.output_norm(pooled))))

        contrast_loss = logits.new_zeros(())
        loss = None
        if labels is not None:
            loss_labels = self._labels_for_loss(labels)
            if self.task_type == "regression":
                prediction = logits.squeeze(-1)
                loss = F.huber_loss(prediction.float(), loss_labels, delta=1.0)
            else:
                loss = F.cross_entropy(logits.float(), loss_labels)
                if self.config.contrast_lambda > 0:
                    modality_features = torch.stack(
                        [feature.mean(dim=1) for feature in features], dim=1)
                    projected = self.contrast_projection(modality_features)
                    contrast_loss = self.contrastive_loss(
                        projected, loss_labels)
                    loss = loss + self.config.contrast_lambda * contrast_loss

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                "fc_features": features[0],
                "psd_features": features[1],
                "raw_features": features[2],
                "knowledge_tokens": sequence,
                "pooled": pooled,
                "contrast_loss": contrast_loss,
            },
        )
