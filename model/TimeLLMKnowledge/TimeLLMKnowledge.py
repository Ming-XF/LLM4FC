"""TimeLLMKnowledge v2: 全局+局部知识 Token 编译器（patch 编码 + 捷径解除）。

输入三段：
  整分钟 EEG → 静态 FC + 数据集层频谱/形态统计 → 32 个全局 display stats
  6 个 10s 片段 → torch 内计算每片段 32 个局部 display stats（同 schema）
  任务 prompt（不变）

编码（全局块与局部块共享编译器参数）：
  q_i = g_num·RMSNorm(W_i·s_i) + g_raw·RMSNorm(R_i·Stem(EEG块))     (128)
  q → reprogram（跨注意力到词表原型）→ soft prompt (4096)
  k_i = ref_rms·RMSNorm(soft_prompt_i + β·phrase_i)   ← phrase 只作身份锚点

序列：[start][task][end][全局 32][局部 6×32][READOUT(固定 token)] → 冻结 LLM
→ 末 K 层 READOUT 位置聚合 → 线性头（bias 固定源域先验）+ 知识 bypass 头。

捷径解除（codex 评审配套）：
  - READOUT 为固定文本 token 词表向量（不可学习）；
  - 层聚合权重默认固定均匀（learn_layer_weights=False）；
  - 输出头 bias 冻结在归一化标签先验（回归 0.5 / 分类 0）；
  - bypass：z = mean_pool(q) → 小线性头 → y = y_llm + λ·y_bypass，
    λ 初始 0.1，给知识路径一条不经过冻结 LLM 的直连监督。
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..base import BaseConfig, ModelOutputs
from ..TimeLLM.graph_evidence import compute_graph_evidence
from .knowledge_compiler import KnowledgeCompiler, rms_norm
from .local_stats import segment_spectral_stats, segment_morph_stats
from .prompts import get_prompt_config

_EPS = 1e-8

# 层权重聚合（learn_layer_weights=True 时）：α = β·e_last + (1-β)·softmax(a/T)
_LAYER_BETA = 0.4
_LAYER_TEMP = 1.5


class RMSNorm(nn.Module):
    """RMS 归一化（手动实现，兼容旧版 torch）。

    learnable=False 时为无参数固定归一化（用于逐层读出前），
    learnable=True 时带可学习 scale（用于输出头前）。
    """

    def __init__(self, dim, learnable=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        if learnable:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", None)

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        out = x / rms
        if self.weight is not None:
            out = out * self.weight
        return out


class ReprogrammingLayer(nn.Module):
    """query 侧 d_model、key 4096→d_model、value 路径保持 4096 的跨注意力。

    与原版 ReprogrammingLayer 的区别：不做 value 投影与 out 投影，
    prototype 的 4096 维表达能力直接在 value 路径保留；softmax 仅沿
    prototype bank 维度（绝不在统计量之间做 attention，避免知识混合）。
    """

    def __init__(self, d_model, n_heads, d_llm, attention_dropout=0.1):
        super().__init__()
        d_keys = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        # target (B, L, d_model)；source/value (S, d_llm)
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        E = target_embedding.shape[-1] // H

        q = self.query_projection(target_embedding).view(B, L, H, E)
        k = self.key_projection(source_embedding).view(S, H, E)
        scores = torch.einsum("blhe,she->bhls", q, k) * (1.0 / sqrt(E))
        attn = self.dropout(torch.softmax(scores, dim=-1))       # (B,H,L,S)
        # value 保持 4096：各头注意力作用于同一完整 prototype 向量后取均值
        out = torch.einsum("bhls,sd->blhd", attn, value_embedding)
        return out.mean(dim=2)                                   # (B,L,d_llm)


class PatchStem(nn.Module):
    """raw EEG 块 (B, C, T) → (B, d_feat)：Conv1d 下采样 + 自适应池化 + RMSNorm。

    60s 全局块与 10s 局部块共享：时间维池化到固定 8 桶，输出维度恒定。
    无 BatchNorm（跨域稳健，LayerNorm 系归一化）。
    """

    def __init__(self, in_channels, d_feat=256, kernel=25):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=kernel,
                              stride=kernel)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        self.d_feat = d_feat
        assert d_feat == 32 * 8, "d_feat 需为 32×池化桶数(8)"

    def forward(self, x):
        h = F.gelu(self.conv(x))                      # (B, 32, ~T/kernel)
        h = F.adaptive_avg_pool1d(h, 8)               # (B, 32, 8)
        return rms_norm(h.flatten(1))                 # (B, d_feat)


class TimeLLMKnowledgeConfig(BaseConfig):
    def __init__(self, node_size,
                 time_series_size=12000,
                 d_model=128,
                 n_heads=8,
                 num_prototypes=1000,
                 dropout=0.1,
                 num_stats=32,
                 readout_layers=4,
                 d_feat=256,
                 hz=200,
                 num_local_segments=6,
                 use_local_knowledge=True,
                 use_raw_branch=True,
                 use_numeric_branch=True,
                 use_knowledge_bypass=True,
                 freeze_head_bias=True,
                 learn_layer_weights=False,
                 phrase_beta_init=0.25,
                 bypass_gate_init=0.1,
                 task_type='classification',
                 output_dim=2,
                 dataset_name='CAUEEG',
                 llm_type='chatglm',
                 llm_path='./model/chatglm-6b',
                 use_task_prompt=False,
                 llm_init='pretrained'):
        super().__init__(node_size=node_size, output_dim=output_dim)
        assert 1 <= readout_layers <= 32, \
            f"readout_layers={readout_layers} 超出 1..32（K>32 会读入 h0）"
        self.time_series_size = time_series_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_prototypes = num_prototypes
        self.dropout = dropout
        self.num_stats = num_stats
        self.readout_layers = readout_layers
        self.d_feat = d_feat
        self.hz = hz
        self.num_local_segments = num_local_segments
        self.use_local_knowledge = use_local_knowledge
        self.use_raw_branch = use_raw_branch
        self.use_numeric_branch = use_numeric_branch
        self.use_knowledge_bypass = use_knowledge_bypass
        self.freeze_head_bias = freeze_head_bias
        self.learn_layer_weights = learn_layer_weights
        self.phrase_beta_init = phrase_beta_init
        self.bypass_gate_init = bypass_gate_init
        self.task_type = task_type
        self.output_dim = output_dim
        self.dataset_name = dataset_name
        self.llm_type = llm_type
        self.llm_path = llm_path
        self.llm_init = llm_init
        self.use_task_prompt = use_task_prompt


class Model(nn.Module):
    def __init__(self, config: TimeLLMKnowledgeConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_prototypes = config.num_prototypes
        self.n_heads = config.n_heads
        self.d_llm = 4096
        self.llm_type = config.llm_type
        self.llm_path = config.llm_path

        # ── 冻结 LLM（加载方式与 TimeLLMPatch 一致）──
        if self.llm_type == 'chatglm':
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_path, trust_remote_code=True)
            if config.llm_init == 'random':
                llm_config = AutoConfig.from_pretrained(
                    self.llm_path, trust_remote_code=True)
                self.llm = AutoModel.from_config(
                    llm_config, trust_remote_code=True).bfloat16()
            else:
                self.llm = AutoModel.from_pretrained(
                    self.llm_path, trust_remote_code=True).bfloat16()
            self.llm.transformer.gradient_checkpointing = False
            self.position_encoding_2d = \
                self.llm.transformer.position_encoding_2d
            self._word_embeddings = self.llm.transformer.word_embeddings
            self._transformer = self.llm.transformer
        elif self.llm_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
            if config.llm_init == 'random':
                llm_config = AutoConfig.from_pretrained(self.llm_path)
                self.llm = AutoModel.from_config(
                    llm_config, torch_dtype=torch.bfloat16)
            else:
                self.llm = AutoModel.from_pretrained(
                    self.llm_path, torch_dtype=torch.bfloat16)
            self.llm.config.use_cache = False
            self.position_encoding_2d = False
            self._word_embeddings = self.llm.embed_tokens
            self._transformer = self.llm
        else:
            raise ValueError(f"Unsupported llm_type: {self.llm_type}")

        # ── 文本 prompt 缓冲（与 TimeLLMPatch 相同的 tag/任务 prompt）──
        prompt_config = get_prompt_config(config.dataset_name)
        task_ids = self.tokenizer.encode(prompt_config.prompt_task)
        with torch.no_grad():
            task_embeds = self._word_embeddings(torch.tensor(task_ids))
        self.register_buffer("task_prompt_embeddings", task_embeds)
        self.P_task = task_embeds.shape[0]

        start_tag_ids = self.tokenizer.encode("<start_prompt>\n")
        with torch.no_grad():
            start_tag_embeds = self._word_embeddings(
                torch.tensor(start_tag_ids))
        self.register_buffer("start_tag_embeddings", start_tag_embeds)
        self.P_start = start_tag_embeds.shape[0]

        end_tag_ids = self.tokenizer.encode("\n<end_prompt>")
        with torch.no_grad():
            end_tag_embeds = self._word_embeddings(torch.tensor(end_tag_ids))
        self.register_buffer("end_tag_embeddings", end_tag_embeds)
        self.P_end = end_tag_embeds.shape[0]

        self.word_embeddings = self._word_embeddings.weight
        self.vocab_size = self.word_embeddings.shape[0]

        # ── 词表 embedding 尺度参照（RMS 中位数）──
        ref_rms = self.word_embeddings.detach().float().pow(2).mean(
            -1).sqrt().median().clamp_min(_EPS)
        self.register_buffer("ref_rms", ref_rms)

        # ── PatchStem + 知识编译器 + reprogram ──
        self.patch_stem = PatchStem(config.node_size, d_feat=config.d_feat)
        self.knowledge_compiler = KnowledgeCompiler(
            d_llm=self.d_llm,
            d_model=config.d_model,
            num_stats=config.num_stats,
            tokenizer=self.tokenizer,
            word_embeddings=self._word_embeddings,
            llm_type=self.llm_type,
            d_feat=config.d_feat,
        )
        self.knowledge_compiler.use_numeric_branch = \
            config.use_numeric_branch
        self.knowledge_compiler.use_raw_branch = config.use_raw_branch

        self.mapping_layer = nn.Linear(self.vocab_size, self.num_prototypes)
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
        )

        # phrase 融合系数（初始 0.25，phrase 只作身份锚点，不参与 q）
        self.phrase_beta = nn.Parameter(
            torch.tensor(config.phrase_beta_init))

        # ── READOUT：固定文本 token 词表向量（不可学习，焊死捷径）──
        readout_ids = self.tokenizer.encode(
            " age", add_special_tokens=False)
        if not readout_ids:
            readout_ids = self.tokenizer.encode(
                "age", add_special_tokens=False)
        if not readout_ids:
            readout_ids = [self.tokenizer.eos_token_id]
        with torch.no_grad():
            rv = self._word_embeddings(
                torch.tensor(readout_ids)).mean(0).float()
        rms = rv.pow(2).mean().sqrt().clamp_min(_EPS)
        self.register_buffer("readout_embedding", rv / rms * ref_rms)

        # ── 最后 K 层聚合权重：默认固定均匀；learn=True 时锚定 softmax ──
        self.layer_logits = (
            nn.Parameter(torch.zeros(config.readout_layers))
            if config.learn_layer_weights else None)

        # ── 输出头：bias 冻结在归一化标签先验（均值预测"送分项"提前给）──
        self.task_type = config.task_type
        self.output_head = nn.Sequential(
            RMSNorm(self.d_llm, learnable=True),
            nn.Linear(self.d_llm, config.output_dim),
            nn.Dropout(config.dropout),
        )
        if config.freeze_head_bias:
            prior = 0.5 if self.task_type == 'regression' else 0.0
            with torch.no_grad():
                self.output_head[1].bias.fill_(prior)
            self.output_head[1].bias.requires_grad_(False)

        # ── 知识 bypass 头：mean_pool(q) → 线性头，直连监督知识路径 ──
        self.bypass_head = nn.Linear(config.d_model, config.output_dim)
        self.bypass_gate = nn.Parameter(
            torch.tensor(config.bypass_gate_init))

        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()  # 冻结 backbone 置 eval：避免其 dropout 引入噪声

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, mode=True):
        """训练循环每轮会调 model.train()——冻结 LLM 必须保持 eval，
        避免其 dropout 引入噪声。"""
        super().train(mode)
        if hasattr(self, 'llm'):
            self.llm.eval()
        return self

    def _layer_weights(self):
        """α = 固定均匀(1/K) 或 β·e_last + (1-β)·softmax(a/T)（learn 时）。"""
        K = self.config.readout_layers
        if self.layer_logits is None:
            return torch.full((K,), 1.0 / K,
                              device=self.readout_embedding.device,
                              dtype=torch.float32)
        a = self.layer_logits
        w = torch.softmax(a / _LAYER_TEMP, dim=0)
        anchor = torch.zeros(K, device=a.device, dtype=a.dtype)
        anchor[-1] = 1.0
        return _LAYER_BETA * anchor + (1 - _LAYER_BETA) * w

    def _local_stats(self, x_z):
        """整分钟 z-scored EEG → (B*nseg, 32) 局部 display stats。

        6 个 10s 段：每段静态 FC + torch 版频谱/形态统计量，
        经 compute_graph_evidence 统一缩放裁剪（与全局同 schema）。
        """
        cfg = self.config
        seg_len = 10 * cfg.hz
        nseg = min(cfg.num_local_segments, x_z.shape[-1] // seg_len)
        xl = x_z[:, :, :nseg * seg_len]
        xl = xl.view(xl.shape[0], xl.shape[1], nseg, seg_len)
        xl = xl.permute(0, 2, 1, 3).contiguous().view(-1, xl.shape[1], seg_len)
        fc_l = torch.bmm(xl, xl.transpose(1, 2)) / (seg_len - 1)
        spec_l = segment_spectral_stats(xl, cfg.hz)
        morph_l = segment_morph_stats(xl, cfg.hz)
        _, stats_l = compute_graph_evidence(
            fc_l.unsqueeze(1), spec_l, morph_l)
        stats_l = torch.nan_to_num(stats_l, nan=0.0)
        return stats_l, nseg, xl                     # (B*nseg, 32)

    def forward(self, x, labels, spectral=None, morph=None):
        B, N, T = x.shape
        device = x.device

        # ── 逐通道 z-score（幅度不变性：跨域零样本迁移的关键）──
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_z = (x - mean) / (std + 1e-5)
        x_z = x_z.to(torch.float32)

        # ── 全局知识：整分钟静态 FC + 数据集层频谱/形态 → (B, 32) ──
        if spectral is None or morph is None:
            raise ValueError(
                "TimeLLMKnowledge requires spectral and morph stats; "
                "make sure the dataset returns them")
        static_fc = torch.bmm(x_z, x_z.transpose(1, 2)) / (T - 1)
        _, stats_g = compute_graph_evidence(
            static_fc.unsqueeze(1), spectral, morph)            # (B, 32)
        stats_g = torch.nan_to_num(stats_g, nan=0.0)

        # ── 局部知识：6 个 10s 段 → (B, nseg*32) ──
        nseg = 0
        if self.config.use_local_knowledge:
            stats_l, nseg, xl = self._local_stats(x_z)          # (B*nseg, 32)

        # ── 知识编译：q = gated(数值分支 + raw 分支) ──
        raw_g = self.patch_stem(x_z)                            # (B, d_feat)
        q_g = self.knowledge_compiler(stats_g, raw_g)           # (B, 32, d)
        if nseg > 0:
            raw_l = self.patch_stem(xl)                         # (B*nseg, d_feat)
            q_l = self.knowledge_compiler(stats_l, raw_l)       # (B*nseg, 32, d)

        # ── reprogram → soft prompt (4096) ──
        word_embeddings = self.word_embeddings.permute(1, 0).to(
            dtype=self.mapping_layer.weight.dtype)
        source_embeddings = self.mapping_layer(
            word_embeddings).permute(1, 0)                      # (M, 4096)
        reprogram_dtype = \
            self.reprogramming_layer.query_projection.weight.dtype
        soft_g = self.reprogramming_layer(
            q_g.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype))        # (B, 32, 4096)
        knowledge = soft_g
        q_all = q_g
        if nseg > 0:
            soft_l = self.reprogramming_layer(
                q_l.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype)
            ).view(B, nseg * self.config.num_stats, self.d_llm)
            knowledge = torch.cat([soft_g, soft_l], dim=1)      # (B, S_k, 4096)
            q_all = torch.cat(
                [q_g, q_l.view(B, nseg * self.config.num_stats, self.d_model)],
                dim=1)

        # ── phrase 融合：k = ref_rms·RMSNorm(soft + β·phrase)，phrase 冻结 ──
        phrase_g = self.knowledge_compiler.phrase_bank.unsqueeze(0).expand(
            B, -1, -1)
        phrase = phrase_g
        if nseg > 0:
            phrase_l = self.knowledge_compiler.phrase_bank.unsqueeze(
                0).unsqueeze(0).expand(
                    B, nseg, -1, -1).reshape(
                        B, nseg * self.config.num_stats, self.d_llm)
            phrase = torch.cat([phrase_g, phrase_l], dim=1)
        knowledge = self.ref_rms * rms_norm(
            knowledge + self.phrase_beta * phrase)
        S_k = knowledge.shape[1]

        # ── 组装序列：[start][task][end][knowledge][READOUT(固定)] ──
        start_tag = self.start_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        end_tag = self.end_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        task_prompt = self.task_prompt_embeddings.unsqueeze(0).expand(
            B, -1, -1)
        readout = self.readout_embedding.unsqueeze(0).unsqueeze(0).expand(
            B, 1, -1).to(dtype=torch.bfloat16)

        prompt_parts = [start_tag]
        prompt_length = self.P_start
        if self.config.use_task_prompt:
            prompt_parts.append(task_prompt)
            prompt_length += self.P_task
        prompt_parts.append(end_tag)
        prompt_length += self.P_end
        prompt_parts.append(knowledge.to(dtype=torch.bfloat16))
        prompt_parts.append(readout)

        inputs_embeds = torch.cat(prompt_parts, dim=1)
        sequence_length = inputs_embeds.shape[1]
        readout_idx = prompt_length + S_k                      # = S - 1

        if self.llm_type == 'chatglm':
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length, sequence_length,
                    dtype=torch.bool, device=device),
                diagonal=1,
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(
                B, 1, -1, -1)
            global_pos = torch.arange(
                sequence_length, dtype=torch.long, device=device
            ).unsqueeze(0).repeat(B, 1)
            if self.position_encoding_2d:
                block_pos = torch.cat([
                    torch.arange(
                        prompt_length, dtype=torch.long, device=device),
                    torch.arange(
                        S_k + 1, dtype=torch.long, device=device),
                ]).unsqueeze(0).repeat(B, 1)
                position_ids = torch.stack((global_pos, block_pos), dim=1)
            else:
                position_ids = global_pos

            outputs = self._transformer(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            position_ids = torch.arange(
                sequence_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(B, -1)
            outputs = self._transformer(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        all_hidden = outputs.hidden_states   # h0(embedding 输出) + 32 层

        # ── 最后 K 层 READOUT 位置聚合读出 ──
        K = self.config.readout_layers
        hs = torch.stack(
            [h[:, readout_idx, :].float() for h in all_hidden[-K:]], dim=1)
        # 逐层固定 RMS 归一化（无参数，防各层范数差异主导聚合）
        hs = hs / (hs.pow(2).mean(-1, keepdim=True).sqrt().add(_EPS))
        alpha = self._layer_weights()                           # (K,)
        agg = (hs * alpha.view(1, -1, 1)).sum(1)                # (B, 4096)

        logits = self.output_head(
            agg.to(dtype=self.output_head[1].weight.dtype))

        # ── 知识 bypass：mean_pool(q) → 线性头，λ 加权 ──
        if self.config.use_knowledge_bypass:
            z = q_all.mean(dim=1)                               # (B, d_model)
            y_bypass = self.bypass_head(z)
            logits = logits + self.bypass_gate * y_bypass

        if self.task_type == 'regression':
            prediction = logits.squeeze(-1)
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            loss = F.huber_loss(
                prediction.float(), labels.float(), delta=1.0)
        else:
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels.argmax(dim=-1)
            loss = F.cross_entropy(logits.float(), labels)

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'q': q_all,
                'knowledge': knowledge,
                'readout_agg': agg,
                'layer_weights': alpha,
                'bypass_gate': self.bypass_gate,
                'num_gate': self.knowledge_compiler.num_gate,
                'raw_gate': self.knowledge_compiler.raw_gate,
                'phrase_beta': self.phrase_beta,
            },
        )
