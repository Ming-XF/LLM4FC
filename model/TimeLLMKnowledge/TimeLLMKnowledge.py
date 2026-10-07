"""TimeLLMKnowledge: 领域知识 Token 编译器（纯知识版，无 patch 路径）。

整分钟 EEG → 静态 FC + 数据集层频谱/形态统计 → 32 个 display stats
→ 每统计量独立编译（短语 mean 向量与数值线性嵌入在 128 维求和）
→ 知识 soft prompt → 冻结 LLM → 最后 K 层 READOUT 位置聚合读出 → 线性头。

序列布局：[start_tag][task prompt][end_tag][knowledge 1..32][READOUT]
（tags 只包文本任务 prompt；READOUT 为可学习读出 token，是序列真正的
最后位置，其 hidden state 经多层聚合后进输出头。）
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..base import BaseConfig, ModelOutputs
from ..TimeLLM.graph_evidence import compute_graph_evidence
from .knowledge_compiler import KnowledgeCompiler
from .prompts import get_prompt_config

_EPS = 1e-8

# 层权重聚合：α = β·e_last + (1-β)·softmax(a/T)，最终层锚定防权重塌缩
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


class TimeLLMKnowledgeConfig(BaseConfig):
    def __init__(self, node_size,
                 time_series_size=12000,
                 d_model=128,
                 n_heads=8,
                 num_prototypes=1000,
                 dropout=0.1,
                 num_stats=32,
                 readout_layers=4,
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

        # ── 知识编译器 + reprogram + READOUT + 输出头 ──
        self.knowledge_compiler = KnowledgeCompiler(
            d_llm=self.d_llm,
            d_model=config.d_model,
            num_stats=config.num_stats,
            tokenizer=self.tokenizer,
            word_embeddings=self._word_embeddings,
            llm_type=self.llm_type,
        )
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_prototypes)
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
        )

        # READOUT：零均值随机初始化，RMS 对齐到词表 embedding 尺度
        ref_rms = self.word_embeddings.detach().float().pow(2).mean(
            -1).sqrt().median().clamp_min(_EPS)
        self.readout_embedding = nn.Parameter(torch.empty(self.d_llm))
        nn.init.normal_(self.readout_embedding, std=float(ref_rms))

        # 最后 K 层聚合权重（锚定 softmax，见 _layer_weights）
        self.layer_logits = nn.Parameter(
            torch.zeros(config.readout_layers))

        self.task_type = config.task_type
        self.output_head = nn.Sequential(
            RMSNorm(self.d_llm, learnable=True),
            nn.Linear(self.d_llm, config.output_dim),
            nn.Dropout(config.dropout),
        )

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
        """α = β·e_last + (1-β)·softmax(a/T)，最终层权重下限 β。"""
        K = self.config.readout_layers
        a = self.layer_logits
        w = torch.softmax(a / _LAYER_TEMP, dim=0)
        anchor = torch.zeros(K, device=a.device, dtype=a.dtype)
        anchor[-1] = 1.0
        return _LAYER_BETA * anchor + (1 - _LAYER_BETA) * w

    def forward(self, x, labels, spectral=None, morph=None):
        B, N, T = x.shape
        device = x.device

        # ── 逐通道 z-score（幅度不变性：跨域零样本迁移的关键）──
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + 1e-5)

        # ── 整分钟静态 FC → 32 维 display stats（fp32）──
        if spectral is None or morph is None:
            raise ValueError(
                "TimeLLMKnowledge requires spectral and morph stats; "
                "make sure the dataset returns them")
        x_z = x.to(torch.float32)
        static_fc = torch.bmm(x_z, x_z.transpose(1, 2)) / (T - 1)
        _, stats = compute_graph_evidence(
            static_fc.unsqueeze(1), spectral, morph)             # (B, 32)
        # 兜底：全平样本的 nan 统计量置 0（inf 已被 clamp 到 ±5），
        # 防止 nan 经 softmax 污染整个 batch 的梯度
        stats = torch.nan_to_num(stats, nan=0.0)

        # ── 知识编译 → soft prompt (B, 32, 4096) ──
        q = self.knowledge_compiler(stats)
        word_embeddings = self.word_embeddings.permute(1, 0).to(
            dtype=self.mapping_layer.weight.dtype)
        source_embeddings = self.mapping_layer(
            word_embeddings).permute(1, 0)                       # (M, 4096)
        reprogram_dtype = \
            self.reprogramming_layer.query_projection.weight.dtype
        knowledge = self.reprogramming_layer(
            q.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
        ).to(dtype=torch.bfloat16)

        # ── 组装序列：[start][task][end][knowledge 32][READOUT] ──
        start_tag = self.start_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        end_tag = self.end_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        task_prompt = self.task_prompt_embeddings.unsqueeze(0).expand(
            B, -1, -1)
        readout = self.readout_embedding.unsqueeze(0).unsqueeze(0).expand(
            B, 1, -1).to(dtype=knowledge.dtype)

        prompt_parts = [start_tag]
        prompt_length = self.P_start
        if self.config.use_task_prompt:
            prompt_parts.append(task_prompt)
            prompt_length += self.P_task
        prompt_parts.append(end_tag)
        prompt_length += self.P_end
        prompt_parts.append(knowledge)
        prompt_parts.append(readout)

        inputs_embeds = torch.cat(prompt_parts, dim=1)
        sequence_length = inputs_embeds.shape[1]
        readout_idx = prompt_length + self.config.num_stats  # = S - 1

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
                        self.config.num_stats + 1,
                        dtype=torch.long, device=device),
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
        alpha = self._layer_weights()                            # (K,)
        agg = (hs * alpha.view(1, -1, 1)).sum(1)                 # (B, 4096)

        logits = self.output_head(
            agg.to(dtype=self.output_head[1].weight.dtype))

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
                'q': q,
                'knowledge': knowledge,
                'readout_agg': agg,
                'layer_weights': alpha,
            },
        )
