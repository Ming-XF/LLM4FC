"""TimeLLMKnowledge v2: 全局+局部 raw 知识 Token 编译器。

输入三段：
  离线整分钟静态 FC / 相对 PSD + 在线波形编码 → 32 个全局 raw tokens
  离线 6 个 10s FC / 相对 PSD + 在线波形编码 → 每片段 32 个局部 raw tokens
  任务 prompt（不变）

编码（全局块与局部块共享编译器参数）：
  q_i = g_raw·RMSNorm(patch_i)                                     (128)
  patch_i：FC 组=FC 图 2D CNN、频谱组=相对 PSD 2D CNN、
           形态组=PatchStem 投影（--no-use_patch_embeds 回退旧共享路径）
  q → reprogram（跨注意力到语义原型 bank，v3）→ soft prompt (4096)
  k_i = ref_rms·RMSNorm(soft_prompt_i + β·phrase_i)   ← phrase 只作身份锚点

语义原型（v3，替代全词表学习混合 mapping_layer）：
  每短语在冻结词表里取余弦 top-K 近邻候选，共享 Linear 压缩到 P 个原型，
  第 i 个知识 token 只在其短语的 P 个原型上做注意力（确定性、域无关预选；
  见 semantic_prototypes.py）。--no-use_semantic_prototypes 回退旧路径。

序列：[全局 32][局部 6×32][任务 prompt(末位 ":" 为读出位置)] → 冻结 LLM
→ 末 K 层 prompt 末位位置聚合 → 4096→128 隐层（知识特征在此直接融合）→ GELU → Dropout → 输出投影（bias 固定源域先验）。

捷径解除：
  - 读出位置为任务 prompt 末位 token（冻结词表向量，不可学习；
    无 prompt 时退化为固定 ":" token 缓冲）；
  - 层聚合权重固定均匀；
  - 输出头 bias 冻结在归一化标签先验（回归 0.5 / 分类 0）；
  - fusion：z = 全局/局部平衡池化(q) → 128 维隐层直接相加（fused = h1 + z），
    经 GELU 后走共享输出投影，给知识路径一条不经过冻结 LLM 的特征直连。
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..base import BaseConfig, ModelOutputs
from .knowledge_compiler import KnowledgeCompiler, rms_norm
from .patch_embeds import FCEmbed, SpecEmbed
from .prompts import get_prompt_config
from .semantic_prototypes import SemanticPrototypeBank

_EPS = 1e-8


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
        # target (B, L, d_model)
        # source/value：共享 bank 时 (S, d_llm)；语义原型时逐 token bank (L, S, d_llm)
        B, L, _ = target_embedding.shape
        H = self.n_heads
        E = target_embedding.shape[-1] // H

        q = self.query_projection(target_embedding).view(B, L, H, E)
        k = self.key_projection(source_embedding)
        if k.dim() == 2:                                   # (S, H*E) 共享 bank
            k = k.view(-1, H, E)
            scores = torch.einsum("blhe,she->bhls", q, k) * (1.0 / sqrt(E))
            attn = self.dropout(torch.softmax(scores, dim=-1))  # (B,H,L,S)
            out = torch.einsum("bhls,sd->blhd", attn, value_embedding)
        else:                                              # (L, S, H*E) 逐 token bank
            k = k.view(L, -1, H, E)
            scores = torch.einsum("blhe,lshe->bhls", q, k) * (1.0 / sqrt(E))
            attn = self.dropout(torch.softmax(scores, dim=-1))  # (B,H,L,S)
            out = torch.einsum("bhls,lsd->blhd", attn, value_embedding)
        return out.mean(dim=2)                             # (B,L,d_llm)


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
                 use_knowledge_fusion=True,
                 freeze_head_bias=True,
                 phrase_beta_init=0.25,
                 fusion_alpha_max=0.5,
                 fusion_gate_init=-1.39,
                 use_semantic_prototypes=True,
                 proto_preselect=1000,
                 proto_per_phrase=100,
                 use_patch_embeds=True,
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
        self.use_knowledge_fusion = use_knowledge_fusion
        self.freeze_head_bias = freeze_head_bias
        self.phrase_beta_init = phrase_beta_init
        self.fusion_alpha_max = fusion_alpha_max
        self.fusion_gate_init = fusion_gate_init
        self.use_semantic_prototypes = use_semantic_prototypes
        self.proto_preselect = proto_preselect
        self.proto_per_phrase = proto_per_phrase
        self.use_patch_embeds = use_patch_embeds
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

        # ── 任务 prompt 缓冲（冻结词表向量，末位 ":" 即读出位置）──
        prompt_config = get_prompt_config(config.dataset_name)
        task_ids = self.tokenizer.encode(prompt_config.prompt_task)
        with torch.no_grad():
            task_embeds = self._word_embeddings(torch.tensor(task_ids))
        self.register_buffer("task_prompt_embeddings", task_embeds)
        self.P_task = task_embeds.shape[0]

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
        )

        # ── raw 分支：分组 patch 嵌入（FC 图 / 相对 PSD / 形态 PatchStem）──
        # schema 分组：FC 9 | 频谱 13 | 形态时域 10（与 graph_evidence 输出一致）
        self.num_fc_stats = 9
        self.num_spec_stats = 13
        self.num_morph_stats = 10
        assert (self.num_fc_stats + self.num_spec_stats
                + self.num_morph_stats) == config.num_stats
        self.use_patch_embeds = config.use_patch_embeds
        if self.use_patch_embeds:
            self.fc_embed = FCEmbed(
                out_tokens=self.num_fc_stats, d_model=config.d_model)
            self.spec_embed = SpecEmbed(
                out_tokens=self.num_spec_stats, d_model=config.d_model)
            self.morph_head = nn.Linear(
                config.d_feat, self.num_morph_stats * config.d_model)
            self.raw_projection = None
        else:
            # 旧路径（消融对照）：共享 PatchStem → 全 32 统计量投影
            self.fc_embed = None
            self.spec_embed = None
            self.morph_head = None
            self.raw_projection = nn.Linear(
                config.d_feat, config.num_stats * config.d_model)

        # ── 语义原型 bank（替代 mapping_layer）或旧全词表学习混合（消融开关）──
        self.use_semantic_prototypes = config.use_semantic_prototypes
        if self.use_semantic_prototypes:
            self.semantic_bank = SemanticPrototypeBank(
                phrase_bank=self.knowledge_compiler.phrase_bank,
                word_embeddings=self.word_embeddings,
                tokenizer=self.tokenizer,
                preselect=config.proto_preselect,
                per_phrase=config.proto_per_phrase,
            )
            self.mapping_layer = None
        else:
            self.semantic_bank = None
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

        # ── 无 prompt 兜底读出 token："：" 词表向量（不可学习，焊死捷径）──
        colon_ids = self.tokenizer.encode(":", add_special_tokens=False)
        if not colon_ids:
            colon_ids = [self.tokenizer.eos_token_id]
        with torch.no_grad():
            rv = self._word_embeddings(
                torch.tensor(colon_ids)).mean(0).float()
        rms = rv.pow(2).mean().sqrt().clamp_min(_EPS)
        self.register_buffer("readout_token", rv / rms * ref_rms)

        # ── 输出头：4096→128→output_dim，知识特征在 128 维隐层直接融合 ──
        self.task_type = config.task_type
        self.output_head = nn.Sequential(
            RMSNorm(self.d_llm, learnable=True),
            nn.Linear(self.d_llm, config.d_model),
        )
        self.output_proj = nn.Linear(config.d_model, config.output_dim)
        self.output_dropout = nn.Dropout(config.dropout)
        # 有界标量门控：α = α_max·σ(g)，初始 α≈0.1（与旧 bypass_gate init 同义）
        self.fusion_gate = nn.Parameter(torch.tensor(config.fusion_gate_init))
        if config.freeze_head_bias:
            # bias 冻结在归一化标签先验（均值预测"送分项"提前给）
            prior = 0.5 if self.task_type == 'regression' else 0.0
            with torch.no_grad():
                self.output_proj.bias.fill_(prior)
            self.output_proj.bias.requires_grad_(False)

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

    def _group_raw_tokens(self, x_z, static_fc, relative_psd):
        """组装 (B, 32, d_model) 分组 raw token：
        FC 图 2D CNN → 9；相对 PSD 2D CNN → 13；形态 PatchStem 投影 → 10。"""
        if self.use_patch_embeds:
            fc_t = self.fc_embed(static_fc.unsqueeze(1))        # (B, 9, d)
            spec_t = self.spec_embed(relative_psd.unsqueeze(1)) # (B, 13, d)
            morph_t = self.morph_head(self.patch_stem(x_z)).view(
                x_z.shape[0], self.num_morph_stats, self.d_model)
            return torch.cat([fc_t, spec_t, morph_t], dim=1)
        # 旧路径（消融对照）：共享 PatchStem → 全 32 统计量投影
        return self.raw_projection(self.patch_stem(x_z)).view(
            x_z.shape[0], self.config.num_stats, self.d_model)

    def forward(self, x, global_fc, global_psd,
                local_fc, local_psd, local_mask, labels):
        B, N, T = x.shape
        device = x.device

        if global_fc.shape != (B, N, N):
            raise ValueError(
                f"global_fc must have shape {(B, N, N)}, got {tuple(global_fc.shape)}")
        if global_psd.ndim != 3 or global_psd.shape[:2] != (B, N):
            raise ValueError(
                f"global_psd must have shape (B, N, F), got {tuple(global_psd.shape)}")

        # ── 逐通道 z-score（幅度不变性：跨域零样本迁移的关键）──
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_z = (x - mean) / (std + 1e-5)
        x_z = x_z.to(torch.float32)

        # ── 局部波形：与离线 sidecar 一致，固定切成 6 个 10s 段 ──
        nseg = 0
        if self.config.use_local_knowledge:
            seg_len = 10 * self.config.hz
            nseg = self.config.num_local_segments
            expected_local_fc = (B, nseg, N, N)
            if local_fc.shape != expected_local_fc:
                raise ValueError(
                    f"local_fc must have shape {expected_local_fc}, "
                    f"got {tuple(local_fc.shape)}")
            if (local_psd.ndim != 4 or
                    local_psd.shape[:3] != (B, nseg, N)):
                raise ValueError(
                    f"local_psd must have shape (B, {nseg}, N, F), "
                    f"got {tuple(local_psd.shape)}")
            if local_psd.shape[-1] != global_psd.shape[-1]:
                raise ValueError("Global and local PSD frequency dimensions differ")
            if local_mask.shape != (B, nseg):
                raise ValueError(
                    f"local_mask must have shape {(B, nseg)}, "
                    f"got {tuple(local_mask.shape)}")
            if not bool(local_mask.all()):
                raise ValueError(
                    "Current one-minute TimeLLMKnowledge input requires all "
                    "six local segments to be valid")
            required_length = nseg * seg_len
            if T < required_length:
                raise ValueError(
                    f"Raw waveform has {T} samples, but {required_length} are "
                    "required to align with offline local features")
            xl = x_z[:, :, :required_length]
            xl = xl.view(B, N, nseg, seg_len)
            xl = xl.permute(0, 2, 1, 3).contiguous().view(
                B * nseg, N, seg_len)

        # ── raw patch 编译为知识 query ──
        group_raw_g = self._group_raw_tokens(
            x_z, global_fc.to(torch.float32), global_psd.to(torch.float32))
        q_g = self.knowledge_compiler(group_raw_g)               # (B, 32, d)
        if nseg > 0:
            fc_l = local_fc.reshape(B * nseg, N, N).to(torch.float32)
            psd_l = local_psd.reshape(
                B * nseg, N, local_psd.shape[-1]).to(torch.float32)
            group_raw_l = self._group_raw_tokens(xl, fc_l, psd_l)
            q_l = self.knowledge_compiler(group_raw_l)

        # ── reprogram → soft prompt (4096) ──
        reprogram_dtype = \
            self.reprogramming_layer.query_projection.weight.dtype
        if self.use_semantic_prototypes:
            # 语义原型：每知识 token 只在其短语的 P 个原型上做注意力
            phrase_ids_g = torch.arange(self.config.num_stats, device=device)
            banks_g = self.semantic_bank(phrase_ids_g)          # (32, P, 4096)
            soft_g = self.reprogramming_layer(
                q_g.to(dtype=reprogram_dtype),
                banks_g.to(dtype=reprogram_dtype),
                banks_g.to(dtype=reprogram_dtype))              # (B, 32, 4096)
        else:
            # 旧路径（消融基线）：全词表学习混合 mapping_layer
            word_embeddings = self.word_embeddings.permute(1, 0).to(
                dtype=self.mapping_layer.weight.dtype)
            source_embeddings = self.mapping_layer(
                word_embeddings).permute(1, 0)                  # (M, 4096)
            soft_g = self.reprogramming_layer(
                q_g.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype))    # (B, 32, 4096)
        knowledge = soft_g
        q_all = q_g
        if nseg > 0:
            if self.use_semantic_prototypes:
                # 每个 10s 段内统计量 schema 与全局相同 → 复用同一份 bank
                banks_l = banks_g
                soft_l = self.reprogramming_layer(
                    q_l.to(dtype=reprogram_dtype),
                    banks_l.to(dtype=reprogram_dtype),
                    banks_l.to(dtype=reprogram_dtype)
                ).view(B, nseg * self.config.num_stats, self.d_llm)
            else:
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

        # ── 组装序列：[knowledge][task prompt(末位 ":" 为读出位置)] ──
        # 无 prompt 时退化为 [knowledge][":" 兜底 token]，读出位置恒为 S-1
        prompt_parts = [knowledge.to(dtype=torch.bfloat16)]
        if self.config.use_task_prompt:
            task_prompt = self.task_prompt_embeddings.unsqueeze(0).expand(
                B, -1, -1)
            prompt_parts.append(task_prompt)
        else:
            prompt_parts.append(
                self.readout_token.unsqueeze(0).expand(
                    B, -1, -1).to(dtype=torch.bfloat16))

        inputs_embeds = torch.cat(prompt_parts, dim=1)
        sequence_length = inputs_embeds.shape[1]
        readout_idx = sequence_length - 1                      # 末位 token

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
                block_pos = torch.arange(
                    sequence_length, dtype=torch.long, device=device
                ).unsqueeze(0).repeat(B, 1)
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
        agg = hs.mean(dim=1)                                    # (B, 4096)

        h1 = self.output_head(
            agg.to(dtype=self.output_head[1].weight.dtype))     # (B, d_model)

        # ── 知识特征融合：全局/局部平衡池化 → 128 维隐层直接相加 ──
        if self.config.use_knowledge_fusion:
            z_g = q_all[:, :self.config.num_stats].mean(dim=1)
            z_l = (q_all[:, self.config.num_stats:].mean(dim=1)
                   if q_all.shape[1] > self.config.num_stats else 0.0)
            z = rms_norm(0.5 * (z_g + z_l))                     # (B, d_model)
            alpha = self.config.fusion_alpha_max * torch.sigmoid(
                self.fusion_gate)                               # 标量，∈[0, α_max]
            fused = h1 + alpha * z
        else:
            alpha = None
            fused = h1

        logits = self.output_proj(self.output_dropout(F.gelu(fused)))

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
                'fusion_gate': alpha,
                'raw_gate': self.knowledge_compiler.raw_gate,
                'phrase_beta': self.phrase_beta,
            },
        )
