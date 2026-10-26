"""TimeLLMKnowledge v3: three-family, two-level semantic knowledge bank.

输入：
  统一滑动窗口（窗口数/窗长/步长全部来自 sidecar 元数据，
  经 data_config 传入；例：11×10s 窗、步长 5s）：
  离线静态 FC / 相对 PSD + 在线波形编码 → 每窗 3 个 family queries
  任务 prompt（不变）

编码（所有窗口共享编译器参数）：
  q_i = g_raw·RMSNorm(patch_i)                                     (128)
  patch_i：FC 组=FC 图 2D CNN、频谱组=相对 PSD 2D CNN、
           形态组=PatchStem 投影（--no-use_patch_embeds 回退旧共享路径）
  q → reprogram（跨注意力到语义原型 bank，v3）→ soft prompt (4096)
  k_i = ref_rms·RMSNorm(soft_prompt_i + β·phrase_i)   ← phrase 只作身份锚点

语义原型（两级候选库）：
  三个方面文本在冻结词表中预选 K 个候选，经共享 Linear(K→N) 融合为 N 个
  refined knowledge；每个 refined knowledge 再预选 M 个描述，经共享
  Linear(M→P) 得到自己的描述原型库。--anchor_select api 时第一级候选
  改为外部 DeepSeek API 生成的关键词（缓存文件优先），每个关键词短语
  以 subword token 平均池化为一行（一短语一知识行）；api_mix 再拼接
  冻结词表余弦 top-(anchor_candidates−K_api) 行成混合池。
  --no-use_semantic_prototypes 回退旧路径。

序列：[N_win×3N 滑动窗口知识][任务 prompt(末位 ":" 为读出位置)] → 冻结 LLM
→ 末层 prompt 末位位置 hidden state（无聚合/归一化）→ 4096→256 隐层
→ GELU → Dropout → 输出投影（bias 固定源域先验）。

捷径解除：
  - 读出位置为任务 prompt 末位 token（冻结词表向量，不可学习；
    无 prompt 时退化为固定 ":" token 缓冲）；
  - 读出只取末层 hidden state（无层聚合/归一化）；
  - 输出头 bias 冻结在归一化标签先验（回归 0.5 / 分类 0）；
  - 无特征融合直连（fusion/bypass 路径已整体移除）。

监督对比学习（三族 token，族输出投影之后）：
  三族编码器（FC/频谱/形态）输出的族 token（每窗每族 1 个，d_model=128 维）
  按族做逐窗 RMSNorm 后跨窗口池化为 3 个 d_model 维向量，L2 归一化后
  按族独立计算监督 InfoNCE（同标签互为正对，无温度），
  loss = CE + contrast_lambda·NCE（回归任务跳过）。
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..base import BaseConfig, ModelOutputs
from .knowledge_compiler import KnowledgeCompiler, rms_norm, _PHRASES
from .patch_embeds import FCEmbed, SpecEmbed
from .prompts import get_prompt_config
from .semantic_prototypes import (
    SemanticPrototypeBank, api_keyword_candidate_embeddings,
    api_keyword_pool_size)

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
                 num_aspects=3,
                 d_feat=256,
                 hz=200,
                 num_knowledge_windows=7,
                 window_seconds=30,
                 stride_seconds=5,
                 contrast_lambda=0.1,
                 freeze_head_bias=True,
                 phrase_beta_init=0.25,
                 use_semantic_prototypes=True,
                 anchor_select='cosine',
                 gen_num_keywords=100,
                 gen_log_path=None,
                 generate_at_init=True,
                 anchor_candidates=1000,
                 num_anchors=10,
                 description_candidates=1000,
                 description_prototypes=100,
                 semantic_similarity='centered_cosine',
                 semantic_aspect_mix=0.6,
                 semantic_anchor_fusion=0.1,
                 use_patch_embeds=True,
                 task_type='classification',
                 output_dim=2,
                 dataset_name='CAUEEG',
                 llm_type='chatglm',
                 llm_path='./model/chatglm-6b',
                 use_task_prompt=False,
                 llm_init='pretrained'):
        super().__init__(node_size=node_size, output_dim=output_dim)
        self.time_series_size = time_series_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_prototypes = num_prototypes
        self.dropout = dropout
        self.num_aspects = num_aspects
        self.num_stats = num_aspects
        self.d_feat = d_feat
        self.hz = hz
        self.num_knowledge_windows = num_knowledge_windows
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.contrast_lambda = contrast_lambda
        self.freeze_head_bias = freeze_head_bias
        self.phrase_beta_init = phrase_beta_init
        self.use_semantic_prototypes = use_semantic_prototypes
        self.anchor_select = anchor_select
        self.gen_num_keywords = gen_num_keywords
        self.gen_log_path = gen_log_path
        self.generate_at_init = generate_at_init
        self.anchor_candidates = anchor_candidates
        self.num_anchors = num_anchors
        self.description_candidates = description_candidates
        self.description_prototypes = description_prototypes
        self.semantic_similarity = semantic_similarity
        self.semantic_aspect_mix = semantic_aspect_mix
        self.semantic_anchor_fusion = semantic_anchor_fusion
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

        # ── 第一级候选来源：余弦检索（默认）、外部 API 关键词（缓存优先）、
        # 或两者混合池 api_mix（API 短语行 + 余弦 top-(K-K_api) 行）──
        anchor_embeds = None
        anchor_candidates = config.anchor_candidates
        if config.anchor_select in ('api', 'api_mix'):
            mix_cosine = (config.anchor_select == 'api_mix')
            if config.generate_at_init:
                anchor_embeds, k_api = api_keyword_candidate_embeddings(
                    self._word_embeddings, self.tokenizer, _PHRASES,
                    num_keywords=config.gen_num_keywords,
                    gen_log_path=config.gen_log_path)
                if not mix_cosine:
                    anchor_candidates = k_api
            else:
                # 评估端跳过生成：K_api = 每方面关键词数（只读缓存不调 API），
                # 占位嵌入只保证形状（混合模式由 bank 补余弦行，最终 K 仍为
                # anchor_candidates），内容由 load_model 用 checkpoint 中的
                # anchor_candidate_embeddings buffer 覆盖
                d_llm = self._word_embeddings.weight.shape[1]
                k_api = api_keyword_pool_size(
                    _PHRASES, config.gen_num_keywords)
                if not mix_cosine:
                    anchor_candidates = k_api
                anchor_embeds = torch.zeros(
                    config.num_aspects, k_api, d_llm,
                    dtype=self._word_embeddings.weight.dtype)

        # ── raw 分支：三个方面各生成一个 d_model query ──
        self.num_aspects = config.num_aspects
        if self.num_aspects != 3:
            raise ValueError("TimeLLMKnowledge 当前固定使用 FC/频谱/形态三个方面")
        self.use_patch_embeds = config.use_patch_embeds
        if self.use_patch_embeds:
            self.fc_embed = FCEmbed(d_model=config.d_model)
            self.spec_embed = SpecEmbed(d_model=config.d_model)
            self.morph_head = nn.Linear(config.d_feat, config.d_model)
            self.raw_projection = None
        else:
            # 旧路径（消融对照）：共享 PatchStem → 三方面投影
            self.fc_embed = None
            self.spec_embed = None
            self.morph_head = None
            self.raw_projection = nn.Linear(
                config.d_feat, self.num_aspects * config.d_model)

        # ── 语义原型 bank（替代 mapping_layer）或旧全词表学习混合（消融开关）──
        self.use_semantic_prototypes = config.use_semantic_prototypes
        if self.use_semantic_prototypes:
            self.semantic_bank = SemanticPrototypeBank(
                phrase_bank=self.knowledge_compiler.phrase_bank,
                word_embeddings=self.word_embeddings,
                tokenizer=self.tokenizer,
                anchor_candidates=anchor_candidates,
                num_anchors=config.num_anchors,
                description_candidates=config.description_candidates,
                num_prototypes=config.description_prototypes,
                similarity=config.semantic_similarity,
                aspect_mix=config.semantic_aspect_mix,
                anchor_fusion_init=config.semantic_anchor_fusion,
                anchor_candidate_embeddings=anchor_embeds,
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

        # ── 输出头：4096→256→output_dim（无特征融合直连）──
        self.task_type = config.task_type
        self.output_head = nn.Sequential(
            RMSNorm(self.d_llm, learnable=True),
            nn.Linear(self.d_llm, 256),
        )
        self.output_proj = nn.Linear(256, config.output_dim)
        self.output_dropout = nn.Dropout(config.dropout)
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
        """组装 (B, 3, d_model) 方面 query。

        三族编码器输出族 token（族输出投影之后）；旧路径（消融对照）
        共享 PatchStem → 三方面投影。"""
        if self.use_patch_embeds:
            fc_t = self.fc_embed(static_fc.unsqueeze(1))        # (B, d)
            spec_t = self.spec_embed(relative_psd)              # (B, d)
            morph_t = self.morph_head(self.patch_stem(x_z))     # (B, d)
            return torch.stack([fc_t, spec_t, morph_t], dim=1)
        # 旧路径（消融对照）：共享 PatchStem → 三方面投影
        return self.raw_projection(self.patch_stem(x_z)).view(
            x_z.shape[0], self.num_aspects, self.d_model)

    def _contrastive_loss(self, v, labels):
        """监督对比损失：三族 token（族输出投影之后）逐族独立 InfoNCE。

        v: (B, 3, d_model) 三族（FC/频谱/形态）token，labels: (B,) 类索引。
        同标签互为正对、自身排除，无温度（logits = 归一化内积）。
        无正对的行跳过；各族损失取平均；整批无正对时返回 None。
        """
        p = F.normalize(v, dim=-1)                              # (B, 3, d_model)
        B = p.shape[0]
        eye = torch.eye(B, device=p.device, dtype=torch.bool)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
        total = 0.0
        count = 0
        for i in range(p.shape[1]):
            sim = p[:, i] @ p[:, i].T                          # (B, B)
            has_pos = pos_mask.any(dim=1)
            if not bool(has_pos.any()):
                continue
            numer = torch.logsumexp(
                sim.masked_fill(~pos_mask, float('-inf')), dim=1)
            denom = torch.logsumexp(
                sim.masked_fill(eye, float('-inf')), dim=1)
            row_loss = (denom - numer)[has_pos]
            total = total + row_loss.mean()
            count += 1
        if count == 0:
            return None
        return total / count

    def forward(self, x, window_fc, window_psd, window_mask, labels):
        B, N, T = x.shape
        device = x.device

        nwin = self.config.num_knowledge_windows
        win_len = int(self.config.window_seconds * self.config.hz)
        stride = int(self.config.stride_seconds * self.config.hz)
        if window_fc.shape != (B, nwin, N, N):
            raise ValueError(
                f"window_fc must have shape {(B, nwin, N, N)}, "
                f"got {tuple(window_fc.shape)}")
        if (window_psd.ndim != 4 or
                window_psd.shape[:3] != (B, nwin, N)):
            raise ValueError(
                f"window_psd must have shape (B, {nwin}, N, F), "
                f"got {tuple(window_psd.shape)}")
        if window_mask.shape != (B, nwin):
            raise ValueError(
                f"window_mask must have shape {(B, nwin)}, "
                f"got {tuple(window_mask.shape)}")
        if not bool(window_mask.all()):
            raise ValueError(
                "TimeLLMKnowledge requires all sliding windows to be valid")

        # ── 逐通道 z-score（幅度不变性：跨域零样本迁移的关键）──
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_z = (x - mean) / (std + 1e-5)
        x_z = x_z.to(torch.float32)

        # ── 统一滑动窗口：在线波形与离线 sidecar 对齐切窗 ──
        required_length = (nwin - 1) * stride + win_len
        if T < required_length:
            raise ValueError(
                f"Raw waveform has {T} samples, but {required_length} are "
                "required to align with offline sliding windows")
        xw = x_z[:, :, :required_length]

        # ── 每窗 raw patch 编译为知识 query ──
        q_wins = []
        tok_wins = []
        for k in range(nwin):
            s = k * stride
            win = xw[:, :, s:s + win_len]
            fc_k = window_fc[:, k].to(torch.float32)
            psd_k = window_psd[:, k].to(torch.float32)
            group_raw = self._group_raw_tokens(win, fc_k, psd_k)
            q_wins.append(self.knowledge_compiler(group_raw))   # (B, 3, d)
            if self.use_patch_embeds:
                tok_wins.append(group_raw)                      # (B, 3, d_model)
        q_all = torch.stack(q_wins, dim=1)                      # (B, nwin, 3, d)
        q_flat = q_all.reshape(B, nwin * self.num_aspects, self.d_model)
        q_all = q_flat                                          # 诊断输出用扁平形式

        # ── 对比学习特征（族 token，输出投影之后）：每窗每族 1 个 d_model 维向量 ──
        contrast_vec = None
        if self.use_patch_embeds and tok_wins:
            toks = torch.stack(tok_wins, dim=1)                 # (B, nwin, 3, d_model)
            toks = toks.permute(0, 2, 1, 3)                     # (B, 3, nwin, d_model)
            # 逐窗固定 RMSNorm 后族内均值（防窗间范数差主导）
            toks = toks / (toks.pow(2).mean(-1, keepdim=True).sqrt().add(_EPS))
            contrast_vec = toks.mean(dim=-2)                    # (B, 3, d_model)

        # ── reprogram → soft prompt (4096) ──
        reprogram_dtype = \
            self.reprogramming_layer.query_projection.weight.dtype
        num_anchors = self.config.num_anchors
        if self.use_semantic_prototypes:
            # Bank order is aspect-major: [FC x N, spectrum x N, morphology x N].
            banks_base = self.semantic_bank().reshape(
                self.num_aspects * num_anchors,
                self.config.description_prototypes,
                self.d_llm)
            # Every window reuses the same frozen semantic ids and
            # trainable candidate-space mappings.
            q_reprogram = q_flat.repeat_interleave(num_anchors, dim=1)
            banks_w = banks_base.repeat(nwin, 1, 1)
            knowledge = self.reprogramming_layer(
                q_reprogram.to(dtype=reprogram_dtype),
                banks_w.to(dtype=reprogram_dtype),
                banks_w.to(dtype=reprogram_dtype)
            ).view(B, nwin * self.num_aspects * num_anchors, self.d_llm)
        else:
            # 旧路径（消融基线）：全词表学习混合 mapping_layer
            word_embeddings = self.word_embeddings.permute(1, 0).to(
                dtype=self.mapping_layer.weight.dtype)
            source_embeddings = self.mapping_layer(
                word_embeddings).permute(1, 0)                  # (M, 4096)
            knowledge = self.reprogramming_layer(
                q_flat.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype)
            ).view(B, nwin * self.num_aspects, self.d_llm)

        # ── aspect phrase 融合：phrase 只作身份锚点 ──
        if self.use_semantic_prototypes:
            # One phrase identity is repeated for each refined anchor.
            phrase_base = self.knowledge_compiler.phrase_bank.repeat_interleave(
                num_anchors, dim=0)
        else:
            # The full-vocabulary ablation emits one token per aspect.
            phrase_base = self.knowledge_compiler.phrase_bank
        phrase_slots = phrase_base.shape[0]
        phrase = phrase_base.unsqueeze(0).unsqueeze(0).expand(
            B, nwin, -1, -1).reshape(
            B, nwin * phrase_slots, self.d_llm)
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

        # ── 末层 hidden state 读出（第 32 层，READOUT 位置，无聚合/归一化）──
        agg = all_hidden[-1][:, readout_idx, :].float()         # (B, 4096)

        h1 = self.output_head(
            agg.to(dtype=self.output_head[1].weight.dtype))     # (B, d_model)

        logits = self.output_proj(self.output_dropout(F.gelu(h1)))

        contrast_loss = torch.zeros((), device=logits.device)
        if self.task_type == 'regression':
            prediction = logits.squeeze(-1)
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            loss = F.huber_loss(
                prediction.float(), labels.float(), delta=1.0)
            # 回归任务跳过对比学习（同标签配对不适用）
        else:
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels.argmax(dim=-1)
            loss = F.cross_entropy(logits.float(), labels)
            if contrast_vec is not None and self.config.contrast_lambda > 0:
                c_loss = self._contrastive_loss(contrast_vec, labels)
                if c_loss is not None:
                    contrast_loss = c_loss
                    loss = loss + self.config.contrast_lambda * contrast_loss

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'q': q_all,
                'knowledge': knowledge,
                'readout_agg': agg,
                'contrast_loss': contrast_loss,
                'raw_gate': self.knowledge_compiler.raw_gate,
                'phrase_beta': self.phrase_beta,
            },
        )
