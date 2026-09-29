from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..base import BaseConfig, ModelOutputs
from .graph_evidence import _EVIDENCE_TEMPLATES, compute_graph_evidence
from .prompts import NEURAL_SEMANTIC_PROTOTYPES, get_prompt_config


def rank_normalize_dfc(dfc):
    """E1: 对每个 DFC 窗口矩阵做边秩变换并映射到 [-1, 1]。

    分布无关（秩统计量），消除不同站点相关分布差异；
    秩对称化保证 (i,j) 与 (j,i) 一致，对角元固定为 1。
    """
    B, W, N, _ = dfc.shape
    flat = dfc.reshape(B, W, -1)                      # (B, W, N*N)
    ranks = flat.argsort(dim=-1).argsort(dim=-1).float()
    ranks = ranks.reshape(B, W, N, N)
    ranks = (ranks + ranks.transpose(-1, -2)) / 2     # 对称化
    ranks = ranks / (N * N - 1)                       # [0, 1]
    ranks = ranks * 2 - 1                             # [-1, 1]
    ranks[..., torch.arange(N), torch.arange(N)] = 1.0
    return ranks


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        x = self.dropout(x)
        support = self.linear(x)
        return torch.bmm(adj_norm, support)


class TimeLLMConfig(BaseConfig):
    def __init__(self, node_size,
                 d_model=64,
                 n_heads=8,
                 d_ff=128,
                 num_prototypes=500,
                 num_gcn_layers=1,
                 dropout=0.1,
                 num_windows=10,
                 task_type='classification',
                 output_dim=2,
                 dataset_name='CAUEEG',
                 llm_type='chatglm',
                 llm_path='./model/chatglm-6b',
                 use_task_prompt=False,
                 llm_init='pretrained',
                 readout='flatten',
                 prototype_mode='sparse',
                 age_head='linear',
                 age_bin_width=5,
                 emd_lambda=0.0,
                 align_lambda=0.0,
                 mv_consistency=False,
                 mv_lambda=0.0,
                 graph_evidence=False,
                 no_patches=False,
                 stat_vector=False,
                 stats_head=False,
                 residual_llm=False,
                 stats_ridge_path=''):
        super().__init__(node_size=node_size, output_dim=output_dim)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_prototypes = num_prototypes
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout
        self.num_windows = num_windows
        self.task_type = task_type
        self.output_dim = output_dim
        self.dataset_name = dataset_name
        self.llm_type = llm_type
        self.llm_path = llm_path
        self.llm_init = llm_init
        self.use_task_prompt = use_task_prompt
        self.readout = readout
        self.prototype_mode = prototype_mode
        self.age_head = age_head
        self.age_bin_width = age_bin_width
        self.emd_lambda = emd_lambda
        self.align_lambda = align_lambda
        self.mv_consistency = mv_consistency
        self.mv_lambda = mv_lambda
        self.graph_evidence = graph_evidence
        self.no_patches = no_patches
        self.stat_vector = stat_vector
        self.stats_head = stats_head
        self.residual_llm = residual_llm
        self.stats_ridge_path = stats_ridge_path


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None,
                 attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(
            B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self._reprogramming(
            target_embedding, source_embedding, value_embedding)
        return self.out_projection(out.reshape(B, L, -1))

    def _reprogramming(self, target_embedding, source_embedding,
                       value_embedding):
        _, _, _, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum(
            "blhe,she->bhls", target_embedding, source_embedding)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum(
            "bhls,she->blhe", attention, value_embedding)


class Model(nn.Module):
    def __init__(self, config: TimeLLMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_prototypes = config.num_prototypes
        self.n_heads = config.n_heads
        self.d_llm = 4096
        self.llm_type = config.llm_type
        self.llm_path = config.llm_path

        self.gcn_layers = nn.ModuleList([
            GCNLayer(self.d_model, self.d_model, dropout=config.dropout)
            for _ in range(config.num_gcn_layers)
        ])
        self.channel_embed_projection = nn.Linear(
            config.node_size, config.d_model)

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

        # Answer-query readout: a trailing <answer> token whose hidden state
        # summarizes all DFC patches. May span several sub-tokens; the readout
        # is the last one (it attends to the whole sequence under the causal
        # mask). Not created for the default flatten readout.
        if config.readout == 'answer_query':
            answer_tag_ids = self.tokenizer.encode("<answer>")
            with torch.no_grad():
                answer_tag_embeds = self._word_embeddings(
                    torch.tensor(answer_tag_ids))
            self.register_buffer("answer_tag_embeddings", answer_tag_embeds)
            self.P_answer = answer_tag_embeds.shape[0]
        else:
            self.P_answer = 0

        if config.readout == 'evidence_query' and not config.graph_evidence:
            raise ValueError(
                "--readout evidence_query requires --graph_evidence")
        if config.no_patches and config.readout != 'evidence_query':
            raise ValueError(
                "--no_patches requires --readout evidence_query")
        # E5: 图证据句模板（前缀/槽位词/后缀的 token id）。
        # 槽位词需为单 token，保证证据序列定长（无需 padding/掩码）。
        self.evidence_templates = None
        if config.graph_evidence:
            self.evidence_templates = []
            for prefix, words, suffix in _EVIDENCE_TEMPLATES:
                p_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                s_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
                w_ids = {}
                for w in words:
                    ids = self.tokenizer.encode(w, add_special_tokens=False)
                    assert len(ids) == 1, \
                        f"evidence word '{w}' is not a single token"
                    w_ids[w] = ids[0]
                self.evidence_templates.append((p_ids, w_ids, s_ids))

        self.word_embeddings = self._word_embeddings.weight
        self.vocab_size = self.word_embeddings.shape[0]

        if config.prototype_mode == 'semantic':
            # E2: 语义原型库。固定 LLM 编码的神经连接短语 embedding
            # 作为 cross-attention 键值，外加小残差提供灵活性。
            self.mapping_layer = None
            phrase_embeds = []
            with torch.no_grad():
                for phrase in NEURAL_SEMANTIC_PROTOTYPES:
                    ids = torch.tensor(self.tokenizer.encode(phrase))
                    emb = self._word_embeddings(ids).mean(dim=0)
                    phrase_embeds.append(emb)
            fixed = torch.stack(phrase_embeds, dim=0)          # (P, d_llm)
            self.register_buffer('semantic_prototypes', fixed)
            self.prototype_residual = nn.Parameter(
                torch.zeros(fixed.shape[0], fixed.shape[1]))
        else:
            self.mapping_layer = nn.Linear(
                self.vocab_size, self.num_prototypes)
            self.semantic_prototypes = None
            self.prototype_residual = None

        self.reprogramming_layer = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
        )

        self.task_type = config.task_type
        self.use_ordinal_age_head = (
            config.task_type == 'regression'
            and config.readout == 'answer_query'
            and config.age_head == 'ordinal')
        self.use_age_alignment = (
            config.task_type == 'regression'
            and config.readout == 'answer_query'
            and config.age_head == 'linear'
            and config.align_lambda > 0)
        if self.use_age_alignment:
            # E3v7: 线性头不变；另建原始 4096 维年龄 bin 短语原型，
            # 对齐损失把 h_answer 拉向标签 bin 的语言锚点。
            self.age_bin_w = config.age_bin_width / 100.0
            self.n_age_bins = int(round(1.0 / self.age_bin_w))
            bin_embeds = []
            with torch.no_grad():
                for k in range(self.n_age_bins):
                    lo = int(k * config.age_bin_width)
                    hi = int((k + 1) * config.age_bin_width)
                    phrase = (f"an EEG pattern associated with age "
                              f"{lo} to {hi}")
                    ids = torch.tensor(self.tokenizer.encode(phrase))
                    emb = self._word_embeddings(ids).mean(dim=0)
                    bin_embeds.append(F.normalize(emb, p=2, dim=-1))
            self.register_buffer(
                'raw_bin_prototypes', torch.stack(bin_embeds, dim=0))
        if self.use_ordinal_age_head:
            # E3: 有序年龄区间原型头。年龄区间描述的冻结 LLM embedding
            # 作为输出原型，h_answer 与各原型余弦匹配 → 区间分布 →
            # 期望年龄。输出空间锚定在语言语义上。
            bin_w = config.age_bin_width / 100.0             # 归一化年龄空间
            self.age_bin_w = bin_w
            self.n_age_bins = int(round(1.0 / bin_w))
            centers = (torch.arange(self.n_age_bins) + 0.5) * bin_w
            self.register_buffer(
                'age_bin_centers', centers.to(torch.bfloat16))
            # 数字主导的短短语：使数字 token 在 mean-pool 中占比更大，
            # 相邻 bin 的 embedding 才可区分
            bin_embeds = []
            with torch.no_grad():
                for k in range(self.n_age_bins):
                    lo = int(k * config.age_bin_width)
                    hi = int((k + 1) * config.age_bin_width)
                    phrase = f"age {lo} to {hi}"
                    ids = torch.tensor(self.tokenizer.encode(phrase))
                    emb = self._word_embeddings(ids).mean(dim=0)
                    bin_embeds.append(emb)
            E = torch.stack(bin_embeds, dim=0).float()        # (K, d_llm)
            # SVD 白化：去相关，使各 bin 原型方向可区分（保持语义张成子空间）
            E_centered = E - E.mean(dim=0, keepdim=True)
            _, S, V = torch.linalg.svd(E_centered, full_matrices=False)
            keep = S > 1e-3 * S[0]                            # 防近零奇异值
            r = int(min(self.n_age_bins - 1, keep.sum()))
            W = V[:r].t() / S[:r]                             # (d_llm, r)
            E_white = E_centered @ W                          # (K, r)
            E_white = F.normalize(E_white, p=2, dim=-1)
            # 可学习投影：初始化=白化矩阵，把 h_answer 压到 19 维
            # "语义年龄子空间"；瓶颈本身即正则（跨域更稳）。
            self.age_projection = nn.Linear(
                self.d_llm, W.shape[1], bias=False)
            with torch.no_grad():
                self.age_projection.weight.copy_(W.t())
            self.register_buffer('age_bin_prototypes', E_white.to(torch.bfloat16))
            # 语义子空间上的线性读出（拟合无忧，替代易饱和的 softmax）
            self.age_readout = nn.Linear(W.shape[1], 1, bias=True)
            nn.init.normal_(self.age_readout.weight, std=0.01)
            nn.init.zeros_(self.age_readout.bias)
            self.output_projection = None
        elif config.readout in ('answer_query', 'evidence_query'):
            # Predict from the <answer> hidden state only (answer_query)
            # or from the LLM-contextualized evidence-text states
            # (evidence_query, E6) instead of flattening all patches.
            # --stat_vector: 连续图统计量直接拼入读出输入，保留可迁移
            # 的连续信号（量化文本会丢失跨域信号，见 WORK.md 诊断）。
            # --stats_head: 头只读 6 维连续统计量（LLM 状态不参与）。
            # --residual_llm (E9): 统计量分支 + 零初始化 LLM 分支残差。
            if config.residual_llm:
                self.stats_branch = nn.Linear(9, config.output_dim)
                self.llm_bottleneck = nn.Linear(self.d_llm, 32)
                self.llm_branch = nn.Linear(32, config.output_dim)
                nn.init.zeros_(self.llm_branch.weight)
                nn.init.zeros_(self.llm_branch.bias)
                self.output_projection = None
                if config.stats_ridge_path:
                    # 冻结 ridge 统计量分支（可迁移下限），LLM 分支学残差
                    spec = torch.load(
                        config.stats_ridge_path, map_location='cpu',
                        weights_only=False)
                    self.register_buffer('stats_mu', spec['mu'])
                    self.register_buffer('stats_sigma', spec['sigma'])
                    with torch.no_grad():
                        self.stats_branch.weight.copy_(
                            spec['w'].unsqueeze(0))
                        self.stats_branch.bias.copy_(
                            spec['b'].reshape(-1))
                    for p in self.stats_branch.parameters():
                        p.requires_grad = False
            elif config.stats_head:
                in_dim = 9
                self.output_projection = nn.Sequential(
                    nn.Linear(in_dim, config.output_dim),
                    nn.Dropout(config.dropout),
                )
                if config.stats_ridge_path:
                    # E8c: 注入源域闭式 ridge 权重并冻结（收缩解跨域可迁移，
                    # SGD 解会依赖跨域符号翻转的特征）
                    spec = torch.load(
                        config.stats_ridge_path, map_location='cpu',
                        weights_only=False)
                    self.register_buffer('stats_mu', spec['mu'])
                    self.register_buffer('stats_sigma', spec['sigma'])
                    with torch.no_grad():
                        self.output_projection[0].weight.copy_(
                            spec['w'].unsqueeze(0))
                        self.output_projection[0].bias.copy_(
                            spec['b'].reshape(-1))
                    for p in self.output_projection.parameters():
                        p.requires_grad = False
            else:
                in_dim = self.d_llm + (9 if config.stat_vector else 0)
                self.output_projection = nn.Sequential(
                    nn.Linear(in_dim, config.output_dim),
                    nn.Dropout(config.dropout),
                )
        else:
            self.head_nf = config.d_ff * config.node_size * config.num_windows
            self.output_projection = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.head_nf, config.output_dim),
                nn.Dropout(config.dropout),
            )

        for param in self.llm.parameters():
            param.requires_grad = False

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, DFC, SFC, labels, gender=None, age=None,
                education=None):
        B, T, C, _ = DFC.shape
        device = DFC.device

        # Shared GCN over each DFC window. The raw signed DFC is used as the
        # adjacency; no additional graph normalization is applied here.
        # (--no_patches: 纯语言通路，跳过全部可学习图编码，序列只含
        #  prompt + 确定性证据文本)
        if self.config.no_patches:
            gcn_out = None
            reprogrammed = None
            mv_loss = None
        else:
            DFC_flat = DFC.reshape(B * T, C, C)
            eye = torch.eye(
                C, device=device,
                dtype=self.channel_embed_projection.weight.dtype)
            node_init = self.channel_embed_projection(eye)
            gcn_out = node_init.unsqueeze(0).expand(B * T, -1, -1)
            for layer in self.gcn_layers:
                gcn_out = F.gelu(layer(gcn_out, DFC_flat))

            # Fixed time-first order:
            # [t0c0, t0c1, ..., t0c(C-1), t1c0, ..., t(T-1)c(C-1)].
            gcn_out = gcn_out.reshape(B, T * C, -1)

            if self.config.prototype_mode == 'semantic':
                # 固定语义原型 + 小残差（训练中保持语言锚点）
                source_embeddings = (self.semantic_prototypes
                                     + self.prototype_residual).to(
                    dtype=self.word_embeddings.dtype)
            else:
                word_embeddings = self.word_embeddings.permute(1, 0).to(
                    dtype=self.mapping_layer.weight.dtype)
                source_embeddings = self.mapping_layer(
                    word_embeddings).permute(1, 0)
                source_embeddings = source_embeddings.to(
                    dtype=self.word_embeddings.dtype)

            reprogram_dtype = \
                self.reprogramming_layer.query_projection.weight.dtype
            reprogrammed = self.reprogramming_layer(
                gcn_out.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype),
            ).to(dtype=torch.bfloat16)

            # E4: 多视图语义一致性。同一 DFC 的秩视图通过共享 GCN +
            # 重编程层，约束两视图 soft token 的语义表示一致（源域-only，
            # 不使用目标数据）。
            mv_loss = None
            if self.config.mv_consistency and self.training:
                dfc_rank = rank_normalize_dfc(DFC).to(
                    dtype=self.channel_embed_projection.weight.dtype)
                rank_flat = dfc_rank.reshape(B * T, C, C)
                gcn_rank = node_init.unsqueeze(0).expand(B * T, -1, -1)
                for layer in self.gcn_layers:
                    gcn_rank = F.gelu(layer(gcn_rank, rank_flat))
                gcn_rank = gcn_rank.reshape(B, T * C, -1)
                reprogrammed_rank = self.reprogramming_layer(
                    gcn_rank.to(dtype=reprogram_dtype),
                    source_embeddings.to(dtype=reprogram_dtype),
                    source_embeddings.to(dtype=reprogram_dtype),
                ).to(dtype=torch.bfloat16)
                cos_sim = F.cosine_similarity(
                    reprogrammed, reprogrammed_rank, dim=-1)
                mv_loss = self.config.mv_lambda * (1 - cos_sim.mean())

        start_tag = self.start_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        end_tag = self.end_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        task_prompt = self.task_prompt_embeddings.unsqueeze(0).expand(
            B, -1, -1)

        # E5: 确定性图证据文本 token（在 DFC patches 之后、<answer> 之前）
        evidence_embeds = None
        evidence_stats = None
        if self.config.graph_evidence:
            levels_all, evidence_stats = compute_graph_evidence(DFC)
            evidence_ids = []
            for levels in levels_all:
                ids = []
                for (prefix, wmap, suffix), lv in zip(
                        self.evidence_templates, levels):
                    ids += prefix + [wmap[lv]] + suffix
                evidence_ids.append(ids)
            ids_t = torch.tensor(evidence_ids, device=device, dtype=torch.long)
            with torch.no_grad():
                evidence_embeds = self._word_embeddings(ids_t)

        prompt_parts = [start_tag]
        prompt_length = self.P_start
        if self.config.use_task_prompt:
            prompt_parts.append(task_prompt)
            prompt_length += self.P_task
        prompt_parts.append(end_tag)
        prompt_length += self.P_end
        if reprogrammed is not None:
            prompt_parts.append(reprogrammed.to(dtype=task_prompt.dtype))
        if evidence_embeds is not None:
            # [task prompt] [DFC soft tokens] [graph evidence text] <answer>
            prompt_parts.append(evidence_embeds)
        if self.config.readout == 'answer_query':
            # [task prompt] [DFC soft tokens] <answer>
            answer_tag = self.answer_tag_embeddings.unsqueeze(0).expand(
                B, -1, -1).to(dtype=task_prompt.dtype)
            prompt_parts.append(answer_tag)

        inputs_embeds = torch.cat(prompt_parts, dim=1)
        sequence_length = inputs_embeds.shape[1]

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
                block_parts = [
                    torch.arange(
                        prompt_length, dtype=torch.long, device=device),
                    torch.arange(T * C, dtype=torch.long, device=device),
                ]
                if self.config.readout == 'answer_query':
                    # The <answer> tokens form their own block after the DFC
                    # patches; block positions restart per block (ChatGLM 2D
                    # rotary position encoding).
                    block_parts.append(
                        torch.arange(self.P_answer, dtype=torch.long,
                                     device=device))
                block_pos = torch.cat(block_parts).unsqueeze(0).repeat(B, 1)
                position_ids = torch.stack((global_pos, block_pos), dim=1)
            else:
                position_ids = global_pos

            transformer_outputs = self._transformer(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            hidden = transformer_outputs[0].transpose(0, 1)
        else:
            position_ids = torch.arange(
                sequence_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(B, -1)
            transformer_outputs = self._transformer(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            hidden = transformer_outputs[0]

        if self.config.readout == 'answer_query':
            # Read only the last <answer> token: under the causal mask it
            # attends to the prompt and every DFC patch, so the LLM must
            # summarize the evidence into a single token before the head.
            h_answer = hidden[:, -1:, :]
            if self.use_ordinal_age_head:
                # E3v6: h_answer → 语义年龄子空间（可学习投影，L2 归一化）
                # → 子空间内线性读出。对齐损失把投影拉向真实年龄 bin 方向。
                # 显式转 fp32 避免 bf16 反向 dtype 不一致。
                h_proj = self.age_projection(h_answer.float())
                h_proj = F.normalize(h_proj, p=2, dim=-1)
                self._h_age_proj = h_proj
                prediction = self.age_readout(h_proj).squeeze(-1)
                logits = prediction.unsqueeze(-1)
            else:
                h_answer = h_answer.to(
                    dtype=self.output_projection[0].weight.dtype)
                logits = self.output_projection(h_answer).squeeze(1)
        elif self.config.readout == 'evidence_query':
            # E6: 从证据文本 token 位置读出。证据状态是冻结 LLM 对
            # 确定性图证据文本的上下文化表示（含前面 DFC patches 的
            # 因果注意力影响），均值池化后送入线性头。
            if self.config.residual_llm:
                # E9: 统计量分支（可迁移下限）+ 零初始化 LLM 分支残差
                st = evidence_stats.to(
                    dtype=self.stats_branch.weight.dtype)
                if self.config.stats_ridge_path:
                    st = (st - self.stats_mu.to(st.device)) / \
                        self.stats_sigma.to(st.device)
                y_stats = self.stats_branch(st)
                ev_start = prompt_length + (
                    T * C if reprogrammed is not None else 0)
                h_ev = hidden[:, ev_start:ev_start + evidence_embeds.shape[1]]
                h_ev = h_ev.mean(dim=1).to(
                    dtype=self.stats_branch.weight.dtype)
                y_llm = self.llm_branch(F.relu(self.llm_bottleneck(h_ev)))
                prediction = (y_stats + y_llm).squeeze(-1)
                logits = prediction.unsqueeze(-1)
            elif self.config.stats_head:
                # E8b: 头只读连续统计量（6 维），LLM 状态不参与
                h_ev = evidence_stats.to(
                    dtype=self.output_projection[0].weight.dtype)
                if self.config.stats_ridge_path:
                    h_ev = (h_ev - self.stats_mu.to(h_ev.device)) / \
                        self.stats_sigma.to(h_ev.device)
                logits = self.output_projection(h_ev).squeeze(1)
            else:
                ev_start = prompt_length + (
                    T * C if reprogrammed is not None else 0)
                h_ev = hidden[:, ev_start:ev_start + evidence_embeds.shape[1]]
                h_ev = h_ev.mean(dim=1).to(
                    dtype=self.output_projection[0].weight.dtype)
                if self.config.stat_vector:
                    h_ev = torch.cat(
                        [h_ev, evidence_stats.to(
                            dtype=self.output_projection[0].weight.dtype)],
                        dim=-1)
                logits = self.output_projection(h_ev).squeeze(1)
        else:
            hidden_patches = hidden[:, prompt_length:, :self.config.d_ff].to(
                dtype=self.output_projection[1].weight.dtype)
            logits = self.output_projection(hidden_patches)

        if self.task_type == 'regression':
            prediction = logits.squeeze(-1)
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            # stats_head/residual_llm: 统计量分支需要鲁棒损失
            # （重尾特征下 delta=1.0 的平方损失收敛到方差塌缩解）
            delta = 0.05 if (
                self.config.stats_head or self.config.residual_llm) else 1.0
            loss = F.huber_loss(
                prediction, labels.float(), delta=delta)
            if self.use_ordinal_age_head and self.config.align_lambda > 0:
                # 语义对齐（有序头模式）：把 h 的语义子空间投影拉向
                # 真实年龄 bin 的语言锚点方向（源域-only）。
                bin_idx = (labels / self.age_bin_w).long().clamp(
                    0, self.n_age_bins - 1)
                e_bin = self.age_bin_prototypes[bin_idx].float()   # (B, r)
                cos = (self._h_age_proj * e_bin).sum(dim=-1)
                align = (1 - cos).mean()
                loss = loss + self.config.align_lambda * align
            elif self.use_age_alignment:
                # 语义对齐（线性头模式，E3v7）：h_answer 直接对齐
                # 4096 维 bin 短语原型方向。
                bin_idx = (labels / self.age_bin_w).long().clamp(
                    0, self.n_age_bins - 1)
                e_bin = self.raw_bin_prototypes[bin_idx].float()   # (B, d_llm)
                h_norm = F.normalize(h_answer.float(), p=2, dim=-1)
                cos = (h_norm * e_bin).sum(dim=-1)
                align = (1 - cos).mean()
                loss = loss + self.config.align_lambda * align
        else:
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels.argmax(dim=-1)
            loss = F.cross_entropy(logits, labels)

        if mv_loss is not None:
            loss = loss + mv_loss

        hidden_state = {
            'gcn_out': gcn_out,
            'reprogrammed': reprogrammed,
        }
        if self.config.readout == 'answer_query':
            hidden_state['HL_answer'] = h_answer.squeeze(1)
        elif self.config.readout == 'evidence_query':
            hidden_state['HL_evidence'] = h_ev
        else:
            hidden_state['HL_patches'] = hidden_patches

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state=hidden_state,
        )
