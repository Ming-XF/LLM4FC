from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from ..base import BaseConfig, ModelOutputs
from .prompts import get_prompt_config

import logging

_log = logging.getLogger(__name__)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        x = self.dropout(x)
        support = self.linear(x)
        out = torch.bmm(adj_norm, support)
        return out


class GATv2Layer(nn.Module):
    """GATv2 图注意力层（Brody et al., ICLR 2022）。

    e_ij = a^T · LeakyReLU( W·h_i ‖ W·h_j )

    共享 W 投影源和目标节点，用 concat（非 add）计算注意力分数，
    邻接矩阵仅作为 mask 定义邻域范围。

    Parameters
    ----------
    in_features : int
    out_features : int   concat 所有 head 后的总输出维度
    n_heads : int        注意力头数（out_features 必须能被 n_heads 整除）
    dropout : float      作用于特征和注意力权重的 dropout
    negative_slope : float  LeakyReLU 的负斜率
    """

    def __init__(self, in_features, out_features, n_heads=4,
                 dropout=0.1, negative_slope=0.2):
        super().__init__()
        assert out_features % n_heads == 0, \
            f"out_features ({out_features}) 必须能被 n_heads ({n_heads}) 整除"
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.negative_slope = negative_slope

        # 共享线性变换 W（源和目标节点使用同一投影）
        self.lin = nn.Linear(in_features, out_features, bias=False)

        # value 投影可独立，也可共享 W；此处独立以获得更大容量
        self.lin_val = nn.Linear(in_features, out_features, bias=False)

        # 注意力向量 a: shape (n_heads, 1, 2*head_dim)
        # 因为拼接的是 [W·h_i ‖ W·h_j]，所以维度为 2×head_dim
        self.attn_vec = nn.Parameter(torch.empty(n_heads, 1, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)

        self.feat_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # head 间信息混合
        self.out_proj = nn.Linear(out_features, out_features)

    def forward(self, x, adj_norm):
        """
        x       : (B, N, d_in)
        adj_norm: (B, N, N)  带符号的归一化邻接矩阵
        returns : (B, N, out_features)
        """
        B, N, _ = x.shape
        H = self.n_heads
        dh = self.head_dim

        x = self.feat_dropout(x)

        # ── 共享线性投影 ──────────────────────────────────
        h = self.lin(x).view(B, N, H, dh)           # (B, N, H, dh)
        val = self.lin_val(x).view(B, N, H, dh)     # (B, N, H, dh)

        # ── 注意力分数: e_ij = a · LeakyReLU( [h_i ‖ h_j] ) ──
        # 拼接源和目标节点特征在最后一维
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)   # (B, N, N, H, dh)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)   # (B, N, N, H, dh)
        concat = torch.cat([h_i, h_j], dim=-1)             # (B, N, N, H, 2*dh)

        e = F.leaky_relu(concat, negative_slope=self.negative_slope)
        e = (e * self.attn_vec).sum(dim=-1)           # (B, N, N, H)
        e = e.permute(0, 3, 1, 2)                     # (B, H, N, N)

        # ── 用邻接矩阵构造 mask ──────────────────────────
        # 0 边不参与 softmax（自环始终保留）
        mask = (adj_norm.unsqueeze(1) != 0)           # (B, 1, N, N)

        e_masked = e.masked_fill(~mask, float('-inf'))
        alpha = F.softmax(e_masked, dim=-1)            # (B, H, N, N)
        alpha = self.attn_dropout(alpha)

        # ── 加权聚合 value ──────────────────────────────
        val = val.permute(0, 2, 1, 3)                  # (B, H, N, dh)
        out = torch.einsum('bhij,bhjd->bhid', alpha, val)  # (B, H, N, dh)
        out = out.permute(0, 2, 1, 3).contiguous()     # (B, N, H, dh)
        out = out.view(B, N, H * dh)                    # (B, N, out_features)

        out = self.out_proj(out)

        return out


def normalize_adjacency(SFC, threshold=0.0, keep_ratio=1.0):
    """对称归一化邻接矩阵（含自环），支持 Top-K 稀疏化。

    Parameters
    ----------
    SFC : (B, N, N) 原始邻接矩阵（DFC 或静态 FC）。
    threshold : float  绝对值低于此值的边直接置零（0 = 不过滤）。
    keep_ratio : float  按绝对值排序后保留的边比例（1.0 = 全保留），
                        仅作用于非对角线元素。
    """
    B, N, _ = SFC.shape
    device = SFC.device

    adj_signed = SFC.clone()          # 保留符号，承载正/负相关
    adj_abs = SFC.abs().clone()       # 用于结构决策（Top-K、阈值、degree）

    idx = torch.arange(N, device=device)
    adj_signed[:, idx, idx] = 0
    adj_abs[:, idx, idx] = 0

    # ── Top-K 稀疏化：每张图独立选择绝对值最大的 keep_ratio 条边 ──
    if keep_ratio < 1.0:
        triu_idx = torch.triu_indices(N, N, offset=1)
        for b in range(B):
            vals = adj_abs[b, triu_idx[0], triu_idx[1]]
            k = int(round(vals.shape[0] * keep_ratio))
            if k < vals.shape[0]:
                th = vals.topk(k).values[-1]
                mask = (adj_abs[b] >= th)
                adj_abs[b] = adj_abs[b] * mask
                adj_signed[b] = adj_signed[b] * mask

    adj_signed[adj_abs < threshold] = 0
    adj_abs[adj_abs < threshold] = 0

    eye = torch.eye(N, device=device, dtype=adj_abs.dtype).unsqueeze(0)
    adj_signed = adj_signed + eye
    adj_abs = adj_abs + eye

    # degree 用绝对值版计算，保证非负
    deg = adj_abs.sum(dim=-1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    # 用符号矩阵参与归一化，GCN 可感知正/负相关
    adj_norm = D_inv_sqrt @ adj_signed @ D_inv_sqrt

    return adj_norm


class TimeLLMConfig(BaseConfig):
    def __init__(self, node_size, num_classes,
                 d_model=64,
                 n_heads=8,
                 d_ff=128,
                 num_prototypes=500,
                 gcn_hidden=128,
                 num_gcn_layers=1,
                 dropout=0.1,
                 num_windows=10,
                 dataset_name='CAUEEG2',
                 llm_type='chatglm',
                 llm_path='./model/chatglm-6b'):
        super().__init__(node_size=node_size, num_classes=num_classes)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_prototypes = num_prototypes
        self.gcn_hidden = gcn_hidden
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout
        self.num_windows = num_windows
        self.dataset_name = dataset_name
        self.llm_type = llm_type
        self.llm_path = llm_path
        self.use_gatv2 = True
        self.gat_n_heads = 4
        self.gat_negative_slope = 0.2


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
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

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self._reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def _reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Model(nn.Module):

    def __init__(self, config: TimeLLMConfig):
        super().__init__()
        self.config = config
        C = config.node_size
        self.d_model = config.d_model
        self.num_prototypes = config.num_prototypes
        self.n_heads = config.n_heads
        self.d_llm = 4096
        self.llm_type = config.llm_type
        self.llm_path = config.llm_path

        if config.use_gatv2:
            self.graph_layers = nn.ModuleList([
                GATv2Layer(config.gcn_hidden, config.gcn_hidden,
                           n_heads=config.gat_n_heads,
                           dropout=config.dropout,
                           negative_slope=config.gat_negative_slope)
                for _ in range(config.num_gcn_layers)
            ])
        else:
            self.graph_layers = nn.ModuleList([
                GCNLayer(config.gcn_hidden, config.gcn_hidden, dropout=config.dropout)
                for _ in range(config.num_gcn_layers)
            ])

        self.node_projection = nn.Sequential(
            nn.Linear(config.gcn_hidden, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.channel_embed_projection = nn.Linear(config.node_size, config.gcn_hidden)

        # ── LLM 加载（分支：chatglm / llama）──────────────────
        if self.llm_type == 'chatglm':
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_path, trust_remote_code=True
            )
            self.llm = AutoModel.from_pretrained(
                self.llm_path, trust_remote_code=True
            ).bfloat16()
            self.llm.transformer.gradient_checkpointing = False
            self.position_encoding_2d = self.llm.transformer.position_encoding_2d
            self._word_embeddings = self.llm.transformer.word_embeddings
            self._transformer = self.llm.transformer
        elif self.llm_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_path
            )
            self.llm = AutoModel.from_pretrained(
                self.llm_path, torch_dtype=torch.bfloat16
            )
            self.llm.config.use_cache = False
            self.position_encoding_2d = False
            # AutoModel 返回 LlamaModel（无 .model 子模块），
            # embed_tokens / layers / norm 直接挂在顶层
            self._word_embeddings = self.llm.embed_tokens
            self._transformer = self.llm
        else:
            raise ValueError(f"Unsupported llm_type: {self.llm_type}")

        self._pc = get_prompt_config(config.dataset_name)
        prefix_ids = self.tokenizer.encode(self._pc.system_prompt)
        with torch.no_grad():
            prefix_embeds = self._word_embeddings(
                torch.tensor(prefix_ids)
            )
        self.register_buffer("prompt_prefix_embeddings", prefix_embeds)
        self.P_prefix: int = prefix_embeds.shape[0]

        start_tag_ids = self.tokenizer.encode("<start_prompt>\n")
        with torch.no_grad():
            start_tag_embeds = self._word_embeddings(
                torch.tensor(start_tag_ids)
            )
        self.register_buffer("start_tag_embeddings", start_tag_embeds)
        self.P_start: int = start_tag_embeds.shape[0]

        end_tag_ids = self.tokenizer.encode("\n<end_prompt>")
        with torch.no_grad():
            end_tag_embeds = self._word_embeddings(
                torch.tensor(end_tag_ids)
            )
        self.register_buffer("end_tag_embeddings", end_tag_embeds)
        self.P_end: int = end_tag_embeds.shape[0]

        self.word_embeddings = self._word_embeddings.weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_prototypes)

        self.reprogramming_layer = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
        )

        # 节点初始特征用 one-hot → Linear 投影，训练中可学习

        self.head_nf = config.d_ff * config.node_size * config.num_windows
        self.output_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.head_nf, config.num_classes),
            nn.Dropout(config.dropout),
        )

        for param in self.llm.parameters():
            param.requires_grad = False

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_stats_prompts(self, SFC):
        pc = self._pc
        B, N, _ = SFC.shape

        prompts = []
        for b in range(B):
            fc = SFC[b].clone()
            fc.fill_diagonal_(0.0)

            triu_idx = torch.triu_indices(N, N, offset=1)
            all_edges = fc[triu_idx[0], triu_idx[1]]
            mean_fc = float(all_edges.mean())
            std_fc = float(all_edges.std())

            edge_values = []
            for i, j in zip(triu_idx[0].tolist(), triu_idx[1].tolist()):
                edge_values.append({
                    'i': i, 'j': j,
                    'val': fc[i, j].item(),
                    'name': f"{pc.channel_names[i]}-{pc.channel_names[j]}",
                })

            edge_values.sort(key=lambda e: e['val'], reverse=True)
            max_edge = edge_values[0]

            pos_edges = [e for e in edge_values if e['val'] > 0]
            min_pos_edge = pos_edges[-1] if pos_edges else max_edge

            neg_edges = [e for e in edge_values if e['val'] < 0]
            max_neg_edge = neg_edges[-1] if neg_edges else edge_values[-1]

            frontal_idx = pc.channel_groups['frontal']
            frontal_vals = []
            for a in range(len(frontal_idx)):
                for k in range(a + 1, len(frontal_idx)):
                    frontal_vals.append(
                        fc[frontal_idx[a], frontal_idx[k]].item())
            fc_frontal = float(torch.tensor(frontal_vals).mean()) if frontal_vals else 0.0

            homo_vals = [fc[i, j].item() for i, j in pc.homologous_pairs]
            fc_homologous = float(torch.tensor(homo_vals).mean())

            prompt = pc.prompt_stats_template.format(
                max_pair=max_edge['name'],
                max_val=max_edge['val'],
                min_pos_pair=min_pos_edge['name'],
                min_pos_val=min_pos_edge['val'],
                max_neg_pair=max_neg_edge['name'],
                max_neg_val=max_neg_edge['val'],
                fc_frontal=fc_frontal,
                fc_homologous=fc_homologous,
                mean_fc=mean_fc,
                std_fc=std_fc,
            )
            prompts.append(prompt)

        return prompts

    def forward(self, DFC, SFC, labels, gender=None, age=None, education=None):
        B, T, C, _ = DFC.shape
        device = DFC.device

        # ── 改动1: 共享 GCN 逐窗口编码 ──
        # (B, T, C, C) → (B*T, C, C)，每个窗口独立过同一个 GCN
        DFC_flat = DFC.reshape(B * T, C, C)
        adj_norm = normalize_adjacency(DFC_flat, threshold=0.0)
        adj_norm = adj_norm.to(dtype=self.node_projection[0].weight.dtype)

        eye = torch.eye(C, device=device, dtype=self.channel_embed_projection.weight.dtype)
        node_init = self.channel_embed_projection(eye)  # (C, gcn_hidden)
        node_init = node_init.unsqueeze(0).expand(B * T, -1, -1)
        gcn_out = node_init
        for layer in self.graph_layers:
            gcn_out = layer(gcn_out, adj_norm)
            gcn_out = F.gelu(gcn_out)

        # ── 改动2: 重组为通道优先序列 ──
        # (B*T, C, gcn_hidden) → (B, T, C, gcn_hidden) → (B, C, T, gcn_hidden)
        # → (B, C*T, gcn_hidden)
        # token order: C0T0, C0T1, ..., C0T9, C1T0, ..., C18T9
        gcn_out = gcn_out.reshape(B, T, C, -1)
        gcn_out = gcn_out.transpose(1, 2).contiguous()
        gcn_out = gcn_out.reshape(B, C * T, -1)

        node_embeddings = self.node_projection(gcn_out)

        we = self.word_embeddings.permute(1, 0).to(
            dtype=self.mapping_layer.weight.dtype)
        source_embeddings = self.mapping_layer(we).permute(1, 0)
        source_embeddings = source_embeddings.to(
            dtype=self.word_embeddings.dtype)

        reprogram_dtype = self.reprogramming_layer.query_projection.weight.dtype
        reprogrammed = self.reprogramming_layer(
            node_embeddings.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
        )

        reprogrammed = reprogrammed.to(dtype=torch.bfloat16)

        stats_texts = self._build_stats_prompts(SFC)
        stats_ids = self.tokenizer(
            stats_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=256
        ).input_ids.to(device)
        stats_embeddings = self._word_embeddings(stats_ids)

        start_tag = self.start_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        end_tag = self.end_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        prompt_prefix = self.prompt_prefix_embeddings.unsqueeze(0).expand(B, -1, -1)

        inputs_embeds = torch.cat(
            [start_tag,
             prompt_prefix,
             stats_embeddings.to(dtype=prompt_prefix.dtype),
             end_tag,
             reprogrammed.to(dtype=prompt_prefix.dtype)],
            dim=1,
        )
        S = inputs_embeds.shape[1]
        P_skip = (self.P_start + self.P_prefix +
                  stats_embeddings.shape[1] + self.P_end)

        # ── 构造 position_ids 与 attention_mask（分支）────
        if self.llm_type == 'chatglm':
            # 因果注意力：token i 只能 attend token 0..i
            # ChatGLM 中 True=遮蔽, False=可见
            causal_mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

            global_pos = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
            if self.position_encoding_2d:
                block_pos = torch.cat([
                    torch.arange(P_skip, dtype=torch.long, device=device),
                    torch.arange(T * C, dtype=torch.long, device=device),
                ]).unsqueeze(0).repeat(B, 1)
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
            # ChatGLM: (S, B, H) → (B, S, H)
            HL = transformer_outputs[0].transpose(0, 1)

        elif self.llm_type == 'llama':
            # RoPE 1D position_ids: (B, S)
            position_ids = torch.arange(S, dtype=torch.long,
                                        device=device).unsqueeze(0).expand(B, -1)

            # 不传 attention_mask，LLaMA 内部自动构造因果 mask
            transformer_outputs = self._transformer(
                input_ids=None,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            # LLaMA: 输出已是 (B, S, H)，无需转置
            HL = transformer_outputs[0]

        else:
            raise ValueError(f"Unsupported llm_type: {self.llm_type}")

        HL_patches = HL[:, P_skip:, :self.config.d_ff].to(
            dtype=self.output_projection[1].weight.dtype)
        logits = self.output_projection(HL_patches)

        if labels.dim() > 1 and labels.shape[-1] > 1:
            labels = labels.argmax(dim=-1)
        loss = F.cross_entropy(logits, labels)

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'gcn_out': gcn_out,
                'reprogrammed': reprogrammed,
                'HL_patches': HL_patches,
            }
        )
