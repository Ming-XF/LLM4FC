from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from ..base import BaseConfig, ModelOutputs
from .prompts import get_prompt_config
from .gc_lora import (
    GCLoRALinear, inject_lora_to_llm,
    set_gc_lora_context, clear_gc_lora_context,
)

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



class TimeLLMConfig(BaseConfig):
    def __init__(self, node_size,
                 d_model=64,
                 n_heads=8,
                 d_ff=128,
                 num_prototypes=500,
                 gcn_hidden=128,
                 num_gcn_layers=1,
                 dropout=0.1,
                 num_windows=10,
                 task_type='classification',
                 output_dim=2,
                 dataset_name='CAUEEG',
                 llm_type='chatglm',
                 llm_path='./model/chatglm-6b',
                 use_dataset_prompt=False,
                 use_task_prompt=False,
                 use_stats_prompt=False,
                 futurefc_aux_weight=0.0,
                 sfc_recon_weight=0.0,
                 use_lora=False,
                 lora_rank=16,
                 lora_alpha=32.0,
                 lora_dropout=0.1,
                 lora_target_modules="q_proj,v_proj",
                 use_gc_lora=False,
                 block_causal_mask=False):
        super().__init__(node_size=node_size, output_dim=output_dim)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_prototypes = num_prototypes
        self.gcn_hidden = gcn_hidden
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout
        self.num_windows = num_windows
        self.task_type = task_type
        self.output_dim = output_dim
        self.dataset_name = dataset_name
        self.llm_type = llm_type
        self.llm_path = llm_path
        self.use_dataset_prompt = use_dataset_prompt
        self.use_task_prompt = use_task_prompt
        self.use_stats_prompt = use_stats_prompt
        self.futurefc_aux_weight = futurefc_aux_weight
        self.sfc_recon_weight = sfc_recon_weight
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.use_gc_lora = use_gc_lora
        self.block_causal_mask = block_causal_mask


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

        self.gcn_layers = nn.ModuleList([
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

        # ── Dataset prompt embedding ──
        ds_ids = self.tokenizer.encode(self._pc.prompt_dataset)
        with torch.no_grad():
            ds_embeds = self._word_embeddings(torch.tensor(ds_ids))
        self.register_buffer("dataset_prompt_embeddings", ds_embeds)
        self.P_dataset: int = ds_embeds.shape[0]

        # ── Task prompt embedding ──
        task_ids = self.tokenizer.encode(self._pc.prompt_task)
        with torch.no_grad():
            task_embeds = self._word_embeddings(torch.tensor(task_ids))
        self.register_buffer("task_prompt_embeddings", task_embeds)
        self.P_task: int = task_embeds.shape[0]

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
        self.task_type = config.task_type

        if self.task_type == 'multi_output_regression':
            # Next-FC prediction head 不使用 Linear 投影；
            # 改为在 forward 中对 LLM 输出的节点 token 做 Pearson 相关得到 FC 矩阵。
            self.output_projection = None  # 由 forward 中的 pearson_fc_head 替代
            self.num_windows = config.num_windows
        else:
            # 分类 / 回归 / fallback 共用同一 Linear 头
            self.output_projection = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.head_nf, config.output_dim),
                nn.Dropout(config.dropout),
            )

        for param in self.llm.parameters():
            param.requires_grad = False

        # ── LoRA / GC-LoRA injection ──
        if config.use_lora:
            target_modules = [m.strip() for m in config.lora_target_modules.split(',')]
            n_injected = inject_lora_to_llm(
                self._transformer,
                llm_type=self.llm_type,
                target_modules=target_modules,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                num_nodes=config.node_size,
                num_windows=config.num_windows,
                use_graph_cond=config.use_gc_lora,
            )
            _log.info(
                "Injected %s-LoRA into %d modules (%s), rank=%d, alpha=%.1f",
                'GC' if config.use_gc_lora else '',
                n_injected, config.lora_target_modules,
                config.lora_rank, config.lora_alpha,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _pearson_fc_head(node_embeddings):
        """从节点嵌入计算 FC 矩阵（成对 Pearson 相关，完全可微）。

        Args:
            node_embeddings: (B, C, d) — C 个节点，每个 d 维表示

        Returns:
            fc: (B, C, C) — 预测的功能连接矩阵，值域 [-1, 1]
        """
        B, C, d = node_embeddings.shape
        node_embeddings = node_embeddings.float()
        centered = node_embeddings - node_embeddings.mean(dim=-1, keepdim=True)
        # 协方差分子: (B, C, C)
        cov = torch.einsum('bic,bjc->bij', centered, centered) / (d - 1)
        # 每节点标准差: (B, C)
        std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2) + 1e-8)
        # Pearson r = cov / (std_i * std_j)
        fc = cov / (std.unsqueeze(1) * std.unsqueeze(2) + 1e-8)
        return fc

    @staticmethod
    def _create_block_causal_mask(S, P_skip, C, T, device, llm_type):
        """构造 Block-Causal Attention Mask.

        - Prompt tokens (0:P_skip): 标准因果注意力
        - Patch tokens: 同一时间窗口内 (每组 C 个) 双向可见，跨窗口因果

        Args:
            S: 总序列长度
            P_skip: prompt token 数量
            C: 每窗口通道数 (19)
            T: 时间窗口数 (10)
            llm_type: 'chatglm' -> bool mask (True=遮蔽, False=可见),
                      'llama' -> float mask (0=可见, -inf=遮蔽)

        Returns:
            (S, S) mask, dtype 与 llm_type 对应
        """
        row_idx = torch.arange(S, device=device).unsqueeze(1)  # (S, 1)
        col_idx = torch.arange(S, device=device).unsqueeze(0)  # (1, S)

        # 标准因果: col <= row
        causal_allowed = (col_idx <= row_idx)

        # 同一时间窗口内: 双向可见
        # window_of[i] = 窗口编号 (0..T-1) for patch tokens, -1 for prompt
        window_of = torch.full((S,), -1, dtype=torch.long, device=device)
        window_of[P_skip:] = torch.arange(T * C, device=device) // C
        same_window = (window_of.unsqueeze(1) == window_of.unsqueeze(0)) \
                      & (window_of.unsqueeze(1) >= 0)

        allowed = causal_allowed | same_window

        if llm_type == 'chatglm':
            return ~allowed                     # True = 遮蔽
        else:
            return torch.where(allowed, 0.0, float('-inf'))   # 0 = attend

    def freeze_for_finetune(self, freeze_lora: bool = True):
        """冻结 GCN 与 reprogram 层，仅保留预测头可训练。

        Args:
            freeze_lora: 同时冻结 LoRA / GC-LoRA 权重（默认 True，
                用于 few-shot transfer）。
        """
        for param in self.gcn_layers.parameters():
            param.requires_grad = False
        for param in self.channel_embed_projection.parameters():
            param.requires_grad = False
        for param in self.node_projection.parameters():
            param.requires_grad = False

        for param in self.mapping_layer.parameters():
            param.requires_grad = False
        for param in self.reprogramming_layer.parameters():
            param.requires_grad = False

        if freeze_lora and self.config.use_lora:
            for module in self.modules():
                if isinstance(module, GCLoRALinear):
                    module.lora_A.requires_grad_(False)
                    module.lora_B.requires_grad_(False)

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
        adj_norm = DFC_flat

        eye = torch.eye(C, device=device, dtype=self.channel_embed_projection.weight.dtype)
        node_init = self.channel_embed_projection(eye)  # (C, gcn_hidden)
        node_init = node_init.unsqueeze(0).expand(B * T, -1, -1)
        gcn_out = node_init
        for layer in self.gcn_layers:
            gcn_out = layer(gcn_out, adj_norm)
            gcn_out = F.gelu(gcn_out)

        # ── 改动2: 重组为时间优先序列 ──
        # (B*T, C, gcn_hidden) → (B, T, C, gcn_hidden) → (B, T*C, gcn_hidden)
        # token order: C0T0, C1T0, ..., C18T0, C0T1, ..., C18T9
        gcn_out = gcn_out.reshape(B, T, C, -1)
        gcn_out = gcn_out.reshape(B, T * C, -1)

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

        stats_texts = self._build_stats_prompts(SFC) if self.config.use_stats_prompt else None
        if self.config.use_stats_prompt:
            stats_ids = self.tokenizer(
                stats_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=256
            ).input_ids.to(device)
            stats_embeddings = self._word_embeddings(stats_ids)
            P_stats = stats_embeddings.shape[1]
        else:
            stats_embeddings = torch.empty(
                B, 0, self.d_llm, device=device, dtype=torch.bfloat16)
            P_stats = 0

        start_tag = self.start_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        end_tag = self.end_tag_embeddings.unsqueeze(0).expand(B, -1, -1)
        ds_prompt = self.dataset_prompt_embeddings.unsqueeze(0).expand(B, -1, -1)
        task_prompt = self.task_prompt_embeddings.unsqueeze(0).expand(B, -1, -1)

        prompt_parts = [start_tag]
        P_skip = self.P_start
        if self.config.use_dataset_prompt:
            prompt_parts.append(ds_prompt)
            P_skip += self.P_dataset
        if self.config.use_task_prompt:
            prompt_parts.append(task_prompt)
            P_skip += self.P_task
        if self.config.use_stats_prompt:
            prompt_parts.append(stats_embeddings.to(dtype=ds_prompt.dtype))
            P_skip += P_stats
        prompt_parts.append(end_tag)
        P_skip += self.P_end
        prompt_parts.append(reprogrammed.to(dtype=ds_prompt.dtype))

        inputs_embeds = torch.cat(prompt_parts, dim=1)
        S = inputs_embeds.shape[1]

        # ── 构造 position_ids 与 attention_mask（分支）────
        if self.llm_type == 'chatglm':
            # ── Block-Causal Mask (同窗口双向) 或 标准因果 ──
            if self.config.block_causal_mask:
                causal_mask = self._create_block_causal_mask(
                    S, P_skip, C, T, device, 'chatglm')
            else:
                causal_mask = torch.triu(
                    torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
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

            if self.config.use_lora:
                set_gc_lora_context(self, fc_adj=DFC, prompt_len=P_skip)

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

            if self.config.use_lora:
                clear_gc_lora_context(self)

        elif self.llm_type == 'llama':
            # RoPE 1D position_ids: (B, S)
            position_ids = torch.arange(S, dtype=torch.long,
                                        device=device).unsqueeze(0).expand(B, -1)

            if self.config.use_lora:
                set_gc_lora_context(self, fc_adj=DFC, prompt_len=P_skip)

            # 默认不传 attention_mask（LLaMA 内部自动因果）；
            # 启用 block-causal mask 时传入双向窗口 mask
            if self.config.block_causal_mask:
                attn_mask_2d = self._create_block_causal_mask(
                    S, P_skip, C, T, device, 'llama')
                attention_mask = attn_mask_2d.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
            else:
                attention_mask = None

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
            # LLaMA: 输出已是 (B, S, H)，无需转置
            HL = transformer_outputs[0]

            if self.config.use_lora:
                clear_gc_lora_context(self)

        else:
            raise ValueError(f"Unsupported llm_type: {self.llm_type}")

        # ── 输出头：分类/回归 用 Linear，多值回归用 Pearson 相关 ──
        if self.task_type == 'multi_output_regression':
            # ── Next-FC Token Prediction Head ──
            # 从 LLM 输出中截取 EEG patch token（不含 prompt 前缀）
            HL_patches = HL[:, P_skip:, :self.config.d_ff]
            B, S_patch, d_ff = HL_patches.shape
            T = self.config.num_windows        # 总窗口数 (10)
            C = self.config.node_size          # 节点数 (19)
            T_out = labels.shape[1]            # 未来窗口数 (2)，由 labels 形状动态推导

            # token 顺序为 time-first: C0T0, C1T0, ..., C18T0, C0T1, ..., C18T9
            # 未来窗口索引: T-T_out 到 T-1（即第 8, 9 个窗口）
            future_windows = torch.arange(T - T_out, T, device=HL_patches.device)

            fc_preds = []
            for w in future_windows:
                # 窗口 w 的 C 个节点 token 索引: w*C, w*C+1, ..., w*C+(C-1)
                indices = w * C + torch.arange(C, device=HL_patches.device)
                node_tokens = HL_patches[:, indices, :]               # (B, C, d_ff)
                fc_window = self._pearson_fc_head(node_tokens)        # (B, C, C)
                fc_preds.append(fc_window)

            logits = torch.stack(fc_preds, dim=1)    # (B, T_out, C, C) = (B, 2, 19, 19)

            # labels 即 DFC_target，形状 (B, T_out, C, C)，直接逐元素 MSE
            loss = F.mse_loss(logits, labels.to(logits.dtype))
        else:
            # ── 分类 / 回归：沿用原有 Linear 头 ──
            HL_patches = HL[:, P_skip:, :self.config.d_ff].to(
                dtype=self.output_projection[1].weight.dtype)
            logits = self.output_projection(HL_patches)

            if self.task_type == 'regression':
                pred = logits.squeeze(-1)
                if labels.dim() > 1 and labels.shape[-1] == 1:
                    labels = labels.squeeze(-1)
                loss = F.mse_loss(pred, labels.float())
            else:
                if labels.dim() > 1 and labels.shape[-1] > 1:
                    labels = labels.argmax(dim=-1)
                loss = F.cross_entropy(logits, labels)

        # ── FutureFC 辅助任务损失 ──
        if self.config.futurefc_aux_weight > 0 and self.task_type != 'multi_output_regression':
            T = self.config.num_windows
            C = self.config.node_size
            T_half = T // 2  # 后一半窗口

            HL_aux = HL[:, P_skip:, :self.config.d_ff]

            fc_preds = []
            for w in range(T_half, T):
                indices = w * C + torch.arange(C, device=HL_aux.device)
                node_tokens = HL_aux[:, indices, :]           # (B, C, d_ff)
                fc_window = self._pearson_fc_head(node_tokens)  # (B, C, C)
                fc_preds.append(fc_window)

            pred_fc = torch.stack(fc_preds, dim=1)            # (B, T_half, C, C)
            target_fc = DFC[:, T_half:, :, :].to(dtype=pred_fc.dtype)
            aux_loss = F.mse_loss(pred_fc, target_fc)
            loss = loss + self.config.futurefc_aux_weight * aux_loss

        # ── SFC Reconstruction 辅助损失 ──
        if self.config.sfc_recon_weight > 0 and self.training:
            T_sfc = self.config.num_windows
            C_sfc = self.config.node_size
            HL_sfc = HL[:, P_skip:, :self.config.d_ff]        # (B, T*C, d_ff)
            HL_reshaped = HL_sfc.reshape(B, T_sfc, C_sfc, self.config.d_ff)

            fc_preds = []
            for w in range(T_sfc):
                node_tokens = HL_reshaped[:, w, :, :]           # (B, C, d_ff)
                fc_window = self._pearson_fc_head(node_tokens)  # (B, C, C)
                fc_preds.append(fc_window)

            pred_sfc_wins = torch.stack(fc_preds, dim=1)        # (B, T, C, C)
            sfc_pred = pred_sfc_wins.mean(dim=1)                # (B, C, C)
            sfc_loss = F.mse_loss(sfc_pred, SFC.to(dtype=sfc_pred.dtype))
            loss = loss + self.config.sfc_recon_weight * sfc_loss

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'gcn_out': gcn_out,
                'reprogrammed': reprogrammed,
                'HL_patches': HL_patches,
            }
        )
