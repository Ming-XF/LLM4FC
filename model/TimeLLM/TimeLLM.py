from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from ..base import BaseConfig, ModelOutputs
from .prompts import get_prompt_config
from .gc_lora import (
    inject_lora_to_llm,
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
                 use_lora=False,
                 lora_rank=16,
                 lora_alpha=32.0,
                 lora_dropout=0.1,
                 lora_target_modules="q_proj,v_proj",
                 use_gc_lora=False,
                 lora_num_layers=-1,
                 block_causal_mask=False,
                 token_order='time_first',
                 use_cvib=False,
                 cvib_mode='vae',
                 cvib_beta=1e-3):
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
        self.use_dataset_prompt = use_dataset_prompt
        self.use_task_prompt = use_task_prompt
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.use_gc_lora = use_gc_lora
        self.lora_num_layers = lora_num_layers
        self.block_causal_mask = block_causal_mask
        self.token_order = token_order
        self.use_cvib = use_cvib
        self.cvib_mode = cvib_mode
        self.cvib_beta = cvib_beta


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


class LabelEncoder(nn.Module):
    """条件标签编码器：分类 -> Embedding，回归 -> 线性中心。

    将标签 y 编码为 (B, d_model) 的类/值条件中心，供 CVIB 的
    类条件先验 (vae) 或对比中心 (contrastive) 使用。
    """

    def __init__(self, task_type, output_dim, d_model):
        super().__init__()
        self.task_type = task_type
        if task_type == 'classification':
            self.emb = nn.Embedding(output_dim, d_model)
        else:
            self.emb = nn.Linear(1, d_model)

    def forward(self, y):
        if self.task_type == 'classification':
            return self.emb(y)                    # (B, d_model)
        return self.emb(y[:, None])               # (B, d_model)


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
            GCNLayer(self.d_model, self.d_model, dropout=config.dropout)
            for _ in range(config.num_gcn_layers)
        ])
        self.channel_embed_projection = nn.Linear(config.node_size, config.d_model)

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

        # ── CVIB：GCN 输出作为 z（分类与回归均支持）──────────
        self.use_cvib = config.use_cvib
        if config.use_cvib:
            self.cvib_mode = config.cvib_mode
            self.cvib_beta = config.cvib_beta
            if config.cvib_mode == 'vae':
                self.mu_head = nn.Linear(config.d_model, config.d_model)
                self.logvar_head = nn.Linear(config.d_model, config.d_model)
                self.label_encoder = LabelEncoder(
                    config.task_type, config.output_dim, config.d_model)
                self.label_logvar_encoder = LabelEncoder(
                    config.task_type, config.output_dim, config.d_model)
            elif config.cvib_mode == 'contrastive':
                self.label_encoder = LabelEncoder(
                    config.task_type, config.output_dim, config.d_model)
            else:
                raise ValueError(f"Unsupported cvib_mode: {config.cvib_mode}")

        # 节点初始特征用 one-hot → Linear 投影，训练中可学习

        self.head_nf = config.d_ff * config.node_size * config.num_windows
        self.task_type = config.task_type

        # 分类 / 回归 共用同一 Linear 头
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
                num_layers=config.lora_num_layers,
                token_order=config.token_order,
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
    def _create_block_causal_mask(S, P_skip, C, T, device, llm_type, token_order):
        """构造 Block-Causal Attention Mask.

        - Prompt tokens (0:P_skip): 标准因果注意力
        - Patch tokens: 同一 group 内双向可见，跨 group 因果
          - time_first: group = 同一时间窗口 (每组 C 个通道)
          - node_first: group = 同一节点 (每组 T 个时间窗口)

        Args:
            S: 总序列长度
            P_skip: prompt token 数量
            C: 通道数 (19)
            T: 时间窗口数 (10)
            llm_type: 'chatglm' -> bool mask (True=遮蔽, False=可见),
                      'llama' -> float mask (0=可见, -inf=遮蔽)
            token_order: 'time_first' | 'node_first'

        Returns:
            (S, S) mask, dtype 与 llm_type 对应
        """
        row_idx = torch.arange(S, device=device).unsqueeze(1)  # (S, 1)
        col_idx = torch.arange(S, device=device).unsqueeze(0)  # (1, S)

        # 标准因果: col <= row
        causal_allowed = (col_idx <= row_idx)

        # 同一 group 内: 双向可见
        # group_of[i] = group 编号 (0..num_groups-1) for patch tokens, -1 for prompt
        group_size = C if token_order == 'time_first' else T
        group_of = torch.full((S,), -1, dtype=torch.long, device=device)
        group_of[P_skip:] = torch.arange(T * C, device=device) // group_size
        same_group = (group_of.unsqueeze(1) == group_of.unsqueeze(0)) \
                     & (group_of.unsqueeze(1) >= 0)

        allowed = causal_allowed | same_group

        if llm_type == 'chatglm':
            return ~allowed                     # True = 遮蔽
        else:
            return torch.where(allowed, 0.0, float('-inf'))   # 0 = attend

    def _normalize_label(self, labels, device):
        """将 labels 归一化为 CVIB 所需的条件变量 y。

        - 分类: one-hot -> argmax，否则取整 → long
        - 回归: float，(B,1) -> squeeze
        """
        if self.task_type == 'regression':
            y = labels.float()
            if y.dim() > 1 and y.shape[-1] == 1:
                y = y.squeeze(-1)
        else:
            y = labels
            if y.dim() > 1 and y.shape[-1] > 1:
                y = y.argmax(dim=-1)
            y = y.long()
        return y.to(device)

    def _cvib_encode(self, gcn_out, labels, device):
        """根据 cvib_mode 生成 z 与辅助损失。

        Args:
            gcn_out: (B, N, d_model) GCN 输出，作为 z 的基底
            labels: 原始标签

        Returns:
            z: (B, N, d_model) 顶替 gcn_out 作为 reprogramming 的 query
            aux: scalar tensor（仅训练时），否则 None
        """
        y = self._normalize_label(labels, device)
        B, N, _ = gcn_out.shape

        if self.cvib_mode == 'vae':
            mu = self.mu_head(gcn_out)                      # (B, N, d)
            logvar = self.logvar_head(gcn_out).clamp(-10.0, 10.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + std * eps
                # 类条件先验 r(z|y) = N(mu_r, sigma_r^2)
                mu_r = self.label_encoder(y).unsqueeze(1)              # (B, 1, d)
                logvar_r = self.label_logvar_encoder(y).unsqueeze(1).clamp(-10.0, 10.0)
                var = torch.exp(logvar)
                var_r = torch.exp(logvar_r)
                kl = 0.5 * (logvar_r - logvar - 1.0 + (var + (mu - mu_r) ** 2) / var_r)
                aux = kl.mean()
            else:
                z = mu
                aux = None
        elif self.cvib_mode == 'contrastive':
            z = gcn_out
            aux = None
            if self.training:
                c_y = self.label_encoder(y)                            # (B, d)
                z_norm = F.normalize(z, dim=-1)                        # (B, N, d)
                c_norm = F.normalize(c_y, dim=-1)                      # (B, d)
                cos = (z_norm * c_norm.unsqueeze(1)).sum(dim=-1)       # (B, N)
                aux = (1.0 - cos).mean()                               # 标量，最小化
        else:
            raise ValueError(f"Unsupported cvib_mode: {self.cvib_mode}")

        return z, aux

    def forward(self, DFC, SFC, labels, gender=None, age=None, education=None):
        B, T, C, _ = DFC.shape
        device = DFC.device

        # ── 共享 GCN 逐窗口编码 ──
        # (B, T, C, C) → (B*T, C, C)，每个窗口独立过同一个 GCN
        DFC_flat = DFC.reshape(B * T, C, C)
        adj_norm = DFC_flat

        eye = torch.eye(C, device=device, dtype=self.channel_embed_projection.weight.dtype)
        node_init = self.channel_embed_projection(eye)  # (C, d_model)
        node_init = node_init.unsqueeze(0).expand(B * T, -1, -1)
        gcn_out = node_init
        for layer in self.gcn_layers:
            gcn_out = layer(gcn_out, adj_norm)
            gcn_out = F.gelu(gcn_out)

        # 重组为 patch 序列
        # (B*T, C, d_model) → (B, T, C, d_model) → (B, T*C, d_model) 或 (B, C*T, d_model)
        gcn_out = gcn_out.reshape(B, T, C, -1)
        if self.config.token_order == 'node_first':
            # Node-First: [c0t0, c0t1, ..., c_{C-1}t_{T-1}]
            gcn_out = gcn_out.permute(0, 2, 1, 3)          # (B, C, T, d_model)
            gcn_out = gcn_out.reshape(B, C * T, -1)        # (B, C*T, d_model)
        else:
            # Time-First: [t0c0, t0c1, ..., t_{T-1}c_{C-1}]
            gcn_out = gcn_out.reshape(B, T * C, -1)        # (B, T*C, d_model)

        # ── CVIB：GCN 输出作为 z ──
        cvib_aux = None
        if self.config.use_cvib:
            z, cvib_aux = self._cvib_encode(gcn_out, labels, device)
        else:
            z = gcn_out

        we = self.word_embeddings.permute(1, 0).to(
            dtype=self.mapping_layer.weight.dtype)
        source_embeddings = self.mapping_layer(we).permute(1, 0)
        source_embeddings = source_embeddings.to(
            dtype=self.word_embeddings.dtype)

        reprogram_dtype = self.reprogramming_layer.query_projection.weight.dtype
        reprogrammed = self.reprogramming_layer(
            z.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
        )

        reprogrammed = reprogrammed.to(dtype=torch.bfloat16)

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
                    S, P_skip, C, T, device, 'chatglm', self.config.token_order)
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
                    S, P_skip, C, T, device, 'llama', self.config.token_order)
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

        # ── 输出头：分类 / 回归 ──
        HL_patches = HL[:, P_skip:, :self.config.d_ff].to(
            dtype=self.output_projection[1].weight.dtype)
        logits = self.output_projection(HL_patches)

        if self.task_type == 'regression':
            pred = logits.squeeze(-1)
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            loss = F.huber_loss(pred, labels.float(), delta=1.0)
        else:
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels.argmax(dim=-1)
            loss = F.cross_entropy(logits, labels)

        # ── CVIB 辅助损失（仅训练时）──
        if cvib_aux is not None:
            loss = loss + self.config.cvib_beta * cvib_aux

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'gcn_out': gcn_out,
                'reprogrammed': reprogrammed,
                'HL_patches': HL_patches,
                'z': z,
            }
        )
