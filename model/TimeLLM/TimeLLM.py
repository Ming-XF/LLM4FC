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
                 use_gcn=False,
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
                 use_channel_prototype=False,
                 channel_proto_rank=32,
                 channel_proto_diversity_weight=0.0):
        super().__init__(node_size=node_size, output_dim=output_dim)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_prototypes = num_prototypes
        self.num_gcn_layers = num_gcn_layers
        self.use_gcn = use_gcn
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
        self.use_channel_prototype = use_channel_prototype
        self.channel_proto_rank = channel_proto_rank
        self.channel_proto_diversity_weight = channel_proto_diversity_weight


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
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)

        # source/value 支持两种形状：
        #   (S, d_llm)         —— 所有 query 共享同一套原型（原路径）
        #   (B, S, d_llm)      —— 每个 query 通道各自一套原型（channel_prototype 向量化）
        if source_embedding.dim() == 2:
            S, _ = source_embedding.shape
            source_embedding = self.key_projection(source_embedding).view(S, H, -1)
            value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        else:
            _, S, _ = source_embedding.shape
            source_embedding = self.key_projection(source_embedding).view(B, S, H, -1)
            value_embedding = self.value_projection(value_embedding).view(B, S, H, -1)

        out = self._reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def _reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)

        if source_embedding.dim() == 3:   # 共享 source: (S, H, E)
            scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        else:                             # 每通道独立 source: (B, S, H, E)
            scores = torch.einsum("blhe,bshe->bhls", target_embedding, source_embedding)
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            reprogramming_embedding = torch.einsum("bhls,bshe->blhe", A, value_embedding)

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

        # 数据流：one-hot → reprogram(d_llm) → GCN(d_llm) → LLM
        self.gcn_layers = nn.ModuleList([
            GCNLayer(self.d_llm, self.d_llm, dropout=config.dropout)
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

        # 通道专属原型集合：对 mapping_layer 做 LoRA 低秩分解，1 套共享 → 每通道 1 套
        # W_c = W_map + B[c] @ A；P_c = P_shared + B[c] @ (A @ we)
        if config.use_channel_prototype:
            self.channel_proto_rank = config.channel_proto_rank
            # A: 共享 down (r, vocab)，随机初始化
            self.channel_lora_A = nn.Parameter(
                torch.empty(config.channel_proto_rank, self.vocab_size))
            nn.init.kaiming_uniform_(self.channel_lora_A)
            # B: 每通道 up (C, S, r)，零初始化 → ΔW=0，初始即共享集合，平滑启动
            self.channel_lora_B = nn.Parameter(
                torch.zeros(config.node_size, config.num_prototypes,
                            config.channel_proto_rank))
        else:
            self.channel_lora_A = None
            self.channel_lora_B = None
            self.channel_proto_rank = 0

        self.reprogramming_layer = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
        )

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

    def forward(self, DFC, SFC, labels, gender=None, age=None, education=None):
        B, T, C, _ = DFC.shape
        device = DFC.device

        we = self.word_embeddings.permute(1, 0).to(
            dtype=self.mapping_layer.weight.dtype)
        source_embeddings = self.mapping_layer(we).permute(1, 0)  # (S, d_llm) 共享 base 原型

        reprogram_dtype = self.reprogramming_layer.query_projection.weight.dtype

        # ── one-hot → reprogram(逐通道) → GCN(d_llm) → LLM ──
        # 先用 one-hot 通道身份做 Reprogram，避免 GCN 提前混通道导致同质化
        eye = torch.eye(C, device=device, dtype=self.channel_embed_projection.weight.dtype)
        node_init = self.channel_embed_projection(eye)         # (C, d_model)

        if self.config.use_channel_prototype:
            A_we = self.channel_lora_A @ we.permute(1, 0)      # (r, d_llm)
            # 向量化：一次性算出所有通道专属原型 (C, S, d_llm)，批量做 cross-attention，
            # 替代原先 for c in range(C) 的 19 次串行调用
            delta = self.channel_lora_B @ A_we                 # (C, S, d_llm)
            P_channels = source_embeddings.unsqueeze(0) + delta  # (C, S, d_llm)
            reprogrammed_channels = self.reprogramming_layer(
                node_init.unsqueeze(1).to(dtype=reprogram_dtype),  # (C, 1, d_model)
                P_channels.to(dtype=reprogram_dtype),
                P_channels.to(dtype=reprogram_dtype),
            ).squeeze(1)                                        # (C, d_llm)
        else:
            reprogrammed_channels = self.reprogramming_layer(
                node_init.unsqueeze(1).to(dtype=reprogram_dtype),  # (C,1,d_model)
                source_embeddings.to(dtype=reprogram_dtype),
                source_embeddings.to(dtype=reprogram_dtype),
            ).squeeze(1)                                        # (C, d_llm)

        # GCN 逐窗口图卷积：窗口无关的通道嵌入 + 每窗口邻接
        DFC_flat = DFC.reshape(B * T, C, C)
        gcn_in = reprogrammed_channels.unsqueeze(0).expand(B * T, -1, -1)  # (B*T, C, d_llm)
        if self.config.use_gcn:
            gcn_out = gcn_in
            for layer in self.gcn_layers:
                gcn_out = layer(gcn_out, DFC_flat)
                gcn_out = F.gelu(gcn_out)
        else:
            gcn_out = gcn_in   # 不做图卷积，窗口间仅靠 LLM 位置编码区分
        # 重组为时间优先序列（token 顺序：C0T0, C1T0, ... C18T0, C0T1, ...）
        reprogrammed = gcn_out.reshape(B, T, C, -1).reshape(B, T * C, -1)  # (B, T*C, d_llm)

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

        # ── 通道原型多样性正则（Barlow-Twins / VICReg 风格协方差）──
        # 仅当 use_channel_prototype 且 weight > 0 时生效，零开销
        if (self.config.use_channel_prototype
                and self.config.channel_proto_diversity_weight > 0.0):
            proto_B = self.channel_lora_B              # (C, S, r)
            Cc, Sr, rr = proto_B.shape                 # 注意勿覆盖 batch 变量 B
            Bc = proto_B.reshape(Cc, Sr * rr)          # (C, S*r)
            Bc = Bc - Bc.mean(dim=0, keepdim=True)     # 中心化
            cov = (Bc @ Bc.t()) / (Sr * rr)            # (C, C)
            off = (cov - torch.diag(cov.diag())).pow(2).mean()   # 非对角→0（去相关）
            on = (cov.diag() - 1.0).pow(2).mean()                # 对角→1（防坍缩）
            L_div = off + on
            loss = loss + self.config.channel_proto_diversity_weight * L_div

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'gcn_out': gcn_out,
                'reprogrammed': reprogrammed,
                'HL_patches': HL_patches,
            }
        )
