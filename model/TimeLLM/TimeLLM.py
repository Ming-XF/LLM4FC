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


def normalize_adjacency(SFC, threshold=0.0):
    B, N, _ = SFC.shape
    device = SFC.device

    adj = SFC.abs().clone()

    idx = torch.arange(N, device=device)
    adj[:, idx, idx] = 0

    adj[adj < threshold] = 0

    adj = adj + torch.eye(N, device=device, dtype=adj.dtype).unsqueeze(0)

    deg = adj.sum(dim=-1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    return adj_norm


class TimeLLMConfig(BaseConfig):
    def __init__(self, node_size, num_classes,
                 d_model=64,
                 n_heads=8,
                 d_ff=128,
                 num_prototypes=500,
                 gcn_hidden=128,
                 dropout=0.1,
                 num_patches=19,
                 dataset_name='CAUEEG2',
                 llm_type='chatglm',
                 llm_path='./model/chatglm-6b'):
        super().__init__(node_size=node_size, num_classes=num_classes)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_prototypes = num_prototypes
        self.gcn_hidden = gcn_hidden
        self.dropout = dropout
        self.num_patches = num_patches
        self.dataset_name = dataset_name
        self.llm_type = llm_type
        self.llm_path = llm_path


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

        self.gcn = GCNLayer(config.gcn_hidden, config.gcn_hidden,
                            dropout=config.dropout)

        self.node_projection = nn.Sequential(
            nn.Linear(config.gcn_hidden, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.channel_embed_projection = nn.Sequential(
            nn.Linear(self.d_llm, config.gcn_hidden),
            nn.LayerNorm(config.gcn_hidden),
            nn.GELU(),
        )

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

        self.node_pos_embed = nn.Parameter(torch.zeros(1, C, self.d_llm))
        nn.init.normal_(self.node_pos_embed, std=0.02)

        # --- 用 LLM 词嵌入初始化通道节点表示 ---
        channel_name_ids = [
            self.tokenizer.encode(name, add_special_tokens=False)
            for name in self._pc.channel_names
        ]
        word_embed_weight = self._word_embeddings.weight.data
        channel_embeds = torch.stack([
            word_embed_weight[torch.tensor(ids)].mean(dim=0)
            for ids in channel_name_ids
        ])  # (C, d_llm)
        self.register_buffer("channel_name_embeddings", channel_embeds)

        self.head_nf = config.d_ff * config.num_patches
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

    def forward(self, SFC, labels, gender=None, age=None, education=None):
        B, C, _ = SFC.shape
        device = SFC.device

        adj_norm = normalize_adjacency(SFC, threshold=0.0)
        adj_norm = adj_norm.to(dtype=self.node_projection[0].weight.dtype)

        node_init = self.channel_embed_projection(
            self.channel_name_embeddings.to(
                dtype=self.channel_embed_projection[0].weight.dtype)
        )  # (C, gcn_hidden)
        node_init = node_init.unsqueeze(0).expand(B, -1, -1)
        gcn_out = self.gcn(node_init, adj_norm)
        gcn_out = F.gelu(gcn_out)

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

        reprogrammed = reprogrammed + self.node_pos_embed
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
                    torch.arange(C, dtype=torch.long, device=device),
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
