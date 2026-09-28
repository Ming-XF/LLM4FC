from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..base import BaseConfig, ModelOutputs
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
                 mv_consistency=False,
                 mv_lambda=0.0):
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
        self.mv_consistency = mv_consistency
        self.mv_lambda = mv_lambda


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
            bin_embeds = []
            with torch.no_grad():
                for k in range(self.n_age_bins):
                    lo = int(k * config.age_bin_width)
                    hi = int((k + 1) * config.age_bin_width)
                    phrase = (f"an EEG pattern associated with age "
                              f"{lo} to {hi}")
                    ids = torch.tensor(self.tokenizer.encode(phrase))
                    emb = self._word_embeddings(ids).mean(dim=0)
                    emb = F.normalize(emb, p=2, dim=-1)
                    bin_embeds.append(emb)
            self.register_buffer(
                'age_bin_prototypes', torch.stack(bin_embeds, dim=0))
            self.age_temperature = nn.Parameter(torch.tensor(1.0))
            self.output_projection = None
        elif config.readout == 'answer_query':
            # Predict from the <answer> hidden state only instead of
            # flattening all T*C patch hidden states.
            self.output_projection = nn.Sequential(
                nn.Linear(self.d_llm, config.output_dim),
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

        prompt_parts = [start_tag]
        prompt_length = self.P_start
        if self.config.use_task_prompt:
            prompt_parts.append(task_prompt)
            prompt_length += self.P_task
        prompt_parts.append(end_tag)
        prompt_length += self.P_end
        prompt_parts.append(reprogrammed.to(dtype=task_prompt.dtype))
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
                # E3: 与年龄区间原型（单位化）做点积 → 区间分布 → 期望年龄。
                # h_answer 不归一化：4096 维归一化余弦量级仅 ~0.02，
                # softmax 梯度几乎为零；保留 |h| 使 logits 量级 ~1。
                # 显式转 fp32 避免 bf16 反向 dtype 不一致。
                cos_sim = h_answer.float() @ self.age_bin_prototypes.float().t()  # (B, K)
                p = torch.softmax(
                    self.age_temperature * cos_sim, dim=-1)
                self._age_bin_probs = p
                prediction = (p * self.age_bin_centers.float()).sum(dim=-1)
                logits = prediction.unsqueeze(-1)
            else:
                h_answer = h_answer.to(
                    dtype=self.output_projection[0].weight.dtype)
                logits = self.output_projection(h_answer).squeeze(1)
        else:
            hidden_patches = hidden[:, prompt_length:, :self.config.d_ff].to(
                dtype=self.output_projection[1].weight.dtype)
            logits = self.output_projection(hidden_patches)

        if self.task_type == 'regression':
            prediction = logits.squeeze(-1)
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            loss = F.huber_loss(
                prediction, labels.float(), delta=1.0)
            if self.use_ordinal_age_head and self.config.emd_lambda > 0:
                # EMD (Wasserstein-1) 有序正则：预测区间分布与
                # 标签 bin 的 CDF 差，约束相邻区间语义。
                bin_idx = (labels / self.age_bin_w).long().clamp(
                    0, self.n_age_bins - 1)
                q = F.one_hot(bin_idx, self.n_age_bins).float()
                cdf_p = self._age_bin_probs.cumsum(dim=-1)
                cdf_q = q.cumsum(dim=-1)
                emd = (cdf_p - cdf_q).abs().mean()
                loss = loss + self.config.emd_lambda * emd
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
        else:
            hidden_state['HL_patches'] = hidden_patches

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state=hidden_state,
        )
