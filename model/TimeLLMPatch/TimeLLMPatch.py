from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..base import BaseConfig, ModelOutputs
from .prompts import get_prompt_config


class PatchEmbedding(nn.Module):
    """Time-LLM patching: replicate-pad, unfold, project each patch to d_model.

    Mirrors the upstream Time-LLM PatchEmbedding (layers/Embed.py).
    """

    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = nn.Conv1d(
            patch_len, d_model, kernel_size=3,
            padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(
            self.value_embedding.weight, mode='fan_in',
            nonlinearity='leaky_relu')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, T) raw EEG
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x.permute(0, 2, 1)).transpose(1, 2)
        return self.dropout(x), n_vars


class TimeLLMPatchConfig(BaseConfig):
    def __init__(self, node_size,
                 time_series_size=12000,
                 d_model=128,
                 n_heads=8,
                 d_ff=128,
                 num_prototypes=1000,
                 dropout=0.1,
                 patch_len=512,
                 patch_stride=512,
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
        self.d_ff = d_ff
        self.num_prototypes = num_prototypes
        self.dropout = dropout
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.task_type = task_type
        self.output_dim = output_dim
        self.dataset_name = dataset_name
        self.llm_type = llm_type
        self.llm_path = llm_path
        self.llm_init = llm_init
        self.use_task_prompt = use_task_prompt

        # Patches per channel (replicate pad of `stride` adds one patch, and
        # the final tail is kept): (T - patch_len) // stride + 2.
        self.patch_nums = (time_series_size - patch_len) // patch_stride + 2
        # Head sees channel mean-pooled patches: (B, patch_nums, d_ff).
        self.head_nf = d_ff * self.patch_nums


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
    def __init__(self, config: TimeLLMPatchConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_prototypes = config.num_prototypes
        self.n_heads = config.n_heads
        self.d_llm = 4096
        self.llm_type = config.llm_type
        self.llm_path = config.llm_path

        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            patch_len=config.patch_len,
            stride=config.patch_stride,
            dropout=config.dropout,
        )

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

        self.word_embeddings = self._word_embeddings.weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(
            self.vocab_size, self.num_prototypes)
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
        )

        self.task_type = config.task_type
        self.output_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(config.head_nf, config.output_dim),
            nn.Dropout(config.dropout),
        )

        for param in self.llm.parameters():
            param.requires_grad = False

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, x, labels):
        B, N, T = x.shape
        device = x.device

        # Per-sample per-channel z-score: amplitude invariance is the key
        # property for cross-domain zero-shot transfer.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + 1e-5)

        # Patch embedding, then channel-major flatten:
        # [c0p0, ..., c0p(P-1), c1p0, ..., c(N-1)p(P-1)].
        x = x.to(dtype=self.patch_embedding.value_embedding.weight.dtype)
        enc_out, _ = self.patch_embedding(x)
        P = enc_out.shape[1]
        enc_out = enc_out.reshape(B, N * P, -1)

        word_embeddings = self.word_embeddings.permute(1, 0).to(
            dtype=self.mapping_layer.weight.dtype)
        source_embeddings = self.mapping_layer(
            word_embeddings).permute(1, 0)
        source_embeddings = source_embeddings.to(
            dtype=self.word_embeddings.dtype)

        reprogram_dtype = \
            self.reprogramming_layer.query_projection.weight.dtype
        reprogrammed = self.reprogramming_layer(
            enc_out.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
            source_embeddings.to(dtype=reprogram_dtype),
        ).to(dtype=torch.bfloat16)

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
                block_pos = torch.cat([
                    torch.arange(
                        prompt_length, dtype=torch.long, device=device),
                    torch.arange(N * P, dtype=torch.long, device=device),
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

        hidden_patches = hidden[:, prompt_length:, :self.config.d_ff].to(
            dtype=self.output_projection[1].weight.dtype)

        # (B, N*P, d_ff) -> (B, N, P, d_ff) -> channel mean-pool ->
        # (B, P, d_ff) -> flatten -> scalar.
        hidden_patches = hidden_patches.reshape(
            B, N, self.config.patch_nums, -1)
        hidden_patches = hidden_patches.mean(dim=1)
        logits = self.output_projection(hidden_patches)

        if self.task_type == 'regression':
            prediction = logits.squeeze(-1)
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            loss = F.huber_loss(
                prediction, labels.float(), delta=1.0)
        else:
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels.argmax(dim=-1)
            loss = F.cross_entropy(logits, labels)

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'patches': enc_out,
                'reprogrammed': reprogrammed,
                'HL_patches': hidden_patches,
            },
        )
