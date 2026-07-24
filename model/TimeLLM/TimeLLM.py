from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from ..base import BaseConfig, ModelOutputs

import logging

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EEG 19-channel layout (kept as reference)
# ═══════════════════════════════════════════════════════════════════════════════

EEG_19_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7',  'T3', 'T5',
    'F8',  'T4', 'T6',
    'Fz',  'Cz', 'Pz',
]

EEG_CHANNEL_GROUPS = {
    'frontal':  [0, 5, 1, 6, 10, 13, 16],
    'temporal': [11, 14, 12, 15],
    'central':  [2, 17, 7],
    'parietal': [3, 18, 8],
    'occipital': [4, 9],
}

HOMOLOGOUS_PAIRS = [
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    (10, 13), (11, 14), (12, 15),
]


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

# ── Prompt Part 1: Dataset description ──
PROMPT_DATASET = (
    "This dataset is used for Alzheimer's disease (AD) dementia diagnosis, "
    "based on resting-state EEG data from over a thousand subjects. The raw "
    "signals are segmented into 1-second windows, and Pearson correlation "
    "coefficients are computed between each pair of channels to construct "
    "brain functional connectivity graphs. A total of 19 channels (international "
    "10-20 system) are covered, in the following order: Fp1, F3, C3, P3, O1, "
    "Fp2, F4, C4, P4, O2, F7, T3, T5, F8, T4, T6, FZ, CZ, PZ."
)

# ── Prompt Part 2: Task instruction ──
PROMPT_TASK = (
    "Given 19-channel brain functional connectivity, classify the subject "
    "as AD (Alzheimer's disease), MCI (mild cognitive impairment), "
    "SCD (subjective cognitive decline), or NC (normal cognition)."
)

# ── Prompt Part 3: Statistical features template (filled per sample) ──
PROMPT_STATS = (
    "Maximum connection: {max_pair} (r={max_val:.3f}). "
    "Minimum positive connection: {min_pos_pair} (r={min_pos_val:.3f}). "
    "Strongest negative connection: {max_neg_pair} (r={max_neg_val:.3f}). "
    "Mean intra-frontal FC: {fc_frontal:.3f}. "
    "Mean inter-hemispheric homologous FC: {fc_homologous:.3f}. "
    "Global mean FC: {mean_fc:.3f} (std: {std_fc:.3f})."
)

# ── Assembled fixed system prompt (Part 1 + Part 2 only) ──
SYSTEM_PROMPT = "\n".join([PROMPT_DATASET, PROMPT_TASK])


# ═══════════════════════════════════════════════════════════════════════════════
# GCN Layer + adjacency normalization
# ═══════════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """Single graph convolution layer.

    H^{l+1} = D^{-1/2} Â D^{-1/2} H^l W^l
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        """
        Args:
            x:        (B, N, in_features)  node features
            adj_norm: (B, N, N)            normalized adjacency with self-loops

        Returns:
            (B, N, out_features)
        """
        x = self.dropout(x)
        support = self.linear(x)
        out = torch.bmm(adj_norm, support)
        return out


def normalize_adjacency(SFC, threshold=0.0):
    """Convert static FC matrix to symmetric-normalized graph adjacency.

    A = |SFC| (with threshold, zero diagonal, self-loop)
    Ã_norm = D^{-1/2} A D^{-1/2}

    Args:
        SFC:      (B, N, N)  Pearson correlation matrix, values in [-1, 1]
        threshold: float      edges with absolute r < threshold are zeroed

    Returns:
        adj_norm: (B, N, N)  normalized adjacency matrix
    """
    B, N, _ = SFC.shape
    device = SFC.device

    adj = SFC.abs().clone()

    # Zero diagonal
    idx = torch.arange(N, device=device)
    adj[:, idx, idx] = 0

    # Threshold weak edges
    adj[adj < threshold] = 0

    # Add self-loops: Â = A + I
    adj = adj + torch.eye(N, device=device).unsqueeze(0)

    # Degree matrix
    deg = adj.sum(dim=-1)  # (B, N)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # D^{-1/2} Â D^{-1/2}
    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # (B, N, N)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    return adj_norm


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TimeLLMConfig(BaseConfig):
    def __init__(self, node_size, num_classes,
                 d_model=64,
                 n_heads=8,
                 num_prototypes=500,
                 gcn_hidden=128,
                 dropout=0.1,
                 llm_layers=28):
        super().__init__(node_size=node_size, num_classes=num_classes)
        self.d_model = d_model                # node embedding dim → ReprogrammingLayer Q
        self.n_heads = n_heads                # ReprogrammingLayer attention heads
        self.num_prototypes = num_prototypes  # number of text prototypes
        self.gcn_hidden = gcn_hidden          # GCN hidden dimension
        self.dropout = dropout
        self.llm_layers = llm_layers          # ChatGLM layers to use


# ═══════════════════════════════════════════════════════════════════════════════
# ReprogrammingLayer — cross-attention: nodes → text prototypes
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# TimeLLM v2 Model — GCN node encoder → Reprogramming → ChatGLM-6B → classify
# ═══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """TimeLLM v2 for static-FC-based dementia classification.

    Architecture:
    1. Static FC (B, 19, 19) → normalize_adjacency → (B, 19, 19)
    2. Learnable node_init (19, gcn_hidden) → GCN(adj_norm) → node_embeddings (B, 19, gcn_hidden)
    3. Node projection → (B, 19, d_model)
    4. Text prototypes from ChatGLM-6B word_embeddings via mapping_layer → (500, 4096)
    5. ReprogrammingLayer: cross-attention (Q=nodes, K/V=prototypes) → (B, 19, 4096)
    6. Add learnable node position embeddings
    7. Assemble [<start_prompt> | frozen_prefix | per-sample_stats | <end_prompt> | node_tokens]
    8. Frozen ChatGLM-6B bidirectional encoding
    9. Mean pool over node tokens only → Linear(4096 → 4) → logits
    """

    def __init__(self, config: TimeLLMConfig):
        super().__init__()
        self.config = config
        C = config.node_size               # 19 EEG channels
        self.d_model = config.d_model      # 64
        self.num_prototypes = config.num_prototypes  # 500
        self.n_heads = config.n_heads      # 8
        self.d_llm = 4096                  # ChatGLM-6B hidden_size
        self.llm_layers = config.llm_layers

        # ── 1. GCN node encoder ──
        # Learnable initial node features: one vector per EEG channel
        self.node_init = nn.Parameter(torch.randn(C, config.gcn_hidden) * 0.02)
        self.gcn = GCNLayer(config.gcn_hidden, config.gcn_hidden,
                            dropout=config.dropout)

        # Project GCN output to d_model for ReprogrammingLayer Q
        self.node_projection = nn.Sequential(
            nn.Linear(config.gcn_hidden, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # ── 2. Tokenizer + LLM (ChatGLM-6B, frozen) ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./model/chatglm-6b", trust_remote_code=True
        )
        self.llm = AutoModel.from_pretrained(
            "./model/chatglm-6b", trust_remote_code=True
        ).bfloat16()

        self.llm.transformer.gradient_checkpointing = False
        self.position_encoding_2d = self.llm.transformer.position_encoding_2d

        # ── Frozen system prompt prefix embeddings (Part 1 + Part 2) ──
        prefix_ids = self.tokenizer.encode(SYSTEM_PROMPT)
        with torch.no_grad():
            prefix_embeds = self.llm.transformer.word_embeddings(
                torch.tensor(prefix_ids)
            )  # (P_prefix, 4096)
        self.register_buffer("prompt_prefix_embeddings", prefix_embeds)
        self.P_prefix: int = prefix_embeds.shape[0]  # number of fixed prefix tokens

        # ── Frozen start / end prompt tags ──
        start_tag_ids = self.tokenizer.encode("<start_prompt>\n")
        with torch.no_grad():
            start_tag_embeds = self.llm.transformer.word_embeddings(
                torch.tensor(start_tag_ids)
            )  # (P_start, 4096)
        self.register_buffer("start_tag_embeddings", start_tag_embeds)
        self.P_start: int = start_tag_embeds.shape[0]

        end_tag_ids = self.tokenizer.encode("\n<end_prompt>")
        with torch.no_grad():
            end_tag_embeds = self.llm.transformer.word_embeddings(
                torch.tensor(end_tag_ids)
            )  # (P_end, 4096)
        self.register_buffer("end_tag_embeddings", end_tag_embeds)
        self.P_end: int = end_tag_embeds.shape[0]

        # ── 3. Text prototypes from LLM word embeddings ──
        self.word_embeddings = self.llm.transformer.word_embeddings.weight
        # ChatGLM-6B vocab_size = 150528
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_prototypes)
        # mapping_layer: Linear(150528, 500), ~75M trainable params

        # ── 4. ReprogrammingLayer: cross-attention nodes ↔ prototypes ──
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
        )

        # ── 5. Learnable node position embeddings ──
        self.node_pos_embed = nn.Parameter(torch.zeros(1, C, self.d_llm))
        nn.init.normal_(self.node_pos_embed, std=0.02)

        # ── 6. Classification head ──
        self.dropout = nn.Dropout(config.dropout)
        self.cls_head = nn.Linear(self.d_llm, config.num_classes)

        # ── 7. Freeze LLM ──
        for param in self.llm.parameters():
            param.requires_grad = False

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── Per-sample stats prompt builder ────────────────────────────

    def _build_stats_prompts(self, SFC):
        """Build per-sample statistical feature prompt from static FC.

        Args:
            SFC: (B, 19, 19) static functional connectivity matrices

        Returns:
            list[str] of length B
        """
        B, N, _ = SFC.shape

        prompts = []
        for b in range(B):
            fc = SFC[b].clone()
            fc.fill_diagonal_(0.0)

            # ── Global stats ──
            triu_idx = torch.triu_indices(N, N, offset=1)
            all_edges = fc[triu_idx[0], triu_idx[1]]
            mean_fc = float(all_edges.mean())
            std_fc = float(all_edges.std())

            # ── Extremal connections ──
            edge_values = []
            for i, j in zip(triu_idx[0].tolist(), triu_idx[1].tolist()):
                edge_values.append({
                    'i': i, 'j': j,
                    'val': fc[i, j].item(),
                    'name': f"{EEG_19_CHANNELS[i]}-{EEG_19_CHANNELS[j]}",
                })

            # Max connection (highest r)
            edge_values.sort(key=lambda e: e['val'], reverse=True)
            max_edge = edge_values[0]

            # Min positive connection
            pos_edges = [e for e in edge_values if e['val'] > 0]
            min_pos_edge = pos_edges[-1] if pos_edges else max_edge

            # Strongest negative connection
            neg_edges = [e for e in edge_values if e['val'] < 0]
            max_neg_edge = neg_edges[0] if neg_edges else edge_values[-1]

            # ── Intra-frontal FC ──
            frontal_idx = EEG_CHANNEL_GROUPS['frontal']
            frontal_vals = []
            for a in range(len(frontal_idx)):
                for k in range(a + 1, len(frontal_idx)):
                    frontal_vals.append(
                        fc[frontal_idx[a], frontal_idx[k]].item())
            fc_frontal = float(torch.tensor(frontal_vals).mean()) if frontal_vals else 0.0

            # ── Inter-hemispheric homologous FC ──
            homo_vals = [fc[i, j].item() for i, j in HOMOLOGOUS_PAIRS]
            fc_homologous = float(torch.tensor(homo_vals).mean())

            prompt = PROMPT_STATS.format(
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

    # ── Forward ────────────────────────────────────────────────────

    def forward(self, SFC, labels, gender=None, age=None, education=None):
        """
        Args:
            SFC:    (B, 19, 19)  static functional connectivity matrix
            labels: (B,) or (B, 4)  class indices or one-hot

        Returns:
            ModelOutputs with logits, loss, hidden_state
        """
        B, C, _ = SFC.shape
        device = SFC.device

        # ════════════════════════════════════════════════════════════
        # Step 1: Normalize adjacency matrix
        # ════════════════════════════════════════════════════════════
        adj_norm = normalize_adjacency(SFC, threshold=0.0)   # (B, C, C)

        # ════════════════════════════════════════════════════════════
        # Step 2: GCN → 19 node embeddings
        # ════════════════════════════════════════════════════════════
        node_init = self.node_init.unsqueeze(0).expand(B, -1, -1)  # (B, C, gcn_hidden)
        gcn_out = self.gcn(node_init, adj_norm)                     # (B, C, gcn_hidden)
        gcn_out = F.gelu(gcn_out)

        # Project to d_model for ReprogrammingLayer Q
        node_embeddings = self.node_projection(gcn_out)             # (B, C, d_model)

        # ════════════════════════════════════════════════════════════
        # Step 3: Text prototypes from LLM word embeddings
        # ════════════════════════════════════════════════════════════
        # word_embeddings is bfloat16 (frozen LLM), mapping_layer is float32.
        # Cast to float32 for the linear, then back to bfloat16 for reprogramming.
        we = self.word_embeddings.permute(1, 0).to(
            dtype=self.mapping_layer.weight.dtype)
        source_embeddings = self.mapping_layer(we).permute(1, 0)  # (N_proto, 4096)
        source_embeddings = source_embeddings.to(
            dtype=self.word_embeddings.dtype)

        # ════════════════════════════════════════════════════════════
        # Step 4: Reprogramming — cross-attention
        #   Q: node embeddings (B, C, d_model)  [float32]
        #   K,V: text prototypes (N_proto, 4096)  [coming from mapping_layer]
        #   ReprogrammingLayer runs in float32; cast output to bfloat16 for LLM.
        # ════════════════════════════════════════════════════════════
        # Cast prototypes to float32 for ReprogrammingLayer (its weights are fp32)
        reprogrammed = self.reprogramming_layer(
            node_embeddings.to(dtype=torch.float32),
            source_embeddings.to(dtype=torch.float32),
            source_embeddings.to(dtype=torch.float32),
        )                                                    # (B, C, 4096)  float32

        # Add learnable node position embeddings, then convert to bfloat16 for LLM
        reprogrammed = reprogrammed + self.node_pos_embed     # (B, C, 4096)
        reprogrammed = reprogrammed.to(dtype=torch.bfloat16)

        # ════════════════════════════════════════════════════════════
        # Step 5: Build per-sample stats prompt & assemble
        #   [<start_prompt> | prefix (Part1+2) | stats (Part3) | <end_prompt> | node_tokens]
        # ════════════════════════════════════════════════════════════
        stats_texts = self._build_stats_prompts(SFC)
        stats_ids = self.tokenizer(
            stats_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=256
        ).input_ids.to(device)
        stats_embeddings = self.llm.transformer.word_embeddings(
            stats_ids)                                       # (B, P_stats, 4096)

        start_tag = self.start_tag_embeddings.unsqueeze(0) \
                        .expand(B, -1, -1)                   # (B, P_start, 4096)
        end_tag = self.end_tag_embeddings.unsqueeze(0) \
                      .expand(B, -1, -1)                     # (B, P_end, 4096)
        prompt_prefix = self.prompt_prefix_embeddings.unsqueeze(0) \
                            .expand(B, -1, -1)                # (B, P_prefix, 4096)

        inputs_embeds = torch.cat(
            [start_tag,
             prompt_prefix,
             stats_embeddings.to(dtype=prompt_prefix.dtype),
             end_tag,
             reprogrammed.to(dtype=prompt_prefix.dtype)],
            dim=1,
        )                                                    # (B, total, 4096)
        S = inputs_embeds.shape[1]
        P_skip = (self.P_start + self.P_prefix +
                  stats_embeddings.shape[1] + self.P_end)     # prompt tokens to skip

        # Bidirectional attention: all False = full visibility
        attention_mask = torch.zeros(B, 1, S, S, dtype=torch.bool,
                                     device=device)

        # Position IDs (matching ChatGLM-6B 2D position encoding convention)
        position_ids = torch.arange(S, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).repeat(B, 1)
        if self.position_encoding_2d:
            block_position_ids = torch.arange(S, dtype=torch.long,
                                              device=device).unsqueeze(0).repeat(B, 1)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)

        transformer_outputs = self.llm.transformer(
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
        # transformer returns [seq_len, batch, hidden_size]
        HL = transformer_outputs[0].transpose(0, 1)           # → (B, P_skip+19, 4096)

        # ════════════════════════════════════════════════════════════
        # Step 6: Mean pool over node tokens only (skip all prompt) → classify
        # ════════════════════════════════════════════════════════════
        # HL is bfloat16; cast to float32 for dropout & cls_head (float32 layers)
        llm_pooled = HL[:, P_skip:, :].mean(dim=1).to(dtype=torch.float32)
        llm_pooled = self.dropout(llm_pooled)
        logits = self.cls_head(llm_pooled)                    # (B, num_classes)

        # Handle labels: convert from one-hot if needed
        if labels.dim() > 1 and labels.shape[-1] > 1:
            labels = labels.argmax(dim=-1)
        loss = F.cross_entropy(logits, labels)

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'gcn_out': gcn_out,
                'reprogrammed': reprogrammed,
                'llm_pooled': llm_pooled,
            }
        )
