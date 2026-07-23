from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from ..base import BaseConfig, ModelOutputs
from ..LDDE2th.LDDE2thLayers import BrainNetCNN

import logging

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Prompt Template
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = (
    "<|start_prompt|>Dataset: Dementia4000 resting-state EEG, 19 channels (10-20 system). "
    "Task: Classify the patient into one of 4 categories (0=NC, 1=SMI, 2=MCI, 3=AD) "
    "based on DFC patterns.\n"
    "Input DFC statistics:\n"
    "- {L} DFC windows after stride={stride} subsampling\n"
    "- Global mean FC: {mean_fc:.3f}, std across edges: {std_fc:.3f}\n"
    "- Positive connections (r>0): {pos_ratio:.1f}%, strong connections (|r|>0.5): {strong_ratio:.1f}%\n"
    "- Weighted clustering coefficient: {cluster_coef:.3f}, "
    "characteristic path length: {path_length:.2f}\n"
    "- Mean temporal variability (DFC std across windows): {temporal_std:.3f}\n"
    "- Most stable connection: {stable_pair} (CV={stable_cv:.3f})\n"
    "- Most unstable connection: {unstable_pair} (CV={unstable_cv:.3f})\n"
    "- Intra-frontal FC: {fc_frontal:.3f}, intra-temporal FC: {fc_temporal:.3f}\n"
    "- Intra-central/SMN FC: {fc_central:.3f}, intra-parietal FC: {fc_parietal:.3f}\n"
    "- Intra-occipital/VIS FC: {fc_occipital:.3f}\n"
    "- Parieto-temporal FC (DMN posterior proxy): {fc_parieto_temporal:.3f}\n"
    "- Inter-hemispheric homologous FC: {fc_homologous:.3f}\n"
    "<|end_prompt|>"
)

# EEG 19-channel layout
EEG_19_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7',  'T3', 'T5',
    'F8',  'T4', 'T6',
    'Fz',  'Cz', 'Pz',
]

EEG_CHANNEL_GROUPS = {
    'frontal':  [0, 5, 1, 6, 10, 13, 16],   # Fp1,Fp2,F3,F4,F7,F8,Fz
    'temporal': [11, 14, 12, 15],              # T3,T4,T5,T6
    'central':  [2, 17, 7],                     # C3,Cz,C4
    'parietal': [3, 18, 8],                     # P3,Pz,P4
    'occipital': [4, 9],                        # O1,O2
}

# Homologous pairs (left-right symmetric)
HOMOLOGOUS_PAIRS = [
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    (10, 13), (11, 14), (12, 15),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TimeLLMConfig(BaseConfig):
    def __init__(self, node_size, num_classes,
                 d_model=64,
                 n_heads=8,
                 d_ff=128,
                 num_prototypes=500,
                 patch_stride=5,
                 dropout=0.1,
                 llm_layers=28):
        super().__init__(node_size=node_size, num_classes=num_classes)
        self.d_model = d_model          # patch feature dim (Q input to Reprogramming)
        self.n_heads = n_heads          # ReprogrammingLayer attention heads
        self.d_ff = d_ff                # unused, kept for compat
        self.num_prototypes = num_prototypes  # N of text prototypes
        self.patch_stride = patch_stride      # DFC temporal subsampling stride
        self.dropout = dropout
        self.llm_layers = llm_layers    # ChatGLM layers to use (all 28)


# ═══════════════════════════════════════════════════════════════════════════════
# ReprogrammingLayer — cross-attention: time-series patches → text prototypes
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
# TimeLLM Model — BNC patch encoder → Reprogramming → ChatGLM-6B → classify
# ═══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """Time-LLM for DFC-based dementia classification.

    Architecture (based on Time-LLM ICLR 2024):
    1. DFC temporal subsampling (stride)
    2. BrainNetCNN per-window graph encoding → (B, L, 256)
    3. Linear projection → (B, L, d_model)
    4. Text prototypes from ChatGLM-6B word_embeddings via mapping_layer
    5. ReprogrammingLayer: cross-attention patches ↔ prototypes → (B, L, 4096)
    6. Prompt assembly: [dynamic_statistics_prompt | reprogrammed_patches]
    7. Frozen ChatGLM-6B encoding
    8. Mean-pool patch hidden states → Linear classifier → 4 classes
    """

    def __init__(self, config: TimeLLMConfig):
        super().__init__()
        self.config = config
        C = config.node_size          # 19
        self.stride = config.patch_stride  # 5
        self.d_model = config.d_model      # 64
        self.num_prototypes = config.num_prototypes  # 500
        self.n_heads = config.n_heads       # 8
        self.d_llm = 4096                   # ChatGLM-6B hidden_size
        self.llm_layers = config.llm_layers  # 28

        # ── 1. BrainNetCNN — DFC graph patch encoder ──
        self.bnc = BrainNetCNN(C)
        # Project BNC output (256) → d_model for ReprogrammingLayer Q input
        self.patch_projection = nn.Sequential(
            nn.Linear(256, config.d_model),
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

        # LLM gradient checkpointing off
        self.llm.transformer.gradient_checkpointing = False

        # ── 2. Text prototypes from LLM word embeddings ──
        self.word_embeddings = self.llm.transformer.word_embeddings.weight
        # ChatGLM-6B vocab_size = 150528
        self.vocab_size = self.word_embeddings.shape[0]  # 150528
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_prototypes)
        # mapping_layer: Linear(150528, 500), weight = (500, 150528) ≈ 75M params

        # ── 4. ReprogrammingLayer — cross-attention ──
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_llm=self.d_llm,
            attention_dropout=config.dropout,
        )

        # ── 5. Classification head ──
        self.dropout = nn.Dropout(config.dropout)
        self.cls_head = nn.Linear(self.d_llm, config.num_classes)

        # ── 6. Freeze LLM ──
        for param in self.llm.parameters():
            param.requires_grad = False

        # Set pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── Forward ────────────────────────────────────────────────────
    def forward(self, time_series, DFC, labels,
                gender=None, age=None, education=None):
        """
        Args:
            time_series: (B, 19, Ts)   EEG time series
            DFC:         (B, L_full, 19, 19)  dynamic FC matrices
            labels:      (B,)           class indices (int)

        Returns:
            ModelOutputs with logits, loss, hidden_state
        """
        B, L_full, C, _ = DFC.shape
        device = DFC.device

        # ════════════════════════════════════════════════════════════
        # Step 0: DFC temporal subsampling
        # ════════════════════════════════════════════════════════════
        DFC = DFC[:, ::self.stride, :, :]              # (B, L, 19, 19)
        L = DFC.shape[1]

        # ════════════════════════════════════════════════════════════
        # Step 1: BrainNetCNN graph patch encoding
        # ════════════════════════════════════════════════════════════
        DFC_flat = DFC.reshape(B * L, 1, C, C)          # (B*L, 1, 19, 19)
        DFC_flat = DFC_flat.to(dtype=next(self.bnc.parameters()).dtype)
        F_per_window = self.bnc(DFC_flat)                # (B*L, 256)
        F_per_window = F_per_window.reshape(B, L, 256)   # (B, L, 256)

        # Project to d_model space
        patches = self.patch_projection(F_per_window)     # (B, L, d_model)

        # ════════════════════════════════════════════════════════════
        # Step 2: Build dynamic prompt from DFC statistics
        # ════════════════════════════════════════════════════════════
        prompt_texts = self._build_dynamic_prompts(DFC, L)
        prompt_tokens = self.tokenizer(
            prompt_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=2048
        ).input_ids.to(device)
        prompt_embeddings = self.llm.transformer.word_embeddings(
            prompt_tokens)                                 # (B, P, 4096)
        P = prompt_embeddings.shape[1]

        # ════════════════════════════════════════════════════════════
        # Step 3: Text prototypes from LLM word embeddings
        # ════════════════════════════════════════════════════════════
        # word_embeddings: (vocab_size, 4096)
        # → permute → (4096, vocab_size)
        # → mapping_layer (Linear(vocab_size, num_prototypes)) → (4096, num_prototypes)
        # → permute → (num_prototypes, 4096)
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)                                  # (N_proto, 4096)

        # ════════════════════════════════════════════════════════════
        # Step 4: Reprogramming (cross-attention)
        #   Q: patch embeddings (B, L, d_model)
        #   K,V: text prototypes (N_proto, 4096)
        # → (B, L, 4096)
        # ════════════════════════════════════════════════════════════
        reprogrammed = self.reprogramming_layer(
            patches.to(dtype=source_embeddings.dtype),
            source_embeddings,
            source_embeddings,
        )                                                 # (B, L, 4096)

        # ════════════════════════════════════════════════════════════
        # Step 5: Assemble LLM input
        #   [dynamic_prompt | reprogrammed_patches]
        # ════════════════════════════════════════════════════════════
        inputs_embeds = torch.cat(
            [prompt_embeddings.to(dtype=reprogrammed.dtype),
             reprogrammed],
            dim=1,
        )                                                 # (B, P+L, 4096)

        # ════════════════════════════════════════════════════════════
        # Step 6: Frozen ChatGLM-6B encoding
        # ════════════════════════════════════════════════════════════
        # forward_from_embeds returns [seq_len, batch, hidden_size]
        HL = self.llm.forward_from_embeds(inputs_embeds)
        HL = HL.transpose(0, 1)              # → (B, S+P+L, 4096)

        # ════════════════════════════════════════════════════════════
        # Step 7: Pool patch hidden states → classify
        # ════════════════════════════════════════════════════════════
        # Take the last L positions (corresponding to DFC patches)
        patch_hidden = HL[:, -L:, :]          # (B, L, 4096)
        pooled = patch_hidden.mean(dim=1)     # (B, 4096)
        pooled = self.dropout(pooled)
        logits = self.cls_head(pooled)        # (B, 4)

        # Handle labels: convert from one-hot if needed
        if labels.dim() > 1 and labels.shape[-1] > 1:
            labels = labels.argmax(dim=-1)
        loss = F.cross_entropy(logits, labels)

        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_state={
                'patches': patches,
                'reprogrammed': reprogrammed,
                'llm_pooled': pooled,
            }
        )

    # ── Dynamic Prompt Building ────────────────────────────────────
    def _build_dynamic_prompts(self, DFC, L):
        """Build per-sample dynamic prompt texts from DFC statistics.

        Args:
            DFC: (B, L, 19, 19) subsampled DFC matrices
            L:   number of windows

        Returns:
            list[str] of length B
        """
        B = DFC.shape[0]
        device = DFC.device
        N = 19  # channels

        # Time-averaged FC matrix
        fc_mean = DFC.mean(dim=1)              # (B, 19, 19)
        # Time std (temporal variability)
        fc_std = DFC.std(dim=1)                # (B, 19, 19)

        prompts = []
        for b in range(B):
            fc = fc_mean[b].clone()            # (19, 19)
            fc_std_b = fc_std[b].clone()       # (19, 19)

            # Zero diagonal
            fc.fill_diagonal_(0.0)
            fc_std_b.fill_diagonal_(0.0)

            # ── Global stats ──
            triu_idx = torch.triu_indices(N, N, offset=1)
            all_edges = fc[triu_idx[0], triu_idx[1]]   # 171 edges
            mean_fc = float(all_edges.mean())
            std_fc = float(all_edges.std())
            pos_ratio = float((all_edges > 0).float().mean()) * 100
            strong_ratio = float((all_edges.abs() > 0.5).float().mean()) * 100

            # ── Graph metrics ──
            cluster_coef, path_length = self._graph_metrics(fc)

            # ── Temporal variability ──
            all_std = fc_std_b[triu_idx[0], triu_idx[1]]
            temporal_std = float(all_std.mean())

            # Most stable / unstable connections
            edges_cv = []
            for i in range(N):
                for j in range(i + 1, N):
                    mu = abs(fc[i, j].item())
                    sigma = fc_std_b[i, j].item()
                    if mu > 0.05 and sigma > 0:
                        edges_cv.append({
                            'i': i, 'j': j,
                            'mu': mu, 'sigma': sigma,
                            'cv': sigma / mu,
                        })
            if edges_cv:
                edges_cv.sort(key=lambda e: e['cv'])
                stable = edges_cv[0]
                unstable = edges_cv[-1]
                stable_pair = f"{EEG_19_CHANNELS[stable['i']]}-{EEG_19_CHANNELS[stable['j']]}"
                unstable_pair = f"{EEG_19_CHANNELS[unstable['i']]}-{EEG_19_CHANNELS[unstable['j']]}"
                stable_cv = stable['cv']
                unstable_cv = unstable['cv']
            else:
                stable_pair, unstable_pair = "N/A", "N/A"
                stable_cv, unstable_cv = 0.0, 0.0

            # ── Network-level FC ──
            def _intra_fc(indices):
                vals = []
                for a in range(len(indices)):
                    for b in range(a + 1, len(indices)):
                        vals.append(fc[indices[a], indices[b]].item())
                return float(torch.tensor(vals).mean()) if vals else 0.0

            fc_frontal = _intra_fc(EEG_CHANNEL_GROUPS['frontal'])
            fc_temporal = _intra_fc(EEG_CHANNEL_GROUPS['temporal'])
            fc_central = _intra_fc(EEG_CHANNEL_GROUPS['central'])
            fc_parietal = _intra_fc(EEG_CHANNEL_GROUPS['parietal'])
            fc_occipital = _intra_fc(EEG_CHANNEL_GROUPS['occipital'])

            # Parieto-temporal cross-network (DMN posterior proxy)
            pt_vals = []
            for pi in EEG_CHANNEL_GROUPS['parietal']:
                for ti in EEG_CHANNEL_GROUPS['temporal']:
                    pt_vals.append(fc[pi, ti].item())
            fc_parieto_temporal = float(torch.tensor(pt_vals).mean()) if pt_vals else 0.0

            # Homologous inter-hemispheric FC
            homo_vals = [fc[i, j].item() for i, j in HOMOLOGOUS_PAIRS]
            fc_homologous = float(torch.tensor(homo_vals).mean())

            prompt = PROMPT_TEMPLATE.format(
                L=L,
                stride=self.stride,
                mean_fc=mean_fc,
                std_fc=std_fc,
                pos_ratio=pos_ratio,
                strong_ratio=strong_ratio,
                cluster_coef=cluster_coef,
                path_length=path_length,
                temporal_std=temporal_std,
                stable_pair=stable_pair,
                stable_cv=stable_cv,
                unstable_pair=unstable_pair,
                unstable_cv=unstable_cv,
                fc_frontal=fc_frontal,
                fc_temporal=fc_temporal,
                fc_central=fc_central,
                fc_parietal=fc_parietal,
                fc_occipital=fc_occipital,
                fc_parieto_temporal=fc_parieto_temporal,
                fc_homologous=fc_homologous,
            )
            prompts.append(prompt)

        return prompts

    @staticmethod
    def _graph_metrics(fc, density=0.25):
        """Compute weighted clustering coefficient and characteristic path length.

        Uses proportional threshold (top density |r| edges) rather than r>0 binary.

        Args:
            fc: (19, 19) FC matrix
            density: fraction of edges to retain (default 0.25 = top 25%)

        Returns:
            (clustering_coef, characteristic_path_length)
        """
        N = fc.shape[0]
        fc_abs = fc.abs().clone()
        fc_abs.fill_diagonal_(0.0)

        k = max(1, int(torch.ceil(torch.tensor(N * density)).item()))
        adj = torch.zeros(N, N, dtype=fc.dtype, device=fc.device)

        for i in range(N):
            row = fc_abs[i]
            _, top_idx = torch.topk(row, k)
            for j in top_idx:
                if row[j] > 0:
                    adj[i, j] = row[j]
        # Symmetrize
        adj = (adj + adj.T) / 2

        # ── Weighted clustering coefficient (Onnela 2005) ──
        degrees = (adj > 0).sum(dim=1).float()
        C_per_node = torch.zeros(N, dtype=fc.dtype, device=fc.device)
        for i in range(N):
            neighbors = torch.where(adj[i] > 0)[0]
            if len(neighbors) < 2:
                continue
            tri_sum = 0.0
            for a_idx, j in enumerate(neighbors):
                for k in neighbors[a_idx + 1:]:
                    if adj[j, k] > 0:
                        tri_sum += (adj[i, j] * adj[j, k] * adj[k, i]) ** (1.0 / 3.0)
            denom = degrees[i] * (degrees[i] - 1)
            C_per_node[i] = tri_sum / denom if denom > 0 else 0.0

        valid = degrees >= 2
        C = float(C_per_node[valid].mean()) if valid.any() else 0.0

        # ── Weighted characteristic path length ──
        dist = torch.full((N, N), float('inf'), device=fc.device)
        dist.fill_diagonal_(0)
        for i in range(N):
            for j in range(N):
                if adj[i, j] > 0:
                    dist[i, j] = 1.0 / adj[i, j]

        # Floyd-Warshall
        for kk in range(N):
            dk = dist[:, kk:kk + 1] + dist[kk:kk + 1, :]
            dist = torch.minimum(dist, dk)

        finite = dist[torch.isfinite(dist)]
        L_char = float(finite.mean()) if len(finite) > 0 else 0.0

        return C, L_char
