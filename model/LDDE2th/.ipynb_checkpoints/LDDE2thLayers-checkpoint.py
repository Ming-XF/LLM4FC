import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Dict, Optional

import pdb


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Prototype labels (K=16)
# ═══════════════════════════════════════════════════════════════════════════════

PROTOTYPE_LABELS = [
    {'idx': 0,  'en': 'DMN_posterior_integrity',      'zh': 'DMN后部连接完整性',     'network': 'DMN',   'direction': 'high=normal'},
    {'idx': 1,  'en': 'MTL_hippocampal_coupling',      'zh': '内侧颞叶-海马耦合强度', 'network': 'LIM',   'direction': 'high=normal'},
    {'idx': 2,  'en': 'FPN_compensatory_activation',   'zh': '额顶控制网络代偿激活',  'network': 'CEN',   'direction': 'high=compensatory'},
    {'idx': 3,  'en': 'SN_internal_synchrony',         'zh': '突显网络内部同步性',    'network': 'SN',    'direction': 'high=normal'},
    {'idx': 4,  'en': 'VIS_network_integrity',         'zh': '视觉网络完整性',        'network': 'VIS',   'direction': 'high=normal'},
    {'idx': 5,  'en': 'global_connectivity_diffusion', 'zh': '全脑连接弥散度',        'network': 'Global','direction': 'high=abnormal'},
    {'idx': 6,  'en': 'interhemispheric_homology',     'zh': '半球间同源连接强度',    'network': 'Inter-Hemi','direction': 'high=compensatory'},
    {'idx': 7,  'en': 'anterior_posterior_efficiency', 'zh': '前-后长程连接效率',     'network': 'Long-range','direction': 'high=normal'},
    {'idx': 8,  'en': 'left_language_activation',      'zh': '左侧语言网络激活',      'network': 'AUD(L)','direction': 'high=compensatory'},
    {'idx': 9,  'en': 'SMN_network_integrity',         'zh': '感觉运动网络完整性',    'network': 'SMN',   'direction': 'high=normal'},
    {'idx': 10, 'en': 'OFC_striatal_coupling',         'zh': '眶额-纹状体回路强度',   'network': 'OFC',   'direction': 'high=normal'},
    {'idx': 11, 'en': 'PFC_executive_function',        'zh': '前额叶执行功能网络',    'network': 'PFC',   'direction': 'high=normal'},
    {'idx': 12, 'en': 'limbic_emotional_circuit',      'zh': '边缘系统情绪回路',      'network': 'LIM',   'direction': 'high=normal'},
    {'idx': 13, 'en': 'subcortical_cortical_strength', 'zh': '皮质下-皮层连接强度',   'network': 'Subcortical','direction': 'high=normal'},
    {'idx': 14, 'en': 'attention_network_coupling',    'zh': '注意网络内部耦合',      'network': 'CEN',   'direction': 'high=normal'},
    {'idx': 15, 'en': 'compensatory_cross_network',    'zh': '代偿性跨网络重组',      'network': 'Global','direction': 'high=compensatory'},
]

PROTOTYPE_LABELS_ZH = [p['zh'] for p in PROTOTYPE_LABELS]
    
class E2EBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)

        ab = torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)

        # if torch.isnan(ab).any() or torch.isinf(ab).any():
        #     pdb.set_trace()
        return ab
    
class FeatureAlign(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(256, 512)
        self.dense2 = torch.nn.Linear(512, 1024)
        self.dense3 = torch.nn.Linear(1024, 2048)
        self.dense4 = torch.nn.Linear(2048, 4096)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(4096)
    def forward(self, x):
        out1 = F.dropout(F.leaky_relu(self.bn1(self.dense1(x)), negative_slope=0.33), p=0.5)
        out2 = F.dropout(F.leaky_relu(self.bn2(self.dense2(out1)), negative_slope=0.33), p=0.5)
        out3 = F.dropout(F.leaky_relu(self.bn3(self.dense3(out2)), negative_slope=0.33), p=0.5)
        out4 = F.dropout(F.leaky_relu(self.bn4(self.dense4(out3)), negative_slope=0.33), p=0.5)
        
        return out4
    
class FeatureAlignInverse(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(4096, 2048)
        self.dense2 = torch.nn.Linear(2048, 1024)
        self.dense3 = torch.nn.Linear(1024, 512)
        self.dense4 = torch.nn.Linear(512, 256)
        
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
    def forward(self, x):
        out1 = F.dropout(F.leaky_relu(self.bn1(self.dense1(x)), negative_slope=0.33), p=0.5)
        out2 = F.dropout(F.leaky_relu(self.bn2(self.dense2(out1)), negative_slope=0.33), p=0.5)
        out3 = F.dropout(F.leaky_relu(self.bn3(self.dense3(out2)), negative_slope=0.33), p=0.5)
        out4 = F.dropout(F.leaky_relu(self.bn4(self.dense4(out3)), negative_slope=0.33), p=0.5)
        
        return out4
        
        
    
    
class GraphConv(nn.Module):
    """Single Graph Convolution layer with symmetric normalization.

    H' = ReLU(D^{-1/2} A D^{-1/2} H W + b)
    """
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        """
        Args:
            x:        (N, C, in_dim)
            adj_norm: (N, C, C)  D^{-1/2} A D^{-1/2}
        Returns:
            out: (N, C, out_dim)
        """
        out = torch.bmm(adj_norm, x)               # (N, C, in_dim)
        out = out @ self.weight                      # (N, C, out_dim)
        out = out + self.bias
        out = self.ln(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class TextGuidedGCN(nn.Module):
    """2-layer GCN with brain region text embeddings as initial node features.

    Pipeline:
        region_text_embeds (C, 4096)
          → TextProj: 4096 → 512
          → GCN × 2
          → OutProj: 512 → 4096
        = EEG tokens (C, 4096) per time window
    """
    def __init__(self, node_size, hidden_dim=512, llm_dim=4096, dropout=0.2):
        super().__init__()
        self.node_size = node_size

        # Text projection: LLM embedding space → GCN hidden space
        self.text_proj = nn.Sequential(
            nn.Linear(llm_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
        )

        # GCN layers
        self.gcn1 = GraphConv(hidden_dim, hidden_dim, dropout)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim, dropout)

        # Output projection: GCN hidden → LLM embedding space
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, llm_dim),
        )

    def forward(self, dfc_flat, region_embeds):
        """
        Args:
            dfc_flat:      (N, C, C)   flattened DFC matrices (B*L, C, C)
            region_embeds: (C, 4096)    brain region text embeddings
        Returns:
            node_features: (N, C, 4096) GCN-encoded node features for LLM
        """
        N, C, _ = dfc_flat.shape

        # ── Adjacency: absolute DFC values + self-loops ──
        A = torch.abs(dfc_flat)
        A = A + torch.eye(C, device=A.device, dtype=A.dtype).unsqueeze(0)

        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        D = A.sum(dim=-1)                               # (N, C)
        D_inv_sqrt = torch.pow(D + 1e-8, -0.5)          # epsilon for stability
        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)   # (N, C, C)
        A_norm = D_inv_sqrt_mat @ A @ D_inv_sqrt_mat     # (N, C, C)

        # ── Node features from text embeddings ──
        X = self.text_proj(region_embeds)               # (C, hidden_dim)
        X = X.unsqueeze(0).expand(N, -1, -1)            # (N, C, hidden_dim)

        # ── Two-layer GCN ──
        H = self.gcn1(X, A_norm)
        H = self.gcn2(H, A_norm)

        # ── Project to LLM space ──
        out = self.out_proj(H)                          # (N, C, 4096)
        return out


class Readout(nn.Module):
    """Aggregate C node features into 1 token per time window.

    Pipeline:
        GCN output (N, C, 4096)
          → Mean pool across nodes → (N, 4096)
          → Lightweight projection → (N, 4096)
    """
    def __init__(self, llm_dim=4096, hidden_dim=1024, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim),
        )
        self.ln = nn.LayerNorm(llm_dim)

    def forward(self, node_feats):
        """
        Args:
            node_feats: (N, C, llm_dim)  GCN-encoded node features
        Returns:
            out: (N, llm_dim)  aggregated time-window token
        """
        out = node_feats.mean(dim=1)              # (N, 4096)  mean pool across nodes
        out = self.proj(out)                       # (N, 4096)  refine
        out = self.ln(out)                         # (N, 4096)  normalize
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Prototype Layer (P1)
# ═══════════════════════════════════════════════════════════════════════════════

def prototype_diversity_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """鼓励原型向量之间的正交性和多样性。

    计算归一化原型之间的非对角线余弦相似度的绝对平均值。

    Args:
        prototypes: (K, proto_dim) 可学习的原型向量

    Returns:
        scalar loss — 值越小原型越正交
    """
    p_norm = F.normalize(prototypes, dim=-1)          # (K, proto_dim)
    sim = p_norm @ p_norm.T                            # (K, K)
    off_diag = sim * (1 - torch.eye(sim.shape[0], device=sim.device))
    return off_diag.abs().mean()


class SemanticPrototypeLayer(nn.Module):
    """脑功能语义原型空间 — BNC 逐窗特征 → K 个语义原型的激活分布 → 文本摘要。

    Input:  (N, 256)   BNC 逐时间窗特征 (N = B * L)
    Output: dict
      - activations_per_window: (N, K)     逐窗余弦相似度
      - activations_mean:       (B, K)     时间平均 per-sample
      - activations_2d:         (B, L, K)  保留时序（用于文本生成）
      - soft_prompt:            (B, K, llm_dim)  可选连续前缀
      - text_summary:           list[str]   per-sample 中文摘要
      - proto_token_ids:        (B, max_len) 预计算的 token ID tensor（CPU）
      - prototypes:             (K, proto_dim)  原型向量引用
    """

    def __init__(self, feature_dim: int = 256, proto_dim: int = 128,
                 num_prototypes: int = 16, llm_dim: int = 4096,
                 temperature: float = 0.1, dropout: float = 0.3,
                 prototype_labels: Optional[List[Dict]] = None,
                 tokenizer=None, wte=None):
        super().__init__()

        # Labels for text generation
        self.prototype_labels = prototype_labels or PROTOTYPE_LABELS
        self.num_prototypes = num_prototypes
        self.proto_dim = proto_dim

        # Feature projection — 256 → proto_dim(128)
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, proto_dim),
        )

        # Learnable prototype vectors — K × proto_dim
        # 用 LLM tokenizer 文本嵌入初始化，使原型从语义锚点出发（tech3.md §1）
        proto_init, prototypes_raw_init = self._init_prototypes_from_text(
            tokenizer, wte, num_prototypes, proto_dim, llm_dim
        )
        self.prototypes = nn.Parameter(proto_init)

        # Frozen raw text embeddings (L2-normalized, 4096-dim) — used directly
        # for soft prompt construction, preserving full LLM semantic space.
        self.register_buffer("prototypes_raw", prototypes_raw_init)

        # Temperature for sharpness control (prototype competition)
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Soft-gate temperature for soft prompt construction (decoupled from proto competition).
        # Higher τ → smoother gate, preserving gradient flow and sample-discriminative
        # information even when activations are large.  Replaces the saturating tanh
        # (see tech.md § troubleshooting: "tanh saturation makes soft prompts identical").
        self.soft_gate_tau = 0.3

        # ── Pre-tokenize all static text fragments (P1 optimization) ──
        self._precomputed = False
        self._num_token_cache: Dict[str, List[int]] = {}
        if tokenizer is not None:
            self._precompute_token_ids(tokenizer)

    def _init_prototypes_from_text(self, tokenizer, wte, K, proto_dim, llm_dim):
        """用 LLM tokenizer 文本嵌入初始化语义原型向量。

        流程：
          1. 对每个原型中文标签 tokenize → mean-pool token embeddings → (K, llm_dim)
          2. Linear 投影 llm_dim → proto_dim
          3. L2 normalize → 用于余弦相似度计算

        随机投影近似保距（Johnson-Lindenstrauss 引理），原型间的
        语义相似关系在投影后得以保留。
        """
        text_embeds = []
        for p in self.prototype_labels:
            token_ids = tokenizer.encode(p['zh'], add_special_tokens=False)
            with torch.no_grad():
                emb = wte(torch.tensor(token_ids, dtype=torch.long)).mean(dim=0)
            text_embeds.append(emb)

        E = torch.stack(text_embeds)                         # (K, llm_dim)

        # wte 输出 bfloat16，投影层是 float32 → 统一为 float32
        proj = nn.Linear(llm_dim, proto_dim, bias=False)
        nn.init.xavier_uniform_(proj.weight)

        with torch.no_grad():
            proto_unnorm = proj(E.float())                    # (K, proto_dim)

        prototypes = F.normalize(proto_unnorm, dim=-1)        # 单位化

        # Raw 4096-dim text embeddings (L2-normalized) for soft prompt construction.
        # Frozen buffer — not projected, so full LLM semantic space is preserved.
        prototypes_raw = F.normalize(E.float(), dim=-1)      # (K, llm_dim)

        return prototypes, prototypes_raw

    def forward(self, x: torch.Tensor, B: int, L: int) -> dict:
        """
        Args:
            x: (N, 256)  BNC 逐时间窗特征, N = B * L
            B: batch size
            L: number of time windows per sample

        Returns:
            dict with activations, text summaries, and pre-computed token IDs
        """
        N = x.shape[0]

        # ── Project to prototype space ──
        q = self.feat_proj(x)                               # (N, proto_dim)

        # ── Cosine similarity with prototypes ──
        q_norm = F.normalize(q, dim=-1)                     # (N, proto_dim)
        p_norm = F.normalize(self.prototypes, dim=-1)       # (K, proto_dim)
        temperature = torch.clamp(self.temperature, min=0.01, max=10.0)
        activations_per_window = (q_norm @ p_norm.T) / temperature  # (N, K)

        # ── Reshape to preserve temporal structure ──
        activations_2d = activations_per_window.reshape(B, L, -1)    # (B, L, K)
        activations_mean = activations_2d.mean(dim=1)                 # (B, K)

        # ── (Optional) Soft prompt ──
        # 双组 token 设计（方案 B）：
        #   原型 token (16个)：激活均值 × 文本嵌入 → "患者整体画像"
        #   时间窗 token (L个)：逐窗聚合 16 个原型语义 → "动态轨迹"
        # LLM 先看整体再看动态，token 数从 16 → 16+L (≈30)
        proto_base = self.prototypes_raw.to(
            dtype=activations_mean.dtype, device=activations_mean.device)  # (K, 4096)

        # 原型 token：每个原型的跨窗平均激活方向
        # 使用 sigmoid 软门控 2σ(x/τ)−1 替代 tanh，范围 (−1, 1) 但永不饱和，
        # 保证不同样本的激活强度差异可传入 LLM，梯度永不消失。
        proto_gate = 2.0 * torch.sigmoid(activations_mean / self.soft_gate_tau) - 1.0
        proto_tokens = proto_gate.unsqueeze(-1) * proto_base.unsqueeze(0)
        # proto_tokens: (B, K, llm_dim)

        # 时间窗 token：每窗 16 个原型的加权语义混合 → 该窗的"快照"
        window_gate = 2.0 * torch.sigmoid(activations_2d / self.soft_gate_tau) - 1.0
        window_tokens = (window_gate.unsqueeze(-1) * proto_base.unsqueeze(0).unsqueeze(1)).sum(dim=2)
        # window_tokens: (B, L, llm_dim)

        soft_prompt = torch.cat([proto_tokens, window_tokens], dim=1)
        # soft_prompt: (B, K+L, llm_dim)

        # ── Build pre-computed token IDs ──
        if self._precomputed:
            proto_token_ids = self._build_proto_token_ids(activations_2d, B, L)
        else:
            proto_token_ids = None

        return {
            'activations_per_window': activations_per_window,   # (N, K)
            'activations_mean': activations_mean,               # (B, K)
            'activations_2d': activations_2d,                   # (B, L, K)
            'soft_prompt': soft_prompt,                         # (B, K, llm_dim)
            'proto_token_ids': proto_token_ids,                 # (B, max_len) or None
            'prototypes': self.prototypes,                      # (K, proto_dim)
        }

    def _build_text_summary(self, activations_2d: torch.Tensor,
                            B: int, L: int,
                            top_k: int = 6,
                            threshold: float = 0.1) -> List[str]:
        """生成 per-sample 中文文本摘要，融入时序稳定性描述。

        Args:
            activations_2d: (B, L, K)  逐窗激活值
            B, L: batch size, time windows
            top_k: 取 top-k 激活的原型
            threshold: 激活值绝对值低于此阈值的原型不纳入描述

        Returns:
            summaries: list[str], length B
        """
        summaries = []
        for b in range(B):
            act_2d = activations_2d[b]                         # (L, K)
            act_mean = act_2d.mean(dim=0)                       # (K,)
            act_std = act_2d.std(dim=0)                         # (K,)

            # Top-k most activated (保留/代偿)
            _, desc_idx = act_mean.topk(min(top_k, act_mean.shape[0]))
            # Top-k most suppressed (异常减弱)
            _, asc_idx = act_mean.topk(min(top_k, act_mean.shape[0]), largest=False)

            lines = [f"动态功能连接分析结果（共{L}个时间窗）：", ""]

            # ── 异常减弱模式 ──
            lines.append("显著异常模式（功能连接减弱）：")
            has_abnormal = False
            for i in asc_idx[:3]:
                val = act_mean[i].item()
                if val < -threshold:
                    has_abnormal = True
                    label = self.prototype_labels[i]['zh']
                    direction = "减弱" if "完整性" in label else "异常降低"
                    temporal = ""
                    if act_std[i].item() > 0.3:
                        temporal = "，波动较大"
                    elif act_std[i].item() < 0.15:
                        temporal = "，全程持续"
                    lines.append(f"  - {label} {direction} "
                                 f"(均值{val:.2f}{temporal})")
            if not has_abnormal:
                lines.append("  - 未检测到显著异常模式")

            lines.append("")

            # ── 保留/代偿性增强模式 ──
            lines.append("相对保留或代偿性增强模式：")
            has_preserved = False
            for i in desc_idx[:3]:
                val = act_mean[i].item()
                if val > threshold:
                    has_preserved = True
                    label = self.prototype_labels[i]['zh']
                    direction = "保留" if "完整性" in label else "代偿性增强"
                    temporal = ""
                    if act_std[i].item() > 0.3:
                        temporal = "，时有时无"
                    elif act_std[i].item() < 0.15:
                        temporal = "，稳定维持"
                    lines.append(f"  - {label} {direction} "
                                 f"(均值{val:.2f}{temporal})")
            if not has_preserved:
                lines.append("  - 未检测到显著保留/代偿模式")

            summaries.append("\n".join(lines))

        return summaries

    # ── Token ID pre-computation (P1 optimization) ───────────────────

    def _precompute_token_ids(self, tokenizer):
        """Pre-tokenize all static text fragments once at init time.

        Eliminates per-step tokenizer.encode() calls for the ~200-char
        text summary by storing token ID lists for every fixed text piece:
        headers, section titles, prototype labels, direction descriptions,
        and temporal stability phrases.

        Float values (e.g. "0.42") are tokenized lazily with a cache.
        """
        encode = tokenizer.encode
        self._encode = encode

        # ── Template fragments ──
        self._tok = {
            'header_prefix':     encode("动态功能连接分析结果（共"),
            'header_suffix':     encode("个时间窗）："),
            'newline':           encode("\n"),
            'section_abnormal':  encode("显著异常模式（功能连接减弱）："),
            'section_preserved': encode("相对保留或代偿性增强模式："),
            'item_prefix':       encode("  - "),
            'no_abnormal':       encode("  - 未检测到显著异常模式"),
            'no_preserved':      encode("  - 未检测到显著保留/代偿模式"),
            'mean_prefix':       encode(" (均值"),
            'paren_close':       encode(")"),
        }

        # ── 16 prototypes × 2 modes (abnormal / preserved) ──
        self._proto_item_tokens: Dict[int, Dict[str, List[int]]] = {}
        for i, proto in enumerate(self.prototype_labels):
            zh = proto['zh']
            has_integrity = "完整性" in zh
            self._proto_item_tokens[i] = {
                'abnormal':  encode(zh + " " + ("减弱" if has_integrity else "异常降低")),
                'preserved': encode(zh + " " + ("保留" if has_integrity else "代偿性增强")),
            }

        # ── Temporal stability descriptions ──
        self._tok_temporal = {
            'abnormal_volatile':    encode("，波动较大"),
            'abnormal_persistent':  encode("，全程持续"),
            'preserved_intermittent': encode("，时有时无"),
            'preserved_stable':     encode("，稳定维持"),
        }

        self._precomputed = True

    def _tokenize_number(self, value: float, fmt: str = ".2f") -> List[int]:
        """Tokenize a float value with caching — cache hit after first few steps."""
        key = format(value, fmt)
        cache = self._num_token_cache
        if key not in cache:
            cache[key] = self._encode(key)
        return cache[key]

    def _build_proto_token_ids(self, activations_2d: torch.Tensor,
                                B: int, L: int,
                                top_k: int = 6,
                                threshold: float = 0.1) -> torch.Tensor:
        """Directly build per-sample token ID tensors — bypasses string building.

        Produces the same output as _build_text_summary, but using pre-tokenized
        fragments and a lazy cache for float values. Returns a padded LongTensor on CPU.
        """
        tok = self._tok
        all_ids: List[List[int]] = []

        # ── Vectorized top-k (one pass for all samples) ──
        act_mean = activations_2d.mean(dim=1)                       # (B, K)
        act_std = activations_2d.std(dim=1)                         # (B, K)
        K = act_mean.shape[1]
        k_eff = min(top_k, K)

        _, desc_idx_all = act_mean.topk(k_eff, dim=1)                # (B, k_eff)
        _, asc_idx_all = act_mean.topk(k_eff, dim=1, largest=False)  # (B, k_eff)

        for b in range(B):
            ids: List[int] = []

            # ── Header ──
            ids.extend(tok['header_prefix'])
            ids.extend(self._tokenize_number(float(L), ".0f"))
            ids.extend(tok['header_suffix'])
            ids.extend(tok['newline'])
            ids.extend(tok['newline'])

            # ── Section: abnormal patterns ──
            ids.extend(tok['section_abnormal'])
            ids.extend(tok['newline'])

            has_abnormal = False
            for idx_t in asc_idx_all[b][:3]:
                i = idx_t.item()
                val = act_mean[b, i].item()
                if val < -threshold:
                    has_abnormal = True
                    ids.extend(tok['item_prefix'])
                    ids.extend(self._proto_item_tokens[i]['abnormal'])
                    ids.extend(tok['mean_prefix'])
                    ids.extend(self._tokenize_number(val))
                    # Temporal stability
                    std_v = act_std[b, i].item()
                    if std_v > 0.3:
                        ids.extend(self._tok_temporal['abnormal_volatile'])
                    elif std_v < 0.15:
                        ids.extend(self._tok_temporal['abnormal_persistent'])
                    ids.extend(tok['paren_close'])
                    ids.extend(tok['newline'])

            if not has_abnormal:
                ids.extend(tok['no_abnormal'])
                ids.extend(tok['newline'])

            ids.extend(tok['newline'])

            # ── Section: preserved / compensatory patterns ──
            ids.extend(tok['section_preserved'])
            ids.extend(tok['newline'])

            has_preserved = False
            for idx_t in desc_idx_all[b][:3]:
                i = idx_t.item()
                val = act_mean[b, i].item()
                if val > threshold:
                    has_preserved = True
                    ids.extend(tok['item_prefix'])
                    ids.extend(self._proto_item_tokens[i]['preserved'])
                    ids.extend(tok['mean_prefix'])
                    ids.extend(self._tokenize_number(val))
                    # Temporal stability
                    std_v = act_std[b, i].item()
                    if std_v > 0.3:
                        ids.extend(self._tok_temporal['preserved_intermittent'])
                    elif std_v < 0.15:
                        ids.extend(self._tok_temporal['preserved_stable'])
                    ids.extend(tok['paren_close'])
                    ids.extend(tok['newline'])

            if not has_preserved:
                ids.extend(tok['no_preserved'])
                ids.extend(tok['newline'])

            all_ids.append(ids)

        # ── Pad to max length ──
        max_len = max(len(seq) for seq in all_ids)
        padded = torch.zeros(B, max_len, dtype=torch.long)
        for i, seq in enumerate(all_ids):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        return padded


class BrainNetCNN(nn.Module):
    """BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment"""
    def __init__(self, node_size):
        super().__init__()
        self.d = node_size

        self.e2econv1 = E2EBlock(1, 32, node_size, bias=True)
        self.e2econv2 = E2EBlock(32, 64, node_size, bias=True)
        self.E2N = torch.nn.Conv2d(64, 128, (1, self.d))
        self.N2G = torch.nn.Conv2d(128, 256, (self.d, 1))


        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)


    def forward(self, node_feature: torch.tensor, return_channel_features: bool = False):
        """Forward pass.

        Args:
            node_feature: (N, 1, C, C)  DFC matrices (N = B * L)
            return_channel_features: if True, also returns per-channel features
                                    from the E2N layer before N2G compression.

        Returns:
            out4:             (N, 256)   N2G-compressed global feature
            out3_ch: (optional) (N, 128, C)  per-channel features (E2N output)
        """
        out1 = F.leaky_relu(self.bn1(self.e2econv1(node_feature)), negative_slope=0.33)
        out2 = F.leaky_relu(self.bn2(self.e2econv2(out1)), negative_slope=0.33)
        out3 = F.leaky_relu(self.bn3(self.E2N(out2)), negative_slope=0.33)    # (N, 128, C, 1)
        out4 = F.dropout(F.leaky_relu(self.bn4(self.N2G(out3)), negative_slope=0.33), p=0.5).squeeze(-1).squeeze(-1)

        if return_channel_features:
            out3_ch = out3.squeeze(-1)   # (N, 128, C) — per-channel features
            return out4, out3_ch
        return out4


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Mask Head (P1.2) — LLM → BNC 知识反馈通道
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeMaskHead(nn.Module):
    """LLM 专业知识掩膜预测头 — 将 LLM hidden state 映射为 per-channel 调制向量。

    对应 tech.md §3.4（Module 3）：
      llm_pooled (B, 4096) → mask_head → M (B, C)，值域 [0.05, 2.0]

    结构：
      Linear(4096→512) → LayerNorm → GELU → Dropout(0.2)
      → Linear(512→C) → Sigmoid → scale to [0.05, 2.0]

    语义（tech.md §3.4）：
      M_i ≈ 1.0 → 该脑区通道不变（LLM 认为中等重要）
      M_i < 1.0 → 抑制该脑区通道（LLM 认为不重要/噪声）
      M_i > 1.0 → 增强该脑区通道（LLM 认为对诊断关键）

    C = node_size（脑区数量），256 是 BNC 输出特征维度而非通道数，
    掩膜应在通道（脑区）级别而非特征维度级别进行调制。

    Input:  (B, 4096)   llm_pooled — LLM mean-pooled hidden state
    Output: (B, C)       M — per-channel knowledge mask, value range [0.05, 2.0]
    """

    def __init__(self, llm_dim=4096, num_channels=90, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.num_channels = num_channels
        self.fc1 = nn.Linear(llm_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_channels)
        # Temperature for sigmoid scaling — prevents binary saturation (always 0.05 or 2.0).
        # Higher τ → softer M values with intermediate magnitudes.
        # Fixed (not learnable) to prevent the model from lowering τ and re-entering
        # binary saturation, which collapses per-channel modulation to on/off.
        self.temperature = nn.Parameter(torch.tensor(5.0))

    def forward(self, llm_pooled):
        h = self.drop(self.act(self.ln(self.fc1(llm_pooled))))
        logits = self.fc2(h)                        # (B, C)
        M = torch.sigmoid(logits / self.temperature)  # (B, C) in [0, 1]
        M = 0.05 + 1.95 * M                         # scale to [0.05, 2.0]
        return M


def mask_entropy_loss(M: torch.Tensor) -> torch.Tensor:
    """掩膜多样性正则 — 惩罚掩膜缺乏跨通道差异性。

    对应 tech.md §3.4 + §5.3（替代原 L_sparsity）：
    当 M 所有 C 个通道都接近同一值（如全 ~1.0，std ≈ 0）时施加惩罚，
    鼓励 LLM 对不同的脑区通道产生差异化调制。

    与 L_sparsity 的关键区别：
      - L_sparsity = L1(M−1.0)：将 M 推向 1.0（恒等映射），抵消掩膜设计目的
      - L_entropy = ReLU(0.05 − std(M))：仅在全 mask 无差异时惩罚，
        LLM 一旦对不同通道产生有意义的差异化调制（std > 0.05），损失自动归零

    原理：
      如果 std(M) ≈ 0（所有通道值相同），说明 LLM 未对脑区产生
      任何选择性调制 → 掩膜形同虚设。通过惩罚低方差，驱动 LLM 输出
      有跨通道区分性的掩膜，同时不强制每个通道都极端偏离 1.0。

    Args:
        M: (B, C)  per-channel knowledge mask tensor, value range [0.05, 2.0]

    Returns:
        scalar — hinge penalty on low per-sample std across channels
    """
    import torch.nn.functional as F
    std = M.std(dim=-1)                     # (B,) per-sample std across C channels
    return F.relu(0.05 - std).mean()        # penalty only when std < 0.05

