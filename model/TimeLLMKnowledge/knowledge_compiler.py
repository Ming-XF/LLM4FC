"""Knowledge compiler v2: 统计数值分支 + raw EEG patch 分支 → 知识 query。

每统计量独立编译（无 token 间混合）：
  q_i = g_num·RMSNorm(W_i·s_i) + g_raw·RMSNorm(R_i·Stem(EEG块))   (d_model 维)
g_num/g_raw 为 softplus 非负 gate（初始均 1.0，两分支约 50/50）；
phrase 不再进入 q——phrase 在 reprogram 之后于 4096 维与 soft prompt
gated 求和（见 TimeLLMKnowledge.py），只作统计量身份/语言锚点（冻结）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 32 个短语，顺序与 compute_graph_evidence 输出的 stats 顺序一一对应：
# FC 9 + 频谱 13（9 带功率 + PSD 谱形 4）+ 形态时域 10
_PHRASES = [
    # FC 9
    "overall connectivity strength",
    "interhemispheric coupling ratio",
    "frontal posterior connectivity ratio",
    "hemispheric asymmetry",
    "central connectivity",
    "temporal connectivity",
    "parietal connectivity",
    "occipital connectivity",
    "midline connectivity",
    # 频谱 13（带功率 9 + PSD 谱形 4）
    "delta relative power",
    "theta relative power",
    "alpha relative power",
    "beta relative power",
    "theta beta power ratio",
    "alpha delta theta power ratio",
    "diffuse slowing index",
    "frontal posterior alpha power ratio",
    "temporal theta power fraction",
    "gamma power fraction",
    "alpha peak frequency",
    "alpha peak frequency variability",
    "aperiodic spectral slope",
    # 形态时域 10
    "line length mean",
    "line length maximum",
    "sharpness mean",
    "sharpness maximum",
    "sharpness focality",
    "amplitude kurtosis mean",
    "amplitude kurtosis maximum",
    "zero crossing rate",
    "spike rate mean",
    "spike rate maximum",
]

_EPS = 1e-8


def rms_norm(x, eps=1e-6):
    """固定 RMS 归一化（无参数，沿最后一维）。"""
    return x / (x.pow(2).mean(-1, keepdim=True).add(eps).sqrt())


def _softplus_inv(v):
    return math.log(math.exp(v) - 1.0)


class KnowledgeCompiler(nn.Module):
    """数值轴 + raw 分支 + 分支 gate。

    forward(stats, raw_feat):
        stats:    (B, num_stats) display 值（clip ±5，经 nan_to_num）
        raw_feat: (B, num_stats, d_model) 组 raw token（外部组装：
                  FC 图/PSD patch 嵌入 + 形态 PatchStem 投影）
        → q: (B, num_stats, d_model)
    """

    def __init__(self, d_llm, d_model, num_stats, tokenizer,
                 word_embeddings, llm_type='llama',
                 numeric_init_std=0.02):
        super().__init__()
        self.d_llm = d_llm
        self.d_model = d_model
        self.num_stats = num_stats
        assert num_stats == len(_PHRASES), \
            f"num_stats={num_stats} 与短语表 {len(_PHRASES)} 不一致"

        # ── phrase bank：tokenize → 冻结词表查表 → mean → RMS 对齐（冻结）──
        ids_list = []
        for phrase in _PHRASES:
            text = " " + phrase if llm_type == 'llama' else phrase
            ids_list.append(tokenizer.encode(text, add_special_tokens=False))
        emb = word_embeddings.weight.detach().float()        # (V, d_llm)
        # 对齐目标 = 词表 embedding 行的 RMS 中位数（LLM 原生 token 尺度）
        ref_rms = emb.pow(2).mean(-1).sqrt().median().clamp_min(_EPS)
        phrase_bank = []
        for ids in ids_list:
            v = emb[ids].mean(0)                             # (d_llm,)
            rms = v.pow(2).mean().sqrt().clamp_min(_EPS)
            phrase_bank.append(v / rms * ref_rms)
        self.register_buffer("phrase_bank",
                             torch.stack(phrase_bank))       # (32, d_llm)

        # ── 数值分支：每统计量独立数值轴（bias=False：s_i=0 时严格为零）──
        self.value_weights = nn.Parameter(
            torch.empty(num_stats, d_model))
        nn.init.normal_(self.value_weights, std=numeric_init_std)

        # ── 分支 gate：softplus 非负，初始 1.0（约 50/50）──
        self.num_gate = nn.Parameter(torch.tensor(_softplus_inv(1.0)))
        self.raw_gate = nn.Parameter(torch.tensor(_softplus_inv(1.0)))

        # 分支开关（由模型 config 注入）
        self.use_numeric_branch = True
        self.use_raw_branch = True

    def forward(self, stats, raw_feat=None):
        """stats: (B, num_stats)；raw_feat: (B, num_stats, d_model) 组 raw token
        （外部组装：FC 图/PSD patch 嵌入 + 形态 PatchStem 投影）→ (B, S, d)。"""
        B = stats.shape[0]
        q = None
        if self.use_numeric_branch:
            v = stats.unsqueeze(-1) * self.value_weights    # (B, S, d_model)
            q = F.softplus(self.num_gate) * rms_norm(v)
        if self.use_raw_branch:
            q_raw = F.softplus(self.raw_gate) * rms_norm(raw_feat)
            q = q_raw if q is None else q + q_raw
        assert q is not None, "数值分支与 raw 分支至少开启一个"
        return q
