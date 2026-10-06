"""Knowledge compiler: 32 个 CAER-v3 display stats → LLM query 向量。

每个统计量完全独立编译（无任何 token 间混合）：
  q_i = P(phrase_i) + W_i * s_i          (d_model 维)
phrase_i 由冻结词表查表求平均 + RMS 对齐得到（buffer，no_grad）；
W_i 为每统计量独立的数值轴（等价 32 个 Linear(1, d_model, bias=False)）。
"""

import torch
import torch.nn as nn

# 32 个短语，顺序与 compute_graph_evidence 输出的 stats 顺序一一对应：
# FC 9 + 频谱 9 + 形态 14
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
    # 频谱 9
    "delta relative power",
    "theta relative power",
    "alpha relative power",
    "beta relative power",
    "theta beta power ratio",
    "alpha delta theta power ratio",
    "diffuse slowing index",
    "frontal posterior alpha power ratio",
    "temporal theta power fraction",
    # 形态 14
    "line length mean",
    "line length maximum",
    "sharpness mean",
    "sharpness maximum",
    "sharpness focality",
    "amplitude kurtosis mean",
    "amplitude kurtosis maximum",
    "zero crossing rate",
    "gamma power fraction",
    "spike rate mean",
    "spike rate maximum",
    "alpha peak frequency",
    "alpha peak frequency variability",
    "aperiodic spectral slope",
]

_EPS = 1e-8


class KnowledgeCompiler(nn.Module):
    """Phrase bank + 数值 encoder + 共享短语投影。

    forward(s): (B, num_stats) display 值 → q: (B, num_stats, d_model)
    """

    def __init__(self, d_llm, d_model, num_stats, tokenizer,
                 word_embeddings, llm_type='llama', numeric_init_std=0.02):
        super().__init__()
        self.d_llm = d_llm
        self.d_model = d_model
        self.num_stats = num_stats
        assert num_stats == len(_PHRASES), \
            f"num_stats={num_stats} 与短语表 {len(_PHRASES)} 不一致"

        # ── phrase bank：tokenize → 冻结词表查表 → mean → RMS 对齐 ──
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

        # 共享短语投影：d_llm → d_model（bias=False，保证线性分解成立）
        self.phrase_projection = nn.Linear(d_llm, d_model, bias=False)
        nn.init.xavier_uniform_(self.phrase_projection.weight)

        # 每统计量独立数值轴（bias=False：s_i=0 时数值分支严格为零）
        self.value_weights = nn.Parameter(
            torch.empty(num_stats, d_model))
        nn.init.normal_(self.value_weights, std=numeric_init_std)
        # 注：数值分支初始 RMS ≈ std × RMS(s_i) ≈ 0.02 ≪ 短语分支，
        # 避免数值一开始淹没短语身份。若需按训练集逐统计量校准，
        # 可用 σ_i = α·ρ_p/ρ_s,i（α≈0.5~1.0，源域校准，勿用目标域）。

    def forward(self, stats):
        """stats: (B, num_stats) display 值（clip ±5）→ (B, num_stats, d_model)"""
        p = self.phrase_projection(self.phrase_bank)         # (32, d_model)
        p = p.unsqueeze(0).expand(stats.shape[0], -1, -1)
        v = stats.unsqueeze(-1) * self.value_weights         # (B, 32, d_model)
        return p + v
