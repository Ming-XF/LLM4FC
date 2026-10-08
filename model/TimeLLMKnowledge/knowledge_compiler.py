"""Raw EEG feature compiler for TimeLLMKnowledge.

Each knowledge slot is compiled independently from a raw feature token. The
phrase bank is kept as a frozen language identity anchor and is fused after
reprogramming; no hand-crafted numeric statistics are used here.
"""

import torch
import torch.nn as nn

# 32 个知识槽位的短语锚点：FC 9 + 频谱/PSD 13 + 波形形态 10。
# 这些短语定义 raw token 的固定语义位置，不对应显式计算的统计值。
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


class KnowledgeCompiler(nn.Module):
    """Compile raw feature tokens into independent knowledge queries.

    ``raw_feat`` has shape ``(B, num_stats, d_model)`` and is assembled by
    the model from FC, relative-PSD, and waveform encoders. There is no
    token-to-token mixing at this stage.
    """

    def __init__(self, d_llm, d_model, num_stats, tokenizer,
                 word_embeddings, llm_type='llama'):
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

        # Raw branch scale: kept learnable so the compiler can adapt the
        # magnitude of its queries without reintroducing a numeric shortcut.
        self.raw_gate = nn.Parameter(torch.tensor(0.541324854612918))

    def forward(self, raw_feat):
        """Return normalized raw knowledge queries: ``(B, S, d_model)``."""
        if raw_feat is None:
            raise ValueError("KnowledgeCompiler requires raw feature tokens")
        expected = (self.num_stats, self.d_model)
        if raw_feat.dim() != 3 or tuple(raw_feat.shape[1:]) != expected:
            raise ValueError(
                "raw_feat must have shape "
                f"(B, {self.num_stats}, {self.d_model}), got "
                f"{tuple(raw_feat.shape)}")
        return torch.nn.functional.softplus(self.raw_gate) * rms_norm(raw_feat)
