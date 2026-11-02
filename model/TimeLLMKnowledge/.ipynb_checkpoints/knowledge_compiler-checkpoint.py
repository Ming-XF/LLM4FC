"""Raw EEG feature compiler for the three broad knowledge families."""

import torch
import torch.nn as nn

# These phrases describe the information families actually provided to the
# encoders. They are semantic anchors, not explicit hand-crafted statistics.
_PHRASES = [
    "EEG functional connectivity patterns across channels",
    "EEG relative spectral distribution across channels and frequencies",
    "EEG time-domain waveform morphology patterns",
]

_EPS = 1e-8


def rms_norm(x, eps=1e-6):
    """固定 RMS 归一化（无参数，沿最后一维）。"""
    return x / (x.pow(2).mean(-1, keepdim=True).add(eps).sqrt())


class KnowledgeCompiler(nn.Module):
    """Compile one learned query for each broad EEG knowledge family."""

    def __init__(self, d_llm, d_model, num_stats, tokenizer,
                 word_embeddings, llm_type='llama'):
        super().__init__()
        self.d_llm = d_llm
        self.d_model = d_model
        self.num_aspects = num_stats
        assert num_stats == len(_PHRASES), \
            f"num_aspects={num_stats} 与短语表 {len(_PHRASES)} 不一致"

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
                             torch.stack(phrase_bank))       # (3, d_llm)

        # Raw branch scale: kept learnable so the compiler can adapt the
        # magnitude of its queries without reintroducing a numeric shortcut.
        self.raw_gate = nn.Parameter(torch.tensor(0.541324854612918))

    def forward(self, raw_feat):
        """Return normalized raw knowledge queries: ``(B, 3, d_model)``."""
        if raw_feat is None:
            raise ValueError("KnowledgeCompiler requires raw feature tokens")
        expected = (self.num_aspects, self.d_model)
        if raw_feat.dim() != 3 or tuple(raw_feat.shape[1:]) != expected:
            raise ValueError(
                "raw_feat must have shape "
                f"(B, {self.num_aspects}, {self.d_model}), got "
                f"{tuple(raw_feat.shape)}")
        return torch.nn.functional.softplus(self.raw_gate) * rms_norm(raw_feat)
