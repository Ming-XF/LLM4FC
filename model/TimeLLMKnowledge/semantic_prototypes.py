"""Semantic prototype bank: 每短语词表语义邻域预选 + 共享 Linear 压缩。

替代 mapping_layer（Linear(V→1000) 全词表学习混合，1.28 亿参数）：
  1. 对 32 个知识短语各自的聚合向量（phrase_bank 行），在冻结词表 embedding
     上算余弦相似度（fp32、归一化），每短语取 top-K 最近 token 作候选
     （确定性、域无关，init 一次性计算）；
  2. 共享 Linear(K → P) 压缩器作用于候选维，得每短语 P 个原型
     (32, P, 4096)，初始化为"均匀下采样近单位映射"——初始原型就是
     真实词向量，从语义邻域出发再学习；
  3. reprogram 时第 i 个知识 token 只在其对应短语的 P 个原型上做注意力。

设计目标（codex 评审 Q1/Q3/Q5 意见）：原型选择变为确定性的语义预选，
学习参数降到 ~10 万级，避免源域统计量分布上学习的原型在目标域 OOD。
"""

import torch
import torch.nn as nn

_EPS = 1e-8


def _uniform_subsample_init(weight):
    """把 (K, P) 压缩权重初始化为均匀下采样近单位映射。

    P 个锚点均匀分布在 top-K 相似度排序上（每 K//P 个候选取 1 个），
    初始原型 = 真实词向量，避免一上来被稠密随机混合推出语义邻域。
    """
    K, P = weight.shape
    with torch.no_grad():
        weight.zero_()
        anchors = torch.linspace(0, K - 1, P).round().long()
        weight[anchors, torch.arange(P)] = 1.0


class SemanticPrototypeBank(nn.Module):
    """32 短语 → 各自 top-K 词表近邻候选 → 共享 Linear → P 原型。

    forward(phrase_ids=None):
        None          → (num_phrases, P, d_llm) 全部原型
        phrase_ids    → (L, P, d_llm)，按 token 的短语归属 gather
    """

    def __init__(self, phrase_bank, word_embeddings, tokenizer,
                 preselect=1000, per_phrase=100):
        super().__init__()
        num_phrases, d_llm = phrase_bank.shape
        V, d_emb = word_embeddings.shape
        assert d_emb == d_llm, "词表 embedding 维度与短语锚不一致"
        assert per_phrase <= preselect, "per_phrase 不能超过 preselect"
        self.preselect = preselect
        self.per_phrase = per_phrase

        # ── 候选预选：fp32 归一化余弦 → 每短语 top-K 索引（冻结）──
        emb = word_embeddings.detach().float()
        emb_n = emb / emb.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(_EPS)
        anchor = phrase_bank.detach().float()
        anchor_n = anchor / anchor.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(_EPS)

        # 排除特殊 token（pad/eos/bos/unk/附加 special）
        special = torch.tensor(
            sorted(set(tokenizer.all_special_ids)), dtype=torch.long)
        ids_list = []
        for p in range(num_phrases):
            s = emb_n @ anchor_n[p]                              # (V,)
            s[special] = -float('inf')
            ids_list.append(s.topk(preselect).indices)
        self.register_buffer('candidate_ids',
                             torch.stack(ids_list, dim=0).long())  # (P_num, K)

        # 候选词向量缓存（冻结、确定性；非持久 buffer，checkpoint 不保存，
        # 加载时由相同 tokenizer/词表重算，保证一致）
        sel = emb[self.candidate_ids]                            # (P_num, K, d)
        self.register_buffer('candidate_embeddings', sel,
                             persistent=False)

        # ── 共享压缩器（bias=False）：均匀下采样近单位初始化 ──
        self.compressor = nn.Linear(preselect, per_phrase, bias=False)
        _uniform_subsample_init(self.compressor.weight)

    def forward(self, phrase_ids=None):
        # (P_num, K, d) @ (P, K) → (P_num, P, d)
        proto = torch.einsum('nkd,pk->npd',
                             self.candidate_embeddings,
                             self.compressor.weight)
        if phrase_ids is not None:
            proto = proto[phrase_ids]
        return proto
