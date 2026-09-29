"""E5: Graph-to-Text —— 把每个样本的图统计量用确定性规则转成自然语言证据。

证据完全由无监督规则生成（不含任何源域学习参数），语言锚定使其
跨数据集稳定；LLM 以文本 token 形式读取证据，真正利用其预训练
语言理解能力。
"""

import torch

# ── 19 通道 10-20 分区（与各 age 数据集通道顺序一致）──
_CH_LEFT = [0, 1, 2, 3, 4, 10, 11, 12]      # Fp1 F3 C3 P3 O1 F7 T3 T5
_CH_RIGHT = [5, 6, 7, 8, 9, 13, 14, 15]     # Fp2 F4 C4 P4 O2 F8 T4 T6
_CH_MID = [16, 17, 18]                       # Fz Cz Pz
_CH_FRONTAL = [0, 1, 5, 6, 10, 13]           # Fp1 F3 Fp2 F4 F7 F8
_CH_POSTERIOR = [3, 4, 8, 9, 12, 15]         # P3 O1 P4 O2 T5 T6


def _build_edge_masks(C, device):
    """构建 (C, C) 布尔上三角边类型掩码集合。"""
    triu = torch.triu(torch.ones(C, C, device=device, dtype=torch.bool),
                      diagonal=1)
    L = torch.zeros(C, device=device, dtype=torch.bool)
    R = torch.zeros(C, device=device, dtype=torch.bool)
    Fm = torch.zeros(C, device=device, dtype=torch.bool)
    P = torch.zeros(C, device=device, dtype=torch.bool)
    L[_CH_LEFT] = True
    R[_CH_RIGHT] = True
    Fm[_CH_FRONTAL] = True
    P[_CH_POSTERIOR] = True
    sides = L | R                                  # 排除中线
    inter = triu & ((L[:, None] & R[None, :]) | (R[:, None] & L[None, :]))
    intra = triu & sides[:, None] & sides[None, :] & ~inter
    frontal = triu & Fm[:, None] & Fm[None, :]
    posterior = triu & P[:, None] & P[None, :]
    return inter, intra, frontal, posterior


def _quantize(x, thresholds, label_list):
    """按阈值把 (B,) 张量离散为标签词列表。thresholds: [t1, t2]（升序）。"""
    idx = torch.zeros_like(x, dtype=torch.long)
    for t in thresholds:
        idx += (x >= t).long()
    return [label_list[i] for i in idx.tolist()]


def compute_graph_evidence(dfc):
    """对 (B, W, C, C) DFC 计算确定性图统计量并量化为证据描述。

    Returns:
        levels: (B,) list，每个元素为 6 个证据词（strength, inter,
            frontal, variability, trend, asymmetry）。
        stats:  (B, 6) 连续统计量（按名义尺度重缩放，供 --stat_vector
            直接拼入读出层输入，保留可迁移的连续信号）。
    """
    B, W, C, _ = dfc.shape
    device = dfc.device

    with torch.no_grad():
        abs_dfc = dfc.abs()
        # 全矩阵强度（含对角：对角=1 作为每窗口的锚点参照，
        # 使统计量跨站点更可比——见 WORK.md 诊断）
        win_strength = abs_dfc.mean(dim=(-1, -2))                       # (B, W)
        strength = win_strength.mean(-1)                                # (B,)
        variability = win_strength.std(-1)                              # (B,)
        # 时间趋势：每窗口强度对窗口序号的归一化斜率
        t = torch.arange(W, device=device).float()
        t = (t - t.mean())
        slope = (t * (win_strength - win_strength.mean(-1, keepdim=True))
                 ).sum(-1) / (t ** 2).sum()
        trend_norm = slope / (win_strength.mean(-1) + 1e-6)

        # 子矩阵均值（含对角，与跨域诊断的 old-def 一致）
        abs_all = abs_dfc.sum(dim=(-1, -2))                             # (B, W)

        def sub_sum(idx):
            return abs_dfc[:, :, idx][:, :, :, idx].sum(dim=(-1, -2))

        left_s = sub_sum(_CH_LEFT)
        right_s = sub_sum(_CH_RIGHT)
        front_s = sub_sum(_CH_FRONTAL)
        post_s = sub_sum(_CH_POSTERIOR)
        inter_w = (abs_all - left_s - right_s) / (
            2 * len(_CH_LEFT) * len(_CH_RIGHT))
        intra_w = (left_s + right_s) / (
            len(_CH_LEFT) ** 2 + len(_CH_RIGHT) ** 2)
        front_w = front_s / len(_CH_FRONTAL) ** 2
        post_w = post_s / len(_CH_POSTERIOR) ** 2
        asym_w = (left_s / len(_CH_LEFT) ** 2
                  - right_s / len(_CH_RIGHT) ** 2).abs()

        inter_ratio = inter_w.mean(-1) / (intra_w.mean(-1) + 1e-6)
        front_ratio = front_w.mean(-1) / (post_w.mean(-1) + 1e-6)
        asym = asym_w.mean(-1)

        # ── 量化（相关值是尺度无关的，固定阈值跨站点可解释）──
        strength_labels = _quantize(
            strength, [0.15, 0.30], ['low', 'mid', 'high'])
        inter_labels = _quantize(
            inter_ratio, [0.7, 1.0], ['weak', 'mid', 'strong'])
        front_labels = _quantize(
            front_ratio, [0.7, 1.0], ['low', 'mid', 'high'])
        var_labels = _quantize(
            variability, [0.02, 0.05], ['low', 'mid', 'high'])
        asym_labels = _quantize(
            asym, [0.02, 0.05], ['small', 'mid', 'large'])
        trend_idx = torch.where(
            trend_norm > 0.02, torch.full_like(trend_norm, 2, dtype=torch.long),
            torch.where(trend_norm < -0.02,
                        torch.full_like(trend_norm, 0, dtype=torch.long),
                        torch.ones_like(trend_norm, dtype=torch.long)))
        trend_words = ['down', 'flat', 'up']

    levels = []
    for b in range(B):
        levels.append([
            strength_labels[b], inter_labels[b], front_labels[b],
            var_labels[b], trend_words[int(trend_idx[b])], asym_labels[b],
        ])

    # 连续统计量按名义尺度重缩放（~单位量级，供线性头学习）
    stats = torch.stack([
        strength / 0.3,
        inter_ratio,
        front_ratio,
        variability / 0.05,
        trend_norm / 0.05,
        asym / 0.05,
    ], dim=-1)                                                 # (B, 6)
    return levels, stats


# ── 证据句模板（槽位词均为 llama 词表的单 token，保证定长；
#    固定前缀/后缀可多 token，不影响定长）──
_EVIDENCE_TEMPLATES = [
    ("Overall connectivity strength is ", ["low", "mid", "high"], "."),
    (" Inter-hemispheric coupling is ", ["weak", "mid", "strong"], "."),
    (" Frontal connectivity is ", ["low", "mid", "high"], "."),
    (" Temporal variability is ", ["low", "mid", "high"], "."),
    (" Connectivity ", ["down", "flat", "up"], " across windows."),
    (" Hemispheric asymmetry is ", ["small", "mid", "large"], "."),
]
