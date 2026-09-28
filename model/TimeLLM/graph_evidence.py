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
        (B,) list，每个元素为 5 个证据词（strength, inter, frontal,
        variability, trend），对应 _EVIDENCE_TEMPLATES 的槽位。
    """
    B, W, C, _ = dfc.shape
    device = dfc.device
    inter_m, intra_m, front_m, post_m = _build_edge_masks(C, device)
    inter_m = inter_m.unsqueeze(0)                 # (1, C, C)
    intra_m = intra_m.unsqueeze(0)
    front_m = front_m.unsqueeze(0)
    post_m = post_m.unsqueeze(0)

    with torch.no_grad():
        abs_dfc = dfc.abs()
        win_strength = abs_dfc.mean(dim=(-1, -2))                       # (B, W)
        strength = win_strength.mean(-1)                                # (B,)
        variability = win_strength.std(-1)                              # (B,)
        # 时间趋势：每窗口强度对窗口序号的归一化斜率
        t = torch.arange(W, device=device).float()
        t = (t - t.mean())
        slope = (t * (win_strength - win_strength.mean(-1, keepdim=True))
                 ).sum(-1) / (t ** 2).sum()
        trend_norm = slope / (win_strength.mean(-1) + 1e-6)

        def edge_mean(mask):
            n = mask.sum()
            return (abs_dfc * mask).sum(dim=(-1, -2)) / n               # (B, W)

        inter_mw = edge_mean(inter_m).mean(-1)                          # (B,)
        intra_mw = edge_mean(intra_m).mean(-1)
        front_mw = edge_mean(front_m).mean(-1)
        post_mw = edge_mean(post_m).mean(-1)
        inter_ratio = inter_mw / (intra_mw + 1e-6)
        front_ratio = front_mw / (post_mw + 1e-6)
        # 半球不对称：左右半球内部边强度的差异
        L_m = torch.zeros(C, device=device, dtype=torch.bool)
        R_m = torch.zeros(C, device=device, dtype=torch.bool)
        L_m[_CH_LEFT] = True
        R_m[_CH_RIGHT] = True
        left_mask = (torch.triu(torch.ones(C, C, device=device,
                                          dtype=torch.bool), diagonal=1)
                     & L_m[:, None] & L_m[None, :]).unsqueeze(0)
        right_mask = (torch.triu(torch.ones(C, C, device=device,
                                           dtype=torch.bool), diagonal=1)
                      & R_m[:, None] & R_m[None, :]).unsqueeze(0)
        left_mw = (abs_dfc * left_mask).sum((-1, -2)) / left_mask.sum()
        right_mw = (abs_dfc * right_mask).sum((-1, -2)) / right_mask.sum()
        asym = (left_mw - right_mw).abs().mean(-1)                      # (B,)

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
    return levels


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
