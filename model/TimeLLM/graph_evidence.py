"""E5: Graph-to-Text —— 把每个样本的图统计量用确定性规则转成自然语言证据。

证据完全由无监督规则生成（不含任何源域学习参数），语言锚定使其
跨数据集稳定；LLM 以文本 token 形式读取证据，真正利用其预训练
语言理解能力。
"""

import torch

# ── 19 通道 10-20 分区（与各 age 数据集通道顺序一致）──
_CH_LEFT = [0, 1, 2, 3, 4, 10, 11, 12]      # Fp1 F3 C3 P3 O1 F7 T3 T5
_CH_RIGHT = [5, 6, 7, 8, 9, 13, 14, 15]     # Fp2 F4 C4 P4 O2 F8 T4 T6
_CH_MIDLINE = [16, 17, 18]                   # Fz Cz Pz
_CH_FRONTAL = [0, 1, 5, 6, 10, 13]           # Fp1 F3 Fp2 F4 F7 F8
_CH_POSTERIOR = [3, 4, 8, 9, 12, 15]         # P3 O1 P4 O2 T5 T6
_CH_CENTRAL = [2, 7, 17]                     # C3 C4 Cz
_CH_TEMPORAL = [11, 12, 14, 15]              # T3 T5 T4 T6
_CH_PARIETAL = [3, 8, 18]                    # P3 P4 Pz
_CH_OCCIPITAL = [4, 9]                       # O1 O2


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


def compute_graph_evidence(dfc, spectral=None, morph=None):
    """对 (B, W, C, C) DFC 计算确定性图统计量并量化为证据描述。

    W=1（整分钟静态 FC，TimeLLMKnowledge）受支持：variability/trend
    置 0，其余统计量照常。

    Args:
        dfc: (B, W, C, C) 动态功能连接。
        spectral: (B, 9) 频谱统计量（数据集层确定性计算），None 时
            仅用 FC 证据（向后兼容）。
        morph: (B, 14) 形态学/周期-非周期统计量（CAER-v3），None 时
            输出保持 18 维（v2 兼容）。

    Returns:
        levels: (B,) list，每个元素为证据词列表
            （FC 6 句 + 频谱 4 句 [+ 形态 2 句]）。
        stats:  (B, 9) / (B, 18) / (B, 32) 连续统计量（裁剪 ±5）。
            32 维分组：FC 9 | 频谱 13（9 带功率 + PSD 谱形 4）| 形态时域 10。
    """
    B, W, C, _ = dfc.shape
    device = dfc.device

    with torch.no_grad():
        abs_dfc = dfc.abs()
        # 全矩阵强度（含对角：对角=1 作为每窗口的锚点参照，
        # 使统计量跨站点更可比——见 WORK.md 诊断）
        win_strength = abs_dfc.mean(dim=(-1, -2))                       # (B, W)
        strength = win_strength.mean(-1)                                # (B,)
        if W > 1:
            variability = win_strength.std(-1)                          # (B,)
            # 时间趋势：每窗口强度对窗口序号的归一化斜率
            t = torch.arange(W, device=device).float()
            t = (t - t.mean())
            slope = (t * (win_strength - win_strength.mean(-1, keepdim=True))
                     ).sum(-1) / (t ** 2).sum()
            trend_norm = slope / (win_strength.mean(-1) + 1e-6)
        else:
            # W=1（整分钟静态 FC）：单窗口 std 为 NaN、斜率分母为 0，
            # 二者均置 0（variability/trend 不进 fc_stats，只喂量化词）
            variability = torch.zeros_like(strength)
            trend_norm = torch.zeros_like(strength)

        # 子矩阵均值（含对角，与跨域诊断的 old-def 一致）
        abs_all = abs_dfc.sum(dim=(-1, -2))                             # (B, W)

        def sub_sum(idx):
            return abs_dfc[:, :, idx][:, :, :, idx].sum(dim=(-1, -2))

        left_s = sub_sum(_CH_LEFT)
        right_s = sub_sum(_CH_RIGHT)
        front_s = sub_sum(_CH_FRONTAL)
        post_s = sub_sum(_CH_POSTERIOR)
        cent_s = sub_sum(_CH_CENTRAL)
        temp_s = sub_sum(_CH_TEMPORAL)
        pari_s = sub_sum(_CH_PARIETAL)
        occ_s = sub_sum(_CH_OCCIPITAL)
        mid_s = sub_sum(_CH_MIDLINE)
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
        cent_mw = (cent_s / len(_CH_CENTRAL) ** 2).mean(-1)
        temp_mw = (temp_s / len(_CH_TEMPORAL) ** 2).mean(-1)
        pari_mw = (pari_s / len(_CH_PARIETAL) ** 2).mean(-1)
        occ_mw = (occ_s / len(_CH_OCCIPITAL) ** 2).mean(-1)
        mid_mw = (mid_s / len(_CH_MIDLINE) ** 2).mean(-1)

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

    # 连续统计量按名义尺度重缩放（~单位量级，供线性头学习）；
    # 裁剪重尾。FC 9 维：4 个跨域一致的核心特征 + 5 个脑区均值
    # （variability/trend 两特征在 CAUEEG 与 TUAB 的年龄相关性符号
    # 相反，已剔除——见 WORK.md 诊断；hub-ness 同样破坏迁移，已剔除）
    fc_stats = torch.stack([
        strength / 0.3,
        inter_ratio,
        front_ratio,
        asym / 0.05,
        cent_mw / 0.3,
        temp_mw / 0.3,
        pari_mw / 0.3,
        occ_mw / 0.3,
        mid_mw / 0.3,
    ], dim=-1)                                                 # (B, 9)
    fc_stats = fc_stats.clamp(-5.0, 5.0)

    if spectral is None:
        return levels, fc_stats

    # ── 频谱证据（CAER-v2）：量化 4 个频谱句 + 9 维连续频谱统计量 ──
    spec = spectral.float().to(device)
    rel_d, rel_t, rel_a, rel_b = spec[:, 0], spec[:, 1], spec[:, 2], spec[:, 3]
    tb_ratio, a_dt, slowing, fa_pa, tt_total = (spec[:, 4], spec[:, 5],
                                                spec[:, 6], spec[:, 7],
                                                spec[:, 8])
    with torch.no_grad():
        slowing_labels = _quantize(
            slowing, [0.6, 1.2], ['low', 'mid', 'high'])
        alpha_labels = _quantize(
            rel_a, [0.25, 0.5], ['low', 'mid', 'high'])
        tb_labels = _quantize(
            tb_ratio, [0.8, 1.6], ['low', 'mid', 'high'])
        fb_idx = torch.where(
            fa_pa < 0.8, torch.zeros_like(fa_pa, dtype=torch.long),
            torch.where(fa_pa > 1.3, torch.full_like(fa_pa, 2, dtype=torch.long),
                        torch.ones_like(fa_pa, dtype=torch.long)))
        fb_words = ['front', 'even', 'back']
    for b in range(B):
        levels[b].extend([
            slowing_labels[b], alpha_labels[b], tb_labels[b],
            fb_words[int(fb_idx[b])],
        ])

    spec_stats = spec.clamp(-5.0, 5.0)
    stats = torch.cat([fc_stats, spec_stats], dim=-1)           # (B, 18)

    if morph is None:
        return levels, stats

    # ── 形态学证据（CAER-v3）：2 句证据文本 + 14 维连续统计量 ──
    # 原 14 维顺序: ll_mean ll_max sharp_mean sharp_max sharp_focal kurt_mean
    #       kurt_max zcr_mean gamma_pow spike_mean spike_max apf apf_std slope
    m = morph.float().to(device)
    with torch.no_grad():
        sharp_labels = _quantize(
            m[:, 3], [6.0, 15.0], ['low', 'mid', 'high'])
        apf_idx = torch.where(
            m[:, 12] < 0.3, torch.zeros_like(m[:, 12], dtype=torch.long),
            torch.where(m[:, 12] > 0.6,
                        torch.full_like(m[:, 12], 2, dtype=torch.long),
                        torch.ones_like(m[:, 12], dtype=torch.long)))
        apf_words = ['smooth', 'mid', 'sharp']
    for b in range(B):
        levels[b].extend([
            sharp_labels[b], apf_words[int(apf_idx[b])],
        ])

    # 按各特征典型量级缩放（~单位量级供线性头学习），裁剪重尾
    _MORPH_SCALES = torch.tensor(
        [0.3, 0.5, 0.4, 15.0, 4.0, 4.0, 10.0, 0.12, 0.03,
         0.01, 0.03, 9.5, 0.5, 1.5], device=device)
    morph_stats = (m / _MORPH_SCALES).clamp(-5.0, 5.0)
    # ── 分组重组（v4）：PSD 依赖 4 维并入频谱块，形态块只留时域 10 维 ──
    # 最终顺序 = FC 9 | 频谱 13（9 带功率 + gamma/apf/apf_std/slope）| 形态时域 10
    morph_psd = morph_stats[:, [8, 11, 12, 13]]                 # gamma apf apf_std slope
    morph_time = morph_stats[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]
    stats = torch.cat(
        [fc_stats, spec_stats, morph_psd, morph_time], dim=-1)  # (B, 32)
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

# 频谱证据句（CAER-v2；频谱统计量由数据集层确定性计算传入）
_SPECTRAL_TEMPLATES = [
    (" Diffuse slowing is ", ["low", "mid", "high"], "."),
    (" Alpha activity is ", ["low", "mid", "high"], "."),
    (" Theta-beta ratio is ", ["low", "mid", "high"], "."),
    (" Frontal-posterior alpha balance is ", ["front", "even", "back"], "."),
]

# 形态学证据句（CAER-v3；形态统计量由数据集层确定性计算传入）
_MORPH_TEMPLATES = [
    (" Epileptiform sharpness is ", ["low", "mid", "high"], "."),
    (" Alpha rhythm regularity is ", ["smooth", "mid", "sharp"], "."),
]
