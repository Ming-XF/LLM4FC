"""分段（局部）EEG 统计量的 torch 实现（与 data/dataset.py 的 numpy 版同公式）。

供 TimeLLMKnowledge 局部知识块在 forward 内对 raw EEG 计算：整分钟切成
6 个 10s 段，每段独立算 9 维频谱 + 14 维形态统计量，量纲与 numpy 版一致
（/scales 缩放与 ±5 裁剪交给 compute_graph_evidence 统一处理）。

频谱窗：4s、50% overlap（10s 段内 4 窗）；按 codex 评审建议先聚合
band power 再算比值，避免短段内每窗比值不稳定。
"""

import torch

_BANDS = {'d': (0.5, 4.0), 't': (4.0, 8.0), 'a': (8.0, 13.0), 'b': (13.0, 30.0)}
_FRONTAL = [0, 1, 5, 6, 10, 13]
_POSTERIOR = [3, 4, 8, 9, 12, 15]
_TEMPORAL = [11, 12, 14, 15]
_EPS = 1e-8


def _welch_windows(x, hz):
    """x: (B, C, T) → (W, B, C, F) 各 4s 窗（50% overlap）的 PSD + 频率轴。"""
    nperseg = 4 * hz
    step = nperseg // 2
    T = x.shape[-1]
    nwin = (T - nperseg) // step + 1
    freqs = torch.fft.rfftfreq(nperseg, 1.0 / hz, device=x.device)
    win = torch.hann_window(nperseg, periodic=False, device=x.device,
                            dtype=x.dtype)
    psd = []
    for k in range(nwin):
        w = x[..., k * step:k * step + nperseg]
        w = (w - w.mean(-1, keepdim=True)) / (w.std(-1, keepdim=True) + _EPS)
        psd.append(torch.abs(torch.fft.rfft(w * win, dim=-1)) ** 2)
    return torch.stack(psd, 0), freqs                    # (W, B, C, F)


def segment_spectral_stats(x, hz):
    """x: (B, C, T) → (B, 9) 频谱统计量（与 numpy 版同顺序/量纲）。

    相对功率 delta/theta/alpha/beta、theta/beta、alpha/(d+t)、慢波指数
    (d+t)/(a+b)、额/枕 alpha 比、颞区 theta 占比。
    """
    psd, freqs = _welch_windows(x, hz)
    masks = {bn: (freqs >= lo) & (freqs < hi) for bn, (lo, hi) in _BANDS.items()}
    # 先聚合 band power（跨窗求和）再算比值
    pw = {bn: psd[:, :, :, m].sum(-1).sum(0) for bn, m in masks.items()}  # (B, C)
    total = sum(pw[bn].mean(1) for bn in _BANDS) + 1e-12                 # (B,)
    rel = {bn: pw[bn].mean(1) / total for bn in _BANDS}
    fa = pw['a'][:, _FRONTAL].sum(1)
    pa = pw['a'][:, _POSTERIOR].sum(1)
    tt = pw['t'][:, _TEMPORAL].sum(1)
    return torch.stack([
        rel['d'], rel['t'], rel['a'], rel['b'],
        rel['t'] / rel['b'].clamp_min(1e-9),
        rel['a'] / (rel['d'] + rel['t']).clamp_min(1e-9),
        (rel['d'] + rel['t']) / (rel['a'] + rel['b']).clamp_min(1e-9),
        fa / pa.clamp_min(1e-9),
        tt / total,
    ], dim=-1)                                                        # (B, 9)


def segment_morph_stats(x, hz):
    """x: (B, C, T) → (B, 14) 形态学统计量（与 numpy 版同顺序/量纲）。

    ll_mean/ll_max、sharp_mean/max/focal、kurt_mean/max、zcr、gamma_pow、
    spike_mean/max、apf、apf_std、slope。std 用总体标准差（correction=0，
    与 numpy np.std 一致）。
    """
    B, C, T = x.shape
    x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + _EPS)
    d1 = x.diff(dim=-1).abs()
    d2 = x.diff(n=2, dim=-1).abs()
    ll = d1.sum(-1) / (T - 1)                                          # (B, C)
    sharp = d2.mean(-1)
    kurt = ((x - x.mean(-1, keepdim=True)) ** 4).mean(-1) / (
        (x ** 2).mean(-1) ** 2 + 1e-12) - 3.0
    zcr = ((x > 0).int().diff(dim=-1) != 0).sum(-1).float() / (T - 1)
    mad = d2.median(-1).values / 0.6745
    spike = (d2 > 6 * mad.unsqueeze(-1)).sum(-1).float() / (T - 2)

    psd, freqs = _welch_windows(x, hz)                                 # (W, B, C, F)
    gmask = (freqs >= 30) & (freqs < 45)
    amask = (freqs >= 7) & (freqs < 13)
    smask = (freqs >= 2) & (freqs < 40)
    tot = psd.sum(-1) + 1e-12                                          # (W, B, C)
    gamma = (psd[:, :, :, gmask].sum(-1) / tot).mean(0).mean(1)        # (B,)
    apf_per = freqs[amask][psd[:, :, :, amask].argmax(-1)].mean(2)     # (W, B)
    apf = apf_per.mean(0)
    lf = freqs[smask].log()
    A = torch.stack([lf, torch.ones_like(lf)], 1)
    slopes = []
    for k in range(psd.shape[0]):
        b = (psd[k, :, :, smask].mean(1) + 1e-12).log()                # (B, Fs)
        sol = torch.linalg.lstsq(A, b.T).solution                      # (2, B)
        slopes.append(sol[0])
    slope = torch.stack(slopes).mean(0)                                # (B,)

    return torch.stack([
        ll.mean(1), ll.max(1).values,
        sharp.mean(1), d2.max(-1).values.max(1).values,
        d2.max(-1).values.std(1, correction=0),
        kurt.mean(1), kurt.max(1).values,
        zcr.mean(1),
        gamma,
        spike.mean(1), spike.max(1).values,
        apf, apf_per.std(0, correction=0),
        slope,
    ], dim=-1)                                                        # (B, 14)
