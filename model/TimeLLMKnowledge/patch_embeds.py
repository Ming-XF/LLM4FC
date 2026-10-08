"""分组 patch embedding：FC 图 / 相对功率 PSD → 组统计量 token。

FCEmbed：静态 FC 相关矩阵 (B,1,19,19) → 2D CNN → (B, 9, 128)
SpecEmbed：相对功率 PSD (B,1,19,F) → 2D CNN → (B, 13, 128)
两个网络相互独立（不共用类、不共用权重）——FC 相关矩阵与频谱矩阵
结构不同，各自定义。

相对 PSD：复用 local_stats._welch_windows（4s hanning 窗、50% overlap），
每窗每通道除以自身总功率（相对化、幅度不变）→ 跨窗平均 → 取 0.5-45Hz。
全局整分钟与 10s 局部段共用同一 helper。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_stats import _welch_windows

_EPS = 1e-12


def relative_psd_matrix(x, hz, fmin=0.5, fmax=45.0):
    """x: (B, C, T) → (B, C, F) 相对功率谱。

    每窗每通道除以自身总功率（相对化），跨窗平均，取 [fmin, fmax) 频段。
    """
    psd, freqs = _welch_windows(x, hz)                  # (W, B, C, F)
    total = psd.sum(-1) + _EPS                          # (W, B, C)
    rel = (psd / total.unsqueeze(-1)).mean(0)           # (B, C, F)
    mask = (freqs >= fmin) & (freqs < fmax)
    return rel[:, :, mask]


class FCEmbed(nn.Module):
    """静态 FC 相关矩阵 (B,1,19,19) → 2D CNN → (B, 9, 128)。"""

    def __init__(self, out_tokens, d_model, mid=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.head = nn.Linear(mid, out_tokens * d_model)
        self.out_tokens = out_tokens
        self.d_model = d_model

    def forward(self, x):
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = F.adaptive_avg_pool2d(h, (1, 1)).squeeze(-1).squeeze(-1)
        return self.head(h).view(h.shape[0], self.out_tokens, self.d_model)


class SpecEmbed(nn.Module):
    """相对功率 PSD (B,1,19,F) → 2D CNN → (B, 13, 128)。"""

    def __init__(self, out_tokens, d_model, mid=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.head = nn.Linear(mid, out_tokens * d_model)
        self.out_tokens = out_tokens
        self.d_model = d_model

    def forward(self, x):
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = F.adaptive_avg_pool2d(h, (1, 1)).squeeze(-1).squeeze(-1)
        return self.head(h).view(h.shape[0], self.out_tokens, self.d_model)
