"""分组 patch embedding：FC 图 / 相对功率 PSD → 三方面特征。

FCEmbed：静态 FC 相关矩阵 (B,1,19,19) → 非对称核 2D CNN(k=(1,6)/(6,1)，
通道全程 32) + 逐方向 AdaptiveAvgPool1d(2/4) → (B,32,2,4) → flatten(256)
→ RMSNorm → Linear → (B, d_model)
SpecEmbed：相对功率 PSD (B,19,F) → 1D CNN(k=6,s=6) + AdaptiveAvgPool1d(8)
→ flatten(256) → RMSNorm → Linear → (B, d_model)（与 PatchStem 同构）
两个网络相互独立（不共用类、不共用权重）。

relative_psd_matrix 保留为在线参考实现；当前 TimeLLMKnowledge 主路径读取
data/knowledge_preprocess.py 生成的离线相对 PSD。两者均使用 4s Hann 窗、
50% overlap，
每窗每通道除以自身总功率（相对化、幅度不变）→ 跨窗平均 → 取 0.5-45Hz。
全局整分钟与 10s 局部段共用同一 helper。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .knowledge_compiler import rms_norm
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
    """静态 FC 相关矩阵 (B,1,19,19) → 非对称核 2D CNN → (B, d_model)。

    Conv2d(1→32, k=(1,6)) → GELU → AdaptiveAvgPool1d(2)（沿通道对维）
    → Conv2d(32→32, k=(6,1)) → GELU → AdaptiveAvgPool1d(4)（沿通道维）
    → (B,32,2,4) → flatten(256) → RMSNorm → Linear(256→d_model)。

    forward 返回 (tokens, feat)：feat = head 之前的 (B,256) 组特征，
    供融合路径使用。
    """

    def __init__(self, d_model, mid=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, mid, (1, 6))
        self.conv2 = nn.Conv2d(mid, mid, (6, 1))
        self.head = nn.Linear(mid * 2 * 4, d_model)
        self.d_model = d_model

    @staticmethod
    def _pool1d(h, out_len):
        """对 4D 特征图沿最后一维做 AdaptiveAvgPool1d(out_len)。"""
        b, c, hh, ww = h.shape
        p = F.adaptive_avg_pool1d(h.flatten(0, 1), out_len)  # (B·C, hh, out)
        return p.unflatten(0, (b, c))                        # (B, C, hh, out)

    def forward(self, x):
        h = F.gelu(self.conv1(x))                     # (B, 32, 19, 14)
        h = self._pool1d(h, 2)                        # 沿 W 维 → (B, 32, 19, 2)
        h = F.gelu(self.conv2(h))                     # (B, 32, 14, 2)
        h = self._pool1d(h.permute(0, 1, 3, 2), 4)    # 沿 H 维 → (B, 32, 2, 4)
        feat = rms_norm(h.flatten(1))                 # (B, 256) 组特征(分化前)
        token = self.head(feat)
        return token, feat


class SpecEmbed(nn.Module):
    """相对功率 PSD (B,19,F) → 1D CNN → (B, d_model)（与 PatchStem 同构）。

    Conv1d(19→32, k=6, stride=6) → GELU → AdaptiveAvgPool1d(8)
    → flatten(256) → RMSNorm → Linear(256→d_model)。

    forward 返回 (tokens, feat)：feat = head 之前的 (B,256) 组特征，
    供融合路径使用。
    """

    def __init__(self, d_model, mid=32, in_channels=19, kernel=6):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, mid, kernel, stride=kernel)
        self.head = nn.Linear(mid * 8, d_model)
        self.d_model = d_model

    def forward(self, x):
        h = F.gelu(self.conv(x))                      # (B, 32, ~F/6)
        h = F.adaptive_avg_pool1d(h, 8)               # (B, 32, 8)
        feat = rms_norm(h.flatten(1))                 # (B, 256) 组特征(分化前)
        token = self.head(feat)
        return token, feat
