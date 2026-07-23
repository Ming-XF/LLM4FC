"""DFC-only BNC baseline — BNC 特征提取器 + 简单分类头。

输入 DFC 逐时间窗 (B*L, 1, C, C)，输出 per-window 分类 logits。
隐含层 feat (N, 256) = tech.md 的 F_per_window。
训练完成后，BNC 权重供 LDDE2th 做 warm start（参数名逐字匹配，零映射开销）。

关键约束：必须复用 LDDE2thLayers.BrainNetCNN（非 model/BrainNetCNN 版本），
因为二者内部通道数、BatchNorm、输出维度均不同。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseConfig, ModelOutputs
from ..LDDE2th.LDDE2thLayers import BrainNetCNN

import pdb


class DFCBNCConfig(BaseConfig):
    def __init__(self, node_size, num_classes):
        super().__init__(node_size=node_size, num_classes=num_classes)


class DFCBNC(nn.Module):
    """DFC-only BNC baseline — 输入 DFC 逐窗 (B*L, 1, C, C)，输出 per-window logits。

    通道流程（与 tech.md §4.1 Step 1 对齐）：
        (B*L, 1, 68, 68) → E2E(1→32) → E2E(32→64) → E2N(64→128)
                         → N2G(128→256) → squeeze → (B*L, 256) = F_per_window
    """

    def __init__(self, config: DFCBNCConfig):
        super().__init__()
        self.bnc = BrainNetCNN(config.node_size)          # 复用 LDDE2thLayers
        self.cls = nn.Linear(256, config.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, node_feature, labels):
        pdb.set_trace()
        feat = self.bnc(node_feature)                     # (N, 256) = F_per_window
        logits = self.cls(feat)                           # (N, num_classes)
        loss = self.loss_fn(logits, labels)
        return ModelOutputs(logits=logits, loss=loss)
