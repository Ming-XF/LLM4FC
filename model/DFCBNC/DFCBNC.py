"""DFC-only BNC baseline — BNC 特征提取器 + 时间池化 + 简单分类头。

输入 DFC 逐时间窗 (B*L, 1, C, C)，输出 per-patient 分类 logits (B, num_classes)。
隐含层流程完全对齐 tech.md §4.1 Step 1：
    BNC(F_per_window) → (B*L, 256) → reshape → (B, L, 256) → time-mean pool → HB (B, 256)
训练完成后，BNC 权重供 LDDE2th 做 warm start（参数名逐字匹配，零映射开销）。

关键约束：必须复用 LDDE2thLayers.BrainNetCNN（非 model/BrainNetCNN 版本），
因为二者内部通道数、BatchNorm、输出维度均不同。
"""

import torch
import torch.nn as nn

from ..base import BaseConfig, ModelOutputs
from ..LDDE2th.LDDE2thLayers import BrainNetCNN


class DFCBNCConfig(BaseConfig):
    def __init__(self, node_size, num_classes):
        super().__init__(node_size=node_size, num_classes=num_classes)


class DFCBNC(nn.Module):
    """DFC-only BNC baseline — 输入 DFC 逐窗 (B*L, 1, C, C)，输出 per-patient logits。

    通道流程（与 tech.md §4.1 Step 1 对齐）：
        (B*L, 1, 68, 68) → E2E(1→32) → E2E(32→64) → E2N(64→128)
                         → N2G(128→256) → squeeze → F_per_window (B*L, 256)
        F_per_window → reshape(B, L, 256) → time-mean pool → HB (B, 256)
        HB → cls(256→num_classes) → logits (B, num_classes)
    """

    def __init__(self, config: DFCBNCConfig):
        super().__init__()
        self.bnc = BrainNetCNN(config.node_size)       # 复用 LDDE2thLayers
        self.cls = nn.Linear(256, config.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, node_feature, labels, B, L):
        """前向传播 — 逐窗编码 + 时间池化 + 分类。

        Args:
            node_feature: (B*L, 1, C, C)  逐时间窗 FC 矩阵
            labels:       (B, num_classes) 或 (B,) per-patient 标签
            B:            batch size（患者数）
            L:            time windows per patient
        Returns:
            ModelOutputs with logits (B, num_classes) and loss
        """
        # Step 1: BNC 逐窗编码 → F_per_window (B*L, 256)
        feat = self.bnc(node_feature)                     # (B*L, 256)

        # Step 2: 时间池化 → HB (B, 256)
        HB = feat.reshape(B, L, -1).mean(dim=1)           # (B, 256)

        # Step 3: 分类
        logits = self.cls(HB)                             # (B, num_classes)
        loss = self.loss_fn(logits, labels.argmax(dim=-1) if labels.dim() > 1 else labels)
        return ModelOutputs(logits=logits, loss=loss)
