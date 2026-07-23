import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# BrainNetCNN building blocks
# ═══════════════════════════════════════════════════════════════════════════════


class E2EBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        ab = torch.cat([a]*self.d, 3) + torch.cat([b]*self.d, 2)
        return ab


class BrainNetCNN(nn.Module):
    """BrainNetCNN: Convolutional neural networks for brain networks.

    Used purely as a feature extractor — encodes DFC matrices into
    per-window global features that serve as input to the LLM.
    """
    def __init__(self, node_size):
        super().__init__()
        self.d = node_size

        self.e2econv1 = E2EBlock(1, 32, node_size, bias=True)
        self.e2econv2 = E2EBlock(32, 64, node_size, bias=True)
        self.E2N = torch.nn.Conv2d(64, 128, (1, self.d))
        self.N2G = torch.nn.Conv2d(128, 256, (self.d, 1))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, node_feature: torch.tensor):
        """Forward pass.

        Args:
            node_feature: (N, 1, C, C)  DFC matrices (N = B * L)

        Returns:
            out4: (N, 256)  N2G-compressed global feature per window
        """
        out1 = F.leaky_relu(self.bn1(self.e2econv1(node_feature)), negative_slope=0.33)
        out2 = F.leaky_relu(self.bn2(self.e2econv2(out1)), negative_slope=0.33)
        out3 = F.leaky_relu(self.bn3(self.E2N(out2)), negative_slope=0.33)
        out4 = F.dropout(
            F.leaky_relu(self.bn4(self.N2G(out3)), negative_slope=0.33),
            p=0.5,
        ).squeeze(-1).squeeze(-1)
        return out4


# ═══════════════════════════════════════════════════════════════════════════════
# BNC Feature Projector — maps BNC time-averaged features to LLM prompt tokens
# ═══════════════════════════════════════════════════════════════════════════════


class BNCFeatureProjector(nn.Module):
    """Projects BNC global features into LLM-embedding-space prompt tokens.

    Takes the time-averaged BNC output (B, 256) and produces a set of
    learnable soft tokens (B, num_tokens, llm_dim) that are directly
    concatenated to the LLM system prompt for classification.

    Architecture:
        BNC_feat (B, 256) → Linear → GELU → Linear → (B, num_tokens * llm_dim)
        → reshape → (B, num_tokens, llm_dim)
    """

    def __init__(self, bnc_dim=256, llm_dim=4096, num_tokens=4, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim

        hidden_dim = llm_dim // 4  # 1024

        self.projector = nn.Sequential(
            nn.Linear(bnc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim * num_tokens),
        )

        # Learnable positional bias for each feature token
        self.token_pos_bias = nn.Parameter(torch.zeros(1, num_tokens, llm_dim))
        nn.init.normal_(self.token_pos_bias, std=0.02)

    def forward(self, bnc_feat):
        """
        Args:
            bnc_feat: (B, 256)  BNC time-averaged global feature per sample

        Returns:
            tokens: (B, num_tokens, llm_dim)  soft prompt tokens
        """
        B = bnc_feat.shape[0]
        x = self.projector(bnc_feat)                          # (B, num_tokens * llm_dim)
        tokens = x.reshape(B, self.num_tokens, self.llm_dim)  # (B, num_tokens, llm_dim)
        tokens = tokens + self.token_pos_bias                 # add learnable positions
        return tokens
