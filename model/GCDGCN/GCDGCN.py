import torch
import torch.nn.functional as F
from torch import nn
import math
from ..base import BaseConfig, ModelOutputs

import pdb


class GCDGCNConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 output_dim=2,
                 layer_sizes=[16, 4],
                 dropout=0.5,
                 readout='fc',
                 embedding_hidden_layers=[240, 240],
                 gamma=0.0001,
                 temperature=0.1,
                 smoothing=0.1,
                 task_type='classification'):
        super(GCDGCNConfig, self).__init__(node_size=node_size, output_dim=output_dim)

        self.node_size=node_size
        self.output_dim=output_dim
        self.layer_sizes=[node_size] + layer_sizes
        self.dropout=dropout
        self.readout=readout
        self.embedding_hidden_layers=embedding_hidden_layers
        self.gamma=gamma
        self.temperature=temperature
        self.smoothing=smoothing
        self.task_type = task_type


class GCDGCN(nn.Module):
    """BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment"""
    def __init__(self, config: GCDGCNConfig):
        super().__init__()
        self.config = config

        # GCD模块 - 独立的前置模块
        self.gcd_embedding = GCDEmbedding(config.node_size, config.embedding_hidden_layers)

        # GCN层，每层都包含GCD嵌入
        self.gcn_layers = nn.ModuleList()
        for i in range(len(config.layer_sizes) - 1):
            self.gcn_layers.append(
               GCNLayer(config.layer_sizes[i], config.layer_sizes[i + 1])
            )

        # 输出头
        if config.readout == 'fc':
            fc_input_dim = config.node_size * config.layer_sizes[-1]
        else:  # 'mean', 'sum', 'max'
            fc_input_dim = config.layer_sizes[-1]

        self.task_type = config.task_type
        self.fc = nn.Linear(fc_input_dim, config.output_dim)

    def normalize_adj(self, adj):
        """归一化邻接矩阵（计算归一化拉普拉斯）"""
        # 添加自环
        adj_with_self = adj + torch.eye(self.config.node_size, device=adj.device, dtype=adj.dtype).unsqueeze(0)

        degree = torch.sum(adj, dim=-1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5).diag_embed()
        laplacian = torch.eye(self.config.node_size, device=adj.device, dtype=adj.dtype) - \
                   torch.matmul(degree_inv_sqrt, torch.matmul(adj, degree_inv_sqrt))
        return laplacian

    def forward(self, node_feature, labels, stage='pretrain'):
        """
        统一前向传播，根据stage计算不同损失

        Args:
            adj: 原始邻接矩阵 (batch_size, node_size, node_size)
            labels: 标签 (batch_size)，计算损失时需要
            stage: 'pretrain' 或 'finetune'

        Returns:
            pretrain: (contrastive_loss, logits)
            finetune: (logits, classification_loss)
        """

        if self.task_type == 'classification':
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = torch.argmax(labels, dim=1)

        B, C, _ = node_feature.shape
        device = node_feature.device
        
        
        x = torch.eye(C, device=device).unsqueeze(0).repeat(B, 1, 1).to(dtype=node_feature.dtype)  # (batch, 16, 16)
        adj = node_feature


         # 步骤1: GCD去噪
        adj_denoised, embedding_vec = self.gcd_embedding(adj)
        
        # 步骤2: 归一化邻接矩阵
        adj_normalized = self.normalize_adj(adj_denoised)
        
        # GCN前向传播
        current_x = x
        for i, layer in enumerate(self.gcn_layers):
            current_x = layer(current_x, adj_normalized)
            
            # 除最后一层外都使用ReLU和Dropout
            if i < len(self.gcn_layers) - 1:
                current_x = F.relu(current_x)
                current_x = F.dropout(current_x, self.config.dropout, training=self.training)
        
        # Readout
        if self.config.readout == 'fc':
            final_output = current_x.view(current_x.size(0), -1)
        elif self.config.readout == 'mean':
            final_output = torch.mean(current_x, dim=1)
        elif self.config.readout == 'sum':
            final_output = torch.sum(current_x, dim=1)
        elif self.config.readout == 'max':
            final_output = torch.max(current_x, dim=1)[0]
        final_output = self.fc(final_output)
        
        # 根据阶段计算损失
        if stage == 'pretrain':
            loss = self._contrastive_loss(embedding_vec, labels)
            return ModelOutputs(logits=final_output, loss=loss)
        elif stage == 'finetune':
            if self.task_type == 'classification':
                loss = self._classification_loss(final_output, labels)
            elif self.task_type == 'regression':
                pred = final_output.squeeze(-1) if final_output.dim() > 1 and final_output.shape[-1] == 1 else final_output
                lbl = labels.float().view(-1) if labels.dim() > 1 else labels.float()
                loss = F.mse_loss(pred, lbl)
            else:  # multi_output_regression
                target_flat = labels.reshape(labels.shape[0], -1).float()
                loss = F.mse_loss(final_output, target_flat)
            return ModelOutputs(logits=final_output, loss=loss)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _contrastive_loss(self, embeddings, labels):
        """
        预训练对比损失（InfoNCE Loss）
        使用第一层GCD的嵌入向量
        """
        batch_size = embeddings.size(0)
        
        # 1. 重建邻接矩阵
        idx = torch.triu_indices(self.config.node_size, self.config.node_size, 1)
        adj = torch.zeros(batch_size, self.config.node_size, self.config.node_size,
                          device=embeddings.device, dtype=embeddings.dtype)
        adj[:, idx[0], idx[1]] = embeddings
        adj[:, idx[1], idx[0]] = embeddings
        
        # 2. 计算归一化拉普拉斯矩阵的特征值（全局特征）
        degree = torch.sum(adj, dim=-1)
        degree_inv_sqrt = torch.pow(degree, -0.5).diag_embed()
        laplacian = torch.eye(self.config.node_size, device=adj.device, dtype=adj.dtype).unsqueeze(0) - \
                   torch.matmul(degree_inv_sqrt, torch.matmul(adj, degree_inv_sqrt))
        
        eigenvalues = torch.linalg.eigvalsh(laplacian.float())
        eigenvalues, _ = torch.sort(eigenvalues, dim=1)
        eigenv = eigenvalues[:, 1:]  # 忽略λ₁=0
        
        # 3. 计算全局特征相似度（RBF核）
        sigma = eigenv.std(dim=0, unbiased=False) + 1e-8
        gamma_vec = 1 / (4 * sigma ** 2)
        
        dist_sq = (eigenv.unsqueeze(1) - eigenv.unsqueeze(0)) ** 2
        weighted_dist_sq = dist_sq * gamma_vec.unsqueeze(0).unsqueeze(0)
        similarity_global = torch.exp(-weighted_dist_sq).mean(dim=-1)
        
        # 4. 计算局部特征相似度（余弦相似度）
        similarity_local = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        # 5. 加权总相似度
        similarity = similarity_global * self.config.gamma + similarity_local
        
        # 6. InfoNCE Loss
        similarity = similarity / self.config.temperature
        labels_expanded = labels.unsqueeze(0)
        
        # 正样本掩码（同类别但不是自身）
        pos_mask = (labels_expanded == labels_expanded.T) & \
                  ~torch.eye(batch_size, device=embeddings.device).bool()
        
        pos_exp = torch.exp(similarity) * pos_mask.float()
        all_exp = torch.exp(similarity).sum(dim=1, keepdim=True)
        
        loss = -torch.log((pos_exp.sum(dim=1) + 1e-8) / (all_exp.squeeze() + 1e-8)).mean()
        
        return loss

    def _classification_loss(self, logits, labels):
        """
        微调分类损失（标签平滑交叉熵）
        """
        confidence = 1.0 - self.config.smoothing
        true_dist = torch.full_like(logits, self.config.smoothing / (logits.size(1) - 1))
        true_dist.scatter_(1, labels.unsqueeze(1), confidence)
        
        return F.kl_div(F.log_softmax(logits, dim=1), true_dist, reduction='batchmean')
        



class GCDEmbedding(nn.Module):
    """
    GCD嵌入模块 - 独立的前置模块
    将原始邻接矩阵映射为去噪后的邻接矩阵
    """
    def __init__(self, node_size, hidden_layers):
        super(GCDEmbedding, self).__init__()
        self.node_size = node_size
        self.input_dim = node_size * (node_size - 1) // 2
        
        # MLP映射网络
        layers = []
        in_dim = self.input_dim
        
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        
        # 输出层：映射回原始维度
        layers.append(nn.Linear(in_dim, self.input_dim))
        layers.append(nn.Sigmoid())  # 确保输出在0-1之间
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, adj):
        """
        输入: adj (batch_size, node_size, node_size)
        输出: 
          - adj_denoised: 去噪后的邻接矩阵
          - embedding_vec: 嵌入向量（用于对比学习）
        """
        # 取上三角元素扁平化
        idx = torch.triu_indices(self.node_size, self.node_size, 1)
        x = adj[:, idx[0], idx[1]]  # (batch_size, input_dim)
        
        # MLP处理
        embedding_vec = self.mlp(x)
        
        # 重建去噪后的邻接矩阵
        adj_denoised = torch.zeros_like(adj, dtype=embedding_vec.dtype)
        adj_denoised[:, idx[0], idx[1]] = embedding_vec
        adj_denoised[:, idx[1], idx[0]] = embedding_vec
        
        return adj_denoised, embedding_vec


class GCNLayer(nn.Module):
    """
    标准GCN层（不使用GCD嵌入）
    GCD在外部作为前置模块
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x, adj_normalized):
        """
        输入:
          - x: 节点特征 (batch_size, node_size, in_features)
          - adj_normalized: 归一化的邻接矩阵（来自GCD）
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_normalized, support)
        return output
