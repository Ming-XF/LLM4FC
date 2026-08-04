import torch
import torch.nn.functional as F
from torch import nn
from ..base import BaseConfig, ModelOutputs


class E2EBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNNConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 output_dim=2,
                 task_type='classification'):
        super(BrainNetCNNConfig, self).__init__(node_size=node_size,
                                                output_dim=output_dim)
        self.task_type = task_type
        self.output_dim = output_dim


class BrainNetCNN(nn.Module):
    """BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment"""
    def __init__(self, config: BrainNetCNNConfig):
        super().__init__()
        self.in_planes = 1
        self.d = config.node_size

        self.e2econv1 = E2EBlock(1, 32, config.node_size, bias=True)
        self.e2econv2 = E2EBlock(32, 64, config.node_size, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.task_type = getattr(config, 'task_type', 'classification')
        self.dense3 = torch.nn.Linear(30, config.output_dim
                                      if hasattr(config, 'output_dim') else config.output_dim)

        if self.task_type == 'classification':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.MSELoss()

    def forward(self, node_feature: torch.tensor, labels: torch.tensor):
        node_feature = node_feature.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        if self.task_type == 'classification':
            loss = self.loss_fn(out, labels)
        elif self.task_type == 'regression':
            pred = out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out
            loss = self.loss_fn(pred, labels.float().squeeze(-1)
                                if labels.dim() > 1 else labels.float())
        else:  # multi_output_regression
            target_flat = labels.reshape(labels.shape[0], -1).float()
            loss = self.loss_fn(out, target_flat)
        return ModelOutputs(logits=out,
                            loss=loss)
