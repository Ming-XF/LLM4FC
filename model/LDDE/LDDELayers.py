import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import pdb
    
class E2EBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        
        ab = torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)

        # if torch.isnan(ab).any() or torch.isinf(ab).any():
        #     pdb.set_trace()
        return ab
    
class FeatureAlign(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(256, 512)
        self.dense2 = torch.nn.Linear(512, 1024)
        self.dense3 = torch.nn.Linear(1024, 2048)
        self.dense4 = torch.nn.Linear(2048, 4096)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(4096)
    def forward(self, x):
        out1 = F.dropout(F.leaky_relu(self.bn1(self.dense1(x)), negative_slope=0.33), p=0.5)
        out2 = F.dropout(F.leaky_relu(self.bn2(self.dense2(out1)), negative_slope=0.33), p=0.5)
        out3 = F.dropout(F.leaky_relu(self.bn3(self.dense3(out2)), negative_slope=0.33), p=0.5)
        out4 = F.dropout(F.leaky_relu(self.bn4(self.dense4(out3)), negative_slope=0.33), p=0.5)
        
        return out4
    
class FeatureAlignInverse(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(4096, 2048)
        self.dense2 = torch.nn.Linear(2048, 1024)
        self.dense3 = torch.nn.Linear(1024, 512)
        self.dense4 = torch.nn.Linear(512, 256)
        
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
    def forward(self, x):
        out1 = F.dropout(F.leaky_relu(self.bn1(self.dense1(x)), negative_slope=0.33), p=0.5)
        out2 = F.dropout(F.leaky_relu(self.bn2(self.dense2(out1)), negative_slope=0.33), p=0.5)
        out3 = F.dropout(F.leaky_relu(self.bn3(self.dense3(out2)), negative_slope=0.33), p=0.5)
        out4 = F.dropout(F.leaky_relu(self.bn4(self.dense4(out3)), negative_slope=0.33), p=0.5)
        
        return out4
        
        
    
    
class BrainNetCNN(nn.Module):
    """BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment"""
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
        # pdb.set_trace()
        # node_feature = node_feature.unsqueeze(dim=1)
        out1 = F.leaky_relu(self.bn1(self.e2econv1(node_feature)), negative_slope=0.33)
        out2 = F.leaky_relu(self.bn2(self.e2econv2(out1)), negative_slope=0.33)
        out3 = F.leaky_relu(self.bn3(self.E2N(out2)), negative_slope=0.33)
        out4 = F.dropout(F.leaky_relu(self.bn4(self.N2G(out3)), negative_slope=0.33), p=0.5).squeeze(-1).squeeze(-1)
        # pdb.set_trace()
        # out5 = out4.view(out4.size(0), -1)
        
        
        

        return out4

