import torch
import torch.nn.functional as F
from torch import nn
from ..base import BaseConfig, ModelOutputs

import pdb


class AlzNetV3Config(BaseConfig):
    def __init__(self,
                 node_size,
                 num_classes,
                 in_channels=4,
                 out_channels=[256, 512, 1024],
                 hidden_layer=512,
                 nm=15):
        super(AlzNetV3Config, self).__init__(node_size=node_size, num_classes=num_classes)

        self.node_size=node_size
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.hidden_layer=hidden_layer
        self.nm=nm


class AlzNetV3(nn.Module):
    """BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment"""
    def __init__(self, config: AlzNetV3Config):
        super().__init__()
        self.config = config

        self.droprate = 0.25
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=config.in_channels,out_channels=config.out_channels[0],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(config.out_channels[0]),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=config.out_channels[0],out_channels=config.out_channels[0],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(config.out_channels[0]),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
            nn.Conv2d(in_channels=config.out_channels[0],out_channels=config.out_channels[1],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(config.out_channels[1]),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=config.out_channels[1],out_channels=config.out_channels[1],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(config.out_channels[1]),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
            nn.Conv2d(in_channels=config.out_channels[1],out_channels=config.out_channels[2],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(config.out_channels[2]),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=config.out_channels[2],out_channels=config.out_channels[2],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(config.out_channels[2]),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
        )
        self.fully_connected_stack = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=config.hidden_layer * config.nm),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=config.hidden_layer * config.nm,out_features=config.num_classes),
            nn.Softmax(dim=1),
        )
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=config.out_channels[2],out_features=config.hidden_layer),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=config.hidden_layer,out_features=config.num_classes),
            nn.Softmax(dim=1),
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, node_feature: torch.tensor, labels: torch.tensor):
        # pdb.set_trace()
        x = self.cnn_relu_stack(node_feature)
        out = self.global_pooling(x)

        loss = self.loss_fn(out, labels)
        return ModelOutputs(logits=out,
                            loss=loss)
