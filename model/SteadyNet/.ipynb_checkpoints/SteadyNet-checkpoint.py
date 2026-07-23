import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot


from ..base import BaseConfig, ModelOutputs

import pdb

class SteadyNetConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 channel_depth1=24,
                 channel_depth2=9):
        super(SteadyNetConfig, self).__init__(node_size=node_size,
                                         node_feature_size=node_feature_size,
                                         time_series_size=time_series_size,
                                         num_classes=num_classes)
        self.channel_depth1 = channel_depth1
        self.channel_depth2 = channel_depth2



class SteadyNet(nn.Module):
    def __init__(self, config: SteadyNetConfig):
        super(SteadyNet, self).__init__()
        
        self.config = config
        
        self.conv1 = nn.Conv2d(1, config.channel_depth1, kernel_size=(3,3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2,21))
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(config.channel_depth1, config.channel_depth2, kernel_size=(3,3), padding='same')
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20*(config.node_size//2)*(config.time_series_size//21), 80)  # Calculate input features based on previous layers
        self.fc2 = nn.Linear(80, 50)
        self.fc3 = nn.Linear(50, config.num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))
        
    def forward(self, time_series, labels):
        time_series = time_series.unsqueeze(1)
        
        x = self.relu(self.conv1(time_series))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.conv2(x))
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.softmax(self.fc3(x))
        
        loss = self.loss_fn(logits, labels)
        
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss)
        else:
            return logits, loss