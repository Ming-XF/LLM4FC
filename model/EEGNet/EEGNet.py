import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseConfig, ModelOutputs


class EEGNetConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 time_series_size,
                 output_dim,
                 task_type='classification',
                 dropout=0.5,
                 frequency=128,
                 D=2,
                 num_kernels=8,
                 p1=4,
                 p2=8):
        super(EEGNetConfig, self).__init__(node_size=node_size,
                                           time_series_size=time_series_size,
                                           output_dim=output_dim,
                                           dropout=dropout)
        self.task_type = task_type
        self.frequency = frequency
        self.num_kernels = num_kernels
        self.D = D
        self.p1 = p1
        self.p2 = p2


# ── 以下三个模块移植自历史 commit c99a3a4 的 model/DFaST/ ──
#    BaseFrequencyModule ← MVA.py / BaseSpatialModule ← DCA.py /
#    BaseTemporalModule ← LTSA.py（原 EEGNet 依赖 DFaST 的复用模块）──

class BaseFrequencyModule(nn.Module):
    """
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung and B. J. Lance

    Journal of neural engineering 2018 Vol. 15 Issue 5 Pages 056013
    """
    def __init__(self, config):
        super(BaseFrequencyModule, self).__init__()
        self.config = config
        kern_length = config.frequency // 2
        self.padding_flag = False if kern_length % 2 else True

        # Layer 1
        self.padding = nn.ZeroPad2d((0, 1, 0, 0))
        self.conv = nn.Conv2d(1, config.num_kernels, (1, kern_length), padding=(0, (kern_length-1)//2), bias=False)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels, False)

    def forward(self, hidden_state):
        hidden_state = self.padding(hidden_state) if self.padding_flag else hidden_state
        hidden_state = self.batch_norm(self.conv(hidden_state))
        return hidden_state


class BaseSpatialModule(nn.Module):
    """
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung and B. J. Lance

    Journal of neural engineering 2018 Vol. 15 Issue 5 Pages 056013
    """
    def __init__(self, config, input_node_size=30, output_node_size=5, pooling=True):
        super(BaseSpatialModule, self).__init__()
        self.config = config
        self.dw_conv = nn.Conv2d(config.num_kernels, config.num_kernels * output_node_size, (input_node_size, 1),
                                 bias=False, groups=config.num_kernels)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels * output_node_size, False)
        self.pooling = nn.AvgPool2d((1, config.p1)) if pooling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_state):
        spatial_feature = F.elu(self.batch_norm(self.dw_conv(hidden_state)))
        if self.pooling is not None:
            spatial_feature = self.pooling(spatial_feature)
        spatial_feature = self.dropout(spatial_feature)
        return spatial_feature


class BaseTemporalModule(nn.Module):
    """
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung and B. J. Lance

    Journal of neural engineering 2018 Vol. 15 Issue 5 Pages 056013
    """
    def __init__(self, config, D=5, pooling=True):
        super(BaseTemporalModule, self).__init__()
        self.config = config
        kern_length = config.frequency // 2
        self.padding = nn.ZeroPad2d((0, 1, 0, 0))
        self.separable_conv = nn.Conv2d(config.num_kernels * D, config.num_kernels * D, (1, int(kern_length*0.5)),
                                        padding=(0, int(kern_length*0.5)//2-1), bias=False)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels * D, False)
        self.pooling = nn.AvgPool2d((1, config.p2)) if pooling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_state):
        hidden_state = self.padding(hidden_state)
        hidden_state = F.elu(self.batch_norm(self.separable_conv(hidden_state)))
        if self.pooling is not None:
            hidden_state = self.pooling(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = hidden_state.squeeze(2)
        return hidden_state


class EEGNet(nn.Module):
    """
    A Reproduction of EEGNet:
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung and B. J. Lance

    Journal of neural engineering 2018 Vol. 15 Issue 5 Pages 056013

    Ported from the historical commit c99a3a4 implementation, adapted to the
    current harness: raw logits output (no softmax inside), cross_entropy /
    huber_loss selected by config.task_type.
    """
    def __init__(self, config: EEGNetConfig):
        super(EEGNet, self).__init__()
        self.config = config
        self.input_node_size = config.node_size
        self.output_node_size = config.D
        self.time_feature_size = config.num_kernels * self.output_node_size
        self.output_time_series_size = int(config.time_series_size//config.p1//config.p2)

        self.frequency_layer = BaseFrequencyModule(config)

        self.spatial_layer = BaseSpatialModule(config,
                                               input_node_size=self.input_node_size,
                                               output_node_size=self.output_node_size)

        self.temporal_layer = BaseTemporalModule(config, D=self.output_node_size)
        hidden_size = config.num_kernels * config.D * int(config.time_series_size // config.p1 // config.p2)

        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.output_dim)
        )

    def forward(self, time_series, labels):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.frequency_layer(time_series)
        hidden_state = self.spatial_layer(hidden_state)
        hidden_state = self.temporal_layer(hidden_state)
        hidden_state = hidden_state.reshape(hidden_state.size(0), -1)
        logits = self.fc(hidden_state)  # raw logits: eval applies softmax, loss applies CE

        if self.config.task_type == 'regression':
            prediction = logits.squeeze(-1)
            loss = F.huber_loss(prediction.float(), labels.float(), delta=1.0)
        else:
            loss = F.cross_entropy(logits.float(), labels)
        return ModelOutputs(logits=logits,
                            loss=loss,
                            hidden_state=hidden_state)
