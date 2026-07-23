import torch
import torch.nn as nn
import numpy as np
from ..base import BaseConfig, ModelOutputs

import pdb


class CEEDNetConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 base_channels: int = 64,
                 dropout: float = 0.5,
                 batch_norm: bool = True,
                 fc_stages: int = 2,
                 base_pool: str = "max",
                 final_pool: str = "average",
                 activation: str = "relu"):
        super(CEEDNetConfig, self).__init__(node_size=node_size,
                                          node_feature_size=node_feature_size,
                                          time_series_size=time_series_size,
                                          num_classes=num_classes)
        self.base_channels = base_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.fc_stages = fc_stages
        self.base_pool = base_pool
        self.final_pool = final_pool
        self.activation = activation


def get_activation_class(activation_type: str):
    """获取激活函数类"""
    if activation_type == "relu":
        return nn.ReLU
    elif activation_type == "gelu":
        return nn.GELU
    elif activation_type == "mish":
        return nn.Mish
    elif activation_type == "tanh":
        return nn.Tanh
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


def program_conv_filters(
    sequence_length: int,
    conv_filter_list: list,
    output_lower_bound: int = 4,
    output_upper_bound: int = 8,
    pad: bool = True,
    stride_to_pool_ratio: float = 1.00,
    trials: int = 5,
    verbose: bool = False,
):
    """
    自动计算卷积层参数，确保输出尺寸在合理范围内
    """
    mid = (output_upper_bound + output_lower_bound) / 2.0
    in_out_ratio = float(sequence_length) / mid

    base_stride = np.power(
        in_out_ratio / np.prod([cf["kernel_size"] for cf in conv_filter_list], dtype=np.float64),
        1.0 / len(conv_filter_list),
    )

    for i in range(len(conv_filter_list)):
        cf = conv_filter_list[i]
        if i == 0 and len(conv_filter_list) > 1:
            total_stride = max(1.0, base_stride * cf["kernel_size"] * 0.7)
            cf["pool"] = max(1, round(np.sqrt(total_stride / stride_to_pool_ratio) * stride_to_pool_ratio * 0.3))
            cf["stride"] = max(1, round(total_stride / cf["pool"]))
        else:
            total_stride = max(1.0, base_stride * cf["kernel_size"])
            if stride_to_pool_ratio > 1.0:
                cf["pool"] = min(
                    max(1, round(np.sqrt(total_stride / stride_to_pool_ratio) * stride_to_pool_ratio)),
                    round(total_stride),
                )
                cf["stride"] = max(1, round(total_stride / cf["pool"]))
            else:
                cf["stride"] = min(max(1, round(np.sqrt(total_stride / stride_to_pool_ratio))), round(total_stride))
                cf["pool"] = max(1, round(total_stride / cf["stride"]))
        
        conv_filter_list[i] = cf

    success = False
    str_debug = f"\n{'-'*100}\nstarting from sequence length: {sequence_length}\n{'-'*100}\n"
    current_length = sequence_length

    for k in range(trials):
        if success:
            break

        for pivot in reversed(range(len(conv_filter_list))):
            current_length = sequence_length

            for cf in conv_filter_list:
                current_length = current_length // cf.get("pool", 1)
                str_debug += f"{cf} >> {current_length} "

                effective_kernel_size = (cf["kernel_size"] - 1) * cf.get("dilation", 1)
                both_side_pad = 2 * (cf["kernel_size"] // 2) if pad is True else 0
                current_length = (current_length + both_side_pad - effective_kernel_size - 1) // cf["stride"] + 1
                str_debug += f">> {current_length}\n"

            pool = conv_filter_list[pivot]["pool"]
            stride = conv_filter_list[pivot]["stride"]
            
            if current_length < output_lower_bound:
                if float(pool) / stride < stride_to_pool_ratio:
                    if stride > 1:
                        conv_filter_list[pivot]["stride"] = max(1, stride - 1)
                    else:
                        conv_filter_list[pivot]["pool"] = max(1, pool - 1)
                else:
                    if pool > 1:
                        conv_filter_list[pivot]["pool"] = max(1, pool - 1)
                    else:
                        conv_filter_list[pivot]["stride"] = max(1, stride - 1)
            elif current_length > output_upper_bound:
                if float(pool) / stride < stride_to_pool_ratio:
                    conv_filter_list[pivot]["pool"] = pool + 1
                else:
                    conv_filter_list[pivot]["stride"] = stride + 1
            else:
                str_debug += f">> Success!"
                success = True
                break

            str_debug += f">> Failed.."
            str_debug += f"\n{'-' * 100}\n"

    if verbose:
        print(str_debug)

    if not success:
        raise RuntimeError(
            f"program_conv_filters() failed to determine proper convolution filter parameters. "
            f"Debug info: {str_debug}"
        )

    return current_length


class CEEDNet(nn.Module):
    """
    2D-VGG-19模型用于EEG分类
    改编自 torchvision VGG 实现，适配EEG时序数据
    
    VGG-19架构: [2层x64, 2层x128, 4层x256, 4层x512, 4层x512]
    
    输入: [batch_size, node_size, time_series_size] -> 转换为 [batch_size, 1, node_size, time_series_size]
    """
    def __init__(self, config: CEEDNetConfig):
        super(CEEDNet, self).__init__()
        self.config = config
        
        # 参数验证
        if config.base_pool not in ["average", "max"] or config.final_pool not in ["average", "max"]:
            raise ValueError("base_pool and final_pool must be one of ['average', 'max']")
        
        if config.fc_stages < 1:
            raise ValueError("fc_stages must be >= 1")

        # 设置池化层
        if config.base_pool == "average":
            self.base_pool = nn.AvgPool2d
        else:
            self.base_pool = nn.MaxPool2d

        # 获取激活函数
        self.nn_act = get_activation_class(config.activation)
        
        # VGG-19层配置 (直接硬编码，无需外部字典)
        # 格式: (卷积层数, 通道数乘数)
        self.layer_cfgs = [
            {"layers": 2, "channel_mul": 1},   # 64 channels
            {"layers": 2, "channel_mul": 2},   # 128 channels  
            {"layers": 4, "channel_mul": 4},   # 256 channels
            {"layers": 4, "channel_mul": 8},   # 512 channels
            {"layers": 4, "channel_mul": 8},   # 512 channels
        ]
        
        # 计算卷积参数
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3改为2，要不然卷积核太大
        conv_filter_list = [{"kernel_size": 2} for _ in self.layer_cfgs]
        
        self.output_length = program_conv_filters(
            sequence_length=config.time_series_size,
            conv_filter_list=conv_filter_list,
            output_lower_bound=4,
            output_upper_bound=8,
            stride_to_pool_ratio=1.5,
        )

        # 构建5个卷积阶段
        self.current_channels = 1  # 输入通道数
        
        self.conv_stage1 = self._make_conv_stage(conv_filter_list[0], self.layer_cfgs[0], config.base_channels)
        self.conv_stage2 = self._make_conv_stage(conv_filter_list[1], self.layer_cfgs[1], config.base_channels)
        self.conv_stage3 = self._make_conv_stage(conv_filter_list[2], self.layer_cfgs[2], config.base_channels)
        self.conv_stage4 = self._make_conv_stage(conv_filter_list[3], self.layer_cfgs[3], config.base_channels)
        self.conv_stage5 = self._make_conv_stage(conv_filter_list[4], self.layer_cfgs[4], config.base_channels)

        # 最终池化层
        if config.final_pool == "average":
            self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.final_pool = nn.AdaptiveMaxPool2d((1, 1))

        # 全连接阶段
        fc_stage = []
        for i in range(config.fc_stages - 1):
            if config.batch_norm:
                layer = nn.Sequential(
                    nn.Linear(self.current_channels, self.current_channels // 2, bias=False),
                    nn.Dropout(p=config.dropout),
                    nn.BatchNorm1d(self.current_channels // 2),
                    self.nn_act(),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(self.current_channels, self.current_channels // 2, bias=True),
                    nn.Dropout(p=config.dropout),
                    self.nn_act(),
                )
            self.current_channels = self.current_channels // 2
            fc_stage.append(layer)

        fc_stage.append(nn.Linear(self.current_channels, config.num_classes, bias=True))
        # fc_stage.append(nn.Linear(self.current_channels // 2, config.num_classes, bias=True))
        self.fc_stage = nn.Sequential(*fc_stage)

        # 初始化权重
        self.reset_weights()
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            weight=torch.tensor(config.class_weight) if config.class_weight is not None else None
        )

    def reset_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _make_conv_stage(self, conv_filter, cfg, base_channels):
        """构建单个卷积阶段"""
        conv_layers = []

        if conv_filter["pool"] > 1:
            conv_layers += [self.base_pool(conv_filter["pool"])]

        for k in range(cfg["layers"]):
            stride = conv_filter["stride"] if k == 0 else 1

            if self.config.batch_norm:
                conv_layers += [
                    nn.Conv2d(
                        in_channels=self.current_channels,
                        out_channels=cfg["channel_mul"] * base_channels,
                        kernel_size=conv_filter["kernel_size"],
                        padding=conv_filter["kernel_size"] // 2,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(cfg["channel_mul"] * base_channels),
                    self.nn_act(),
                ]
            else:
                conv_layers += [
                    nn.Conv2d(
                        in_channels=self.current_channels,
                        out_channels=cfg["channel_mul"] * base_channels,
                        kernel_size=conv_filter["kernel_size"],
                        padding=conv_filter["kernel_size"] // 2,
                        stride=stride,
                        bias=True,
                    ),
                    self.nn_act(),
                ]

            self.current_channels = cfg["channel_mul"] * base_channels
        
        return nn.Sequential(*conv_layers)

    def forward(self, time_series, labels):
        """
        前向传播
        
        Args:
            time_series: [batch_size, node_size, time_series_size] 或 [batch_size, time_series_size]
            labels: [batch_size]
        """
        x = time_series.unsqueeze(1)
        
        # 卷积阶段
        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)
        x = self.conv_stage5(x)
        
        x = self.final_pool(x)
        x = torch.flatten(x, 1)
        
        logits = self.fc_stage(x)
        
        loss = self.loss_fn(logits, labels)
        
        if self.config.dict_output:
            return ModelOutputs(logits=logits, loss=loss)
        else:
            return logits, loss

    def get_output_length(self):
        return self.output_length

    def get_num_fc_stages(self):
        return self.config.fc_stages