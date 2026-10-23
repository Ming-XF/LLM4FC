import json
import os
import numpy as np
import torch

from utils import *

import pdb


class EEGNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, episode_seed=None):
        super(EEGNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id,
                                            episode_seed=episode_seed)

    def prepare_inputs_kwargs(self, inputs):
        labels = inputs['labels']
        if self.data_config.is_classification:
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels.argmax(dim=-1)
            labels = labels.long()
        else:
            labels = labels.float()
        return {"time_series": inputs['time_series'].float().to(self.device),
                "labels": labels.to(self.device)}


# ═══════════════════════════════════════════════════════════════════════════════
# TimeLLMKnowledgeTrainer — raw EEG 知识 Token 编译器
# ═══════════════════════════════════════════════════════════════════════════════

class TimeLLMKnowledgeTrainer(Trainer):
    """Offline FC/PSD plus raw-waveform knowledge compiler trainer.

    Architecture: offline FC/PSD plus raw EEG waveform tokens →
    semantic reprogramming → frozen LLM → last-K-layer READOUT aggregation
    → linear head.
    """

    def __init__(self, args, local_rank=0, task_id=0, episode_seed=None):
        super().__init__(args, local_rank=local_rank, task_id=task_id,
                         episode_seed=episode_seed)

    def prepare_inputs_kwargs(self, inputs):
        """Extract fields for TimeLLMKnowledge forward pass.

        FC and relative PSD are read from the offline feature sidecar.  The raw
        EEG time series is retained only for the PatchStem waveform branch.

        按任务类型自适应：
        - 分类：标签 → integer index
        - 回归：标签 → float32
        """
        labels = inputs['labels']
        if self.data_config.is_classification:
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels.argmax(dim=-1)
            labels = labels.long()
        elif self.data_config.is_regression:
            labels = labels.float()
        kwargs = {
            "x": inputs['time_series'].float().to(self.device),
            "window_fc": inputs['window_fc'].float().to(self.device),
            "window_psd": inputs['window_psd'].float().to(self.device),
            "window_mask": inputs['window_mask'].bool().to(self.device),
            "labels": labels.to(self.device),
        }
        return kwargs
