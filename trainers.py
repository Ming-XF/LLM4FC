import json
import os
import numpy as np
import torch

from utils import *

import pdb

class BNTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, episode_seed=None):
        super(BNTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id,
                                         episode_seed=episode_seed)

    def prepare_inputs_kwargs(self, inputs):
        node_feature = inputs['correlation']
        labels = inputs['labels']

        return {"node_feature": node_feature.to(self.device),
                "labels": labels.float().to(self.device)}




class BrainNetCNNTrainer(BNTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, episode_seed=None):
        super(BrainNetCNNTrainer, self ).__init__(args, local_rank=local_rank, task_id=task_id,
                                                  episode_seed=episode_seed)




class ALTERTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, episode_seed=None):
        super(ALTERTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id,
                                           episode_seed=episode_seed)

    def prepare_inputs_kwargs(self, inputs):
        node_feature = inputs['correlation']
        labels = inputs['labels']

        return {"node_feature": node_feature.to(self.device),
                "labels": labels.float().to(self.device)}


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

class GCDGCNTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, episode_seed=None):
        super(GCDGCNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id,
                                            episode_seed=episode_seed)

    def prepare_inputs_kwargs(self, inputs, epoch=None):
        node_feature = inputs['correlation']
        labels = inputs['labels']
        stage = "pretrain"
        if epoch is None or epoch > 100:
            stage = "finetune"

        return {"node_feature": node_feature.to(self.device),
                "labels": labels.float().to(self.device),
                "stage": stage}

    def _early_stop_enabled(self, epoch):
        return epoch is not None and epoch > 100

    def train_epoch(self, epoch=None):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, epoch)
            outputs = self._forward(input_kwargs)
            loss = outputs.loss
            self._backward_and_step(loss)
            losses += loss.item()
            loss_list.append(loss.item())
        return losses / len(loss_list)

# ═══════════════════════════════════════════════════════════════════════════════
# TimeLLMTrainer — Time-LLM based DFC prediction with a frozen LLM
# ═══════════════════════════════════════════════════════════════════════════════

class TimeLLMTrainer(Trainer):
    """TimeLLM v2 trainer for dynamic-FC-based dementia classification.

    Architecture: DFC → shared GCN per-window → (B, T*C, 128)
    → Reprogramming (cross-attention to text prototypes) → frozen LLM
    → Flatten → classifier.

    Supports: Single GPU (AMP), Multi-GPU DDP, DeepSpeed ZeRO-2
    """

    def __init__(self, args, local_rank=0, task_id=0, episode_seed=None):
        super().__init__(args, local_rank=local_rank, task_id=task_id,
                         episode_seed=episode_seed)

    def prepare_inputs_kwargs(self, inputs):
        """Extract fields for TimeLLM v2 forward pass.

        Uses dynamic FC (DFC) as main graph input;
        static FC (correlation) for stats prompt.

        按任务类型自适应：
        - 分类：标签 → integer index
        - 回归：标签 → float32
        - 多值回归：DFC → DFC_input（前 k 个窗口），标签 → DFC_target
        """
        labels = inputs['labels']
        if self.data_config.is_classification:
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels.argmax(dim=-1)
            labels = labels.long()
        elif self.data_config.is_regression:
            labels = labels.float()
        kwargs = {
            "DFC": inputs['DFC'].float().to(self.device),
            "SFC": inputs['correlation'].float().to(self.device),
            "labels": labels.to(self.device),
        }
        if 'spectral' in inputs:
            kwargs["spectral"] = inputs['spectral'].float().to(self.device)
        if 'morph' in inputs:
            kwargs["morph"] = inputs['morph'].float().to(self.device)
        return kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# TimeLLMPatchTrainer — raw-EEG Time-LLM patching with a frozen LLM
# ═══════════════════════════════════════════════════════════════════════════════

class TimeLLMPatchTrainer(Trainer):
    """TimeLLM raw-EEG patch trainer (Time-LLM-style reprogramming).

    Architecture: raw EEG (B, 19, T) → per-channel z-score → patch embedding
    → (B, N*P, 128) → Reprogramming (cross-attention to text prototypes)
    → frozen LLM → channel mean-pool → flatten → regressor/classifier.
    """

    def __init__(self, args, local_rank=0, task_id=0, episode_seed=None):
        super().__init__(args, local_rank=local_rank, task_id=task_id,
                         episode_seed=episode_seed)

    def prepare_inputs_kwargs(self, inputs):
        """Extract fields for TimeLLMPatch forward pass.

        Uses the raw EEG time series as the sole input (no DFC/SFC).

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
            "labels": labels.to(self.device),
        }
        return kwargs


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
            "global_fc": inputs['global_fc'].float().to(self.device),
            "global_psd": inputs['global_psd'].float().to(self.device),
            "local_fc": inputs['local_fc'].float().to(self.device),
            "local_psd": inputs['local_psd'].float().to(self.device),
            "local_mask": inputs['local_mask'].bool().to(self.device),
            "labels": labels.to(self.device),
        }
        return kwargs
