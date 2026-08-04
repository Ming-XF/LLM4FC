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
# TimeLLMTrainer — Time-LLM based SFC classification (single-stage, no LoRA)
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
        elif self.data_config.is_multi_output_regression:
            # Next-FC prediction: 使用完整 10 窗口 DFC 作为输入
            # LLM 因果注意力自动实现 next-token prediction —
            # 未来窗口 token 只能 attend 历史窗口，无法看到真实未来
            labels = labels.float()
        return {
            "DFC": inputs['DFC'].float().to(self.device),
            "SFC": inputs['correlation'].float().to(self.device),
            "labels": labels.to(self.device),
        }
