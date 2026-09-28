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
# TimeLLMTrainer — Time-LLM based DFC prediction with a frozen LLM
# ═══════════════════════════════════════════════════════════════════════════════

class TimeLLMTrainer(Trainer):
    """TimeLLM v2 trainer for dynamic-FC-based dementia classification.

    Architecture: DFC → shared GCN per-window → (B, T*C, 128)
    → Reprogramming (cross-attention to text prototypes) → frozen LLM
    → readout → classifier.
    Readout: 'flatten' (default, all T*C patch hidden states) or
    'answer_query' (trailing <answer> token hidden state only).

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
        dfc = inputs['DFC'].float().to(self.device)
        if getattr(self.data_config, 'fc_rank_norm', False):
            dfc = self._rank_normalize_dfc(dfc)
        kwargs = {
            "DFC": dfc,
            "SFC": inputs['correlation'].float().to(self.device),
            "labels": labels.to(self.device),
        }
        return kwargs

    @staticmethod
    def _rank_normalize_dfc(dfc):
        """E1: 对每个 DFC 窗口矩阵做边秩变换并映射到 [-1, 1]。

        分布无关（秩统计量），消除不同站点相关分布差异；
        秩对称化保证 (i,j) 与 (j,i) 一致，对角元固定为 1。
        """
        B, W, N, _ = dfc.shape
        flat = dfc.reshape(B, W, -1)                      # (B, W, N*N)
        ranks = flat.argsort(dim=-1).argsort(dim=-1).float()
        ranks = ranks.reshape(B, W, N, N)
        ranks = (ranks + ranks.transpose(-1, -2)) / 2     # 对称化
        ranks = ranks / (N * N - 1)                       # [0, 1]
        ranks = ranks * 2 - 1                             # [-1, 1]
        ranks[..., torch.arange(N), torch.arange(N)] = 1.0
        return ranks
