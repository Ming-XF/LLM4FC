import json
import os
import numpy as np
import pywt
import torch
from einops import repeat, rearrange
from scipy import signal
import numpy as np

from utils import *

import pdb

class BNTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0):
        super(BNTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']

        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                time_series, node_feature, y1=labels.float())
            return {"node_feature": node_feature.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"node_feature": node_feature.to(self.device),
                    "labels": labels.float().to(self.device)}


class FBNetGenTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0):
        super(FBNetGenTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        time_series_size = time_series.shape[-1] // self.model_config.window_size * self.model_config.window_size
        time_series = time_series[:, :, :time_series_size]
        node_feature = inputs['correlation']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                time_series, node_feature, y1=labels.float())
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.float().to(self.device)}


class BrainNetCNNTrainer(BNTTrainer):
    def __init__(self, args, local_rank=0, task_id=0):
        super(BrainNetCNNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id)


class STAGINTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0):
        super(STAGINTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id)

    def prepare_inputs_kwargs(self, inputs, **kwargs):
        dyn_a, sampling_points = process_dynamic_fc(inputs['time_series'].transpose(1, 2),
                                                    self.model_config.window_size,
                                                    self.model_config.window_stride,
                                                    self.model_config.sampling_init)
        sampling_endpoints = [p + self.model_config.window_size for p in sampling_points]
        dyn_v = repeat(torch.eye(self.model_config.node_size), 'n1 n2 -> b t n1 n2', t=len(sampling_points),
                       b=self.args.batch_size)
        if len(dyn_a) < self.args.batch_size:
            dyn_v = dyn_v[:len(dyn_a)]
        return {'v': dyn_v.to(self.device),
                'a': dyn_a.to(self.device),
                't': inputs['time_series'].permute(2, 0, 1).to(self.device),
                'sampling_endpoints': sampling_endpoints,
                'labels': inputs['labels'].float().to(self.device)}

    def train_epoch(self, epoch=None):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, step=step)
            outputs = self._forward(input_kwargs)
            loss = outputs.loss
            if self.model_config.clip_grad > 0.0:
                if self.args.deepspeed:
                    torch.nn.utils.clip_grad_value_(self.model.module.parameters(), self.model_config.clip_grad)
                else:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)
            self._backward_and_step(loss)

            losses += loss.item()
            loss_list.append(loss.item())

        return losses / len(loss_list)


class TCACNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0):
        super(TCACNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        wpser = self.wpser(time_series)
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, wpser, labels, _ = continues_mixup_data(
                time_series, wpser, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "node_feature": wpser.to(self.device),
                "labels": labels.float().to(self.device)}

    @staticmethod
    def wpser(time_series):
        fs = 200
        lowcut = 8
        highcut = 30
        order = 4
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band', output='ba')
        time_series = signal.filtfilt(b, a, time_series)
        _, n, _ = time_series.shape
        coeffs = pywt.wavedec(time_series, 'db4', level=5)
        energy = np.array([np.square(level).sum(-1) for level in coeffs])
        wpser = energy / np.repeat(np.expand_dims(energy.sum(-1), -1), n, -1)
        wpser = wpser.sum(0)
        wpser = torch.from_numpy(wpser).float()
        return wpser

    def train_epoch(self, epoch=None):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self._forward(input_kwargs)
            loss_global_model = outputs.loss_global_model
            loss_local_and_top = outputs.loss_local_and_top

            if self.args.deepspeed:
                # DeepSpeed ZeRO cannot handle per-param freeze in
                # dual-gradient training — combine losses as single step.
                total_loss = loss_global_model + loss_local_and_top
                self.model.backward(total_loss)
                self.model.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            else:
                # Freeze local/top, backward global only
                for param in self.model.local_network.parameters():
                    param.requires_grad = False
                for param in self.model.top_layer.parameters():
                    param.requires_grad = False
                loss_global_model.backward(retain_graph=True)

                # Unfreeze local/top, freeze global, backward local+top
                for param in self.model.local_network.parameters():
                    param.requires_grad = True
                for param in self.model.top_layer.parameters():
                    param.requires_grad = True
                for param in self.model.global_network.parameters():
                    param.requires_grad = False
                loss_local_and_top.backward()
                self.optimizer.step()
                self.scheduler.step()
                for param in self.model.global_network.parameters():
                    param.requires_grad = True

            losses += loss_global_model.item()
            loss_list.append(loss_global_model.item())

        return losses / len(loss_list)


class ALTERTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0):
        super(ALTERTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']

        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                time_series, node_feature, y1=labels.float())
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.float().to(self.device)}

class GCDGCNTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0):
        super(GCDGCNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id)

    def prepare_inputs_kwargs(self, inputs, epoch=None):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']
        stage = "pretrain"
        if epoch is None or epoch > 100:
            stage = "finetune"

        return {"node_feature": node_feature.to(self.device),
                "labels": labels.float().to(self.device),
                "stage": stage}

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
    """TimeLLM v2 trainer for static-FC-based dementia classification.

    Architecture: SFC → adjacency norm → GCN node encoder → Reprogramming
    (cross-attention to text prototypes) → frozen ChatGLM-6B (19 tokens)
    → mean-pool → classifier.

    Supports: Single GPU (AMP), Multi-GPU DDP, DeepSpeed ZeRO-2
    """

    def __init__(self, args, local_rank=0, task_id=0):
        super().__init__(args, local_rank=local_rank, task_id=task_id)

    def prepare_inputs_kwargs(self, inputs):
        """Extract fields for TimeLLM v2 forward pass.

        Uses static FC (correlation) as input; labels are converted to
        class indices inside the model forward.
        """
        labels = inputs['labels']
        if labels.dim() > 1 and labels.shape[-1] > 1:
            labels = labels.argmax(dim=-1)
        return {
            "SFC": inputs['correlation'].float().to(self.device),
            "labels": labels.long().to(self.device),
        }
