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

class DFaSTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(DFaSTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                time_series, y1=labels.float())
            return {"time_series": time_series.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": time_series.to(self.device),
                    "labels": labels.float().to(self.device)}


class DFaSTOnlySpatialTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(DFaSTOnlySpatialTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class FaSPTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(FaSPTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class LMDATrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(LMDATrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class ShallowConvNetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(ShallowConvNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class DeepConvNetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(DeepConvNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BNTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BNTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

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
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(FBNetGenTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

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
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainNetCNNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class DFCBNCTrainer(BNTTrainer):
    """DFC BNC 基线训练器 — DFC 逐窗编码 + 时间池化后 per-patient 分类。

    关键区别 vs BrainNetCNNTrainer：
      - 输入：DFC (B, L, C, C) 而非 SFC correlation (B, C, C)
      - reshape 为 (B*L, 1, C, C) 送入 BNC 逐窗编码
      - 模型内部做 time-mean pool → (B, 256) → (B, num_classes) per-patient 分类
      - BNC 权重产出供 LDDE2th 做 warm start（参数名逐字匹配）
    """
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super().__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        DFC = inputs['DFC']                    # (B, L, C, C)
        labels = inputs['labels']              # (B, num_classes) one-hot
        B, L, C, _ = DFC.shape
        # 逐窗展平为 (B*L, 1, C, C)，保持 B 和 L 供模型做 time-mean pool
        node_feature = DFC.reshape(B * L, 1, C, C).to(self.device)
        return {"node_feature": node_feature,
                "labels": labels.float().to(self.device),
                "B": B,
                "L": L}


class TransformerTrainer(BNTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TransformerTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class STAGINTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(STAGINTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs, **kwargs):
        dyn_a, sampling_points = process_dynamic_fc(inputs['time_series'].transpose(1, 2),
                                                    self.model_config.window_size,
                                                    self.model_config.window_stride,
                                                    # self.model_config.dynamic_length,
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

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, step=step)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.model_config.clip_grad > 0.0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        if self.args.visualize:
            self.visualize()

        from utils.early_stopping import EarlyStopping
        early_stopper = EarlyStopping(
            patience=self.args.early_stop_patience,
            min_delta=self.args.early_stop_min_delta,
            mode='max' if self.args.early_stop_metric != 'Loss' else 'min',
        )

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2

            val_result = self.evaluate(dataloader_key='val')
            monitor_score = val_result.get(self.args.early_stop_metric, 0.0)

            improved = early_stopper.step(monitor_score)
            if improved:
                self.best_result = val_result
                self.save_model()
                logger.info("Best model saved (val_%s=%.4f)",
                            self.args.early_stop_metric, early_stopper.best_score)

            msg = (f"Epoch: {epoch}, Train loss: {train_loss:.5f}, "
                   f"Val loss: {val_result['Loss']:.5f}, "
                   f"Val {self.args.early_stop_metric}: {monitor_score:.4f}, "
                   f"Best: {early_stopper.best_score:.4f}, "
                   f"Time: {(end_time - start_time):.1f}s")
            print(msg)
            logger.info(msg)

            if early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.load_model()
        self.test_result = self.evaluate(dataloader_key='test')
        logger.info("=== Final test result ===")
        for k, v in self.test_result.items():
            if v is not None:
                logger.info(f"  {k}: {v:.5f}")


class EEGNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(EEGNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha,
                # beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}


class EEGChannelNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(EEGChannelNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}


class RACNNTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(RACNNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        # time_series = self.get_complex_morlet_wavelets(time_series)
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}

    @staticmethod
    def get_complex_morlet_wavelets(time_series):
        time_series = time_series.numpy()
        Fa = np.arange(4, 31)
        new_time_series = []
        for i, ts in enumerate(time_series):
            cwt = abs(pywt.cwt(ts, Fa, 'cmor1-1', 1/200)[0])
            new_time_series.append(torch.from_numpy(cwt))
        time_series = torch.stack(new_time_series, dim=0)
        time_series -= time_series.mean()
        time_series /= time_series.std()
        return time_series


class SBLESTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(SBLESTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
        self.W = None
        self.Wh = None

    def load_datasets(self):
        datasets = eval(
            f"{self.args.dataset}Dataset")(self.data_config, k=self.task_id, subject_id=self.subject_id)

        if self.args.do_parallel:
            data_loaders = init_distributed_dataloader(self.data_config, datasets)
        else:
            data_loaders = init_StratifiedKFold_dataloader(self.data_config, datasets)
        return data_loaders

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        labels = ((labels[:, 1] == 1) * 2 - 1)
        idx0 = torch.argwhere(labels == -1)
        idx1 = torch.argwhere(labels == 1)
        idx0 = idx0[:len(idx1)]
        idx = torch.concat([idx0, idx1], dim=0).squeeze()
        time_series = time_series[idx]
        labels = labels[idx].unsqueeze(-1)
        time_series = time_series.permute(1, 2, 0)

        return {"time_series": time_series.double().to(self.device),
                "labels": labels.double().to(self.device)}

    def train(self):
        train_dataloader = self.data_loaders['train']
        self.model.eval()
        inputs = {"time_series": torch.DoubleTensor().to(self.device),
                  "labels": torch.DoubleTensor().to(self.device)}
        for batch in train_dataloader:
            input_kwargs = self.prepare_inputs_kwargs(batch)
            inputs["time_series"] = torch.concat([inputs["time_series"], input_kwargs["time_series"]], dim=-1)
            inputs["labels"] = torch.concat([inputs["labels"], input_kwargs["labels"]], dim=0)
            # break

        self.W, alpha, V, self.Wh = self.model(**inputs)
        self.best_result = self.test_result = self.evaluate()
        # wandb.log({f"Best {k}": v for k, v in self.best_result.items()})

    def evaluate(self):
        test_dataloader = self.data_loaders['test']
        self.model.eval()
        inputs = {"time_series": torch.DoubleTensor().to(self.device),
                  "labels": torch.DoubleTensor().to(self.device)}
        for batch in test_dataloader:
            input_kwargs = self.prepare_inputs_kwargs(batch)
            inputs["time_series"] = torch.concat([inputs["time_series"], input_kwargs["time_series"]], dim=-1)
            inputs["labels"] = torch.concat([inputs["labels"], input_kwargs["labels"]], dim=0)
        R_test, _ = self.model.enhance_conv(inputs["time_series"], self.Wh)
        vec_W = self.W.T.flatten()  # vec operation (Torch)
        preds = R_test @ vec_W
        result = self.metrix(preds, inputs["labels"])
        # wandb.log(result)
        return result

    @staticmethod
    def metrix(predict_Y, Y_test):
        """Compute classification accuracy for test set"""

        predict_Y = predict_Y.cpu().numpy()
        Y_test = torch.squeeze(Y_test).cpu().numpy()
        total_num = len(predict_Y)
        error_num = 0
        auc = roc_auc_score(Y_test*0.5+0.5, predict_Y*0.5+0.5)
        # Compute classification accuracy for test set
        Y_predict = np.zeros(total_num)
        for i in range(total_num):
            if predict_Y[i] > 0:
                Y_predict[i] = 1
            else:
                Y_predict[i] = -1

        # Compute classification accuracy
        for i in range(total_num):
            if Y_predict[i] != Y_test[i]:
                error_num = error_num + 1

        acc = (total_num - error_num) / total_num

        report = classification_report(
            Y_test * 0.5 + 0.5, Y_predict * 0.5 + 0.5, output_dict=True, zero_division=0)
        recall = [0, 0]
        for k, v in report.items():
            if '.' in k:
                recall[int(float(k))] = v['recall']
        specificity = recall[0]
        sensitivity = recall[1]

        result = {"Accuracy": acc,
                  "AUC": auc,
                  "Specificity": specificity,
                  "Sensitivity": sensitivity}
        return result


class TCANetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TCANetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss_global_model = outputs.loss_global_model
            loss_local_and_top = outputs.loss_local_and_top

            for param in self.model.local_network.parameters():
                param.requires_grad = False
            for param in self.model.top_layer.parameters():
                param.requires_grad = False
            loss_global_model.backward(retain_graph=True)

            for param in self.model.local_network.parameters():
                param.requires_grad = True
            for param in self.model.top_layer.parameters():
                param.requires_grad = True
            for param in self.model.global_network.parameters():
                param.requires_grad = False
            loss_local_and_top.backward()
            for param in self.model.global_network.parameters():
                param.requires_grad = True

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss_local_and_top.item()
            loss_list.append(loss_local_and_top.item())
            # wandb.log({'Training loss': loss_local_and_top.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)


class TCACNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TCACNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        wpser = self.wpser(time_series)
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, wpser, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
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

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss_global_model = outputs.loss_global_model
            loss_local_and_top = outputs.loss_local_and_top

            for param in self.model.local_network.parameters():
                param.requires_grad = False
            for param in self.model.top_layer.parameters():
                param.requires_grad = False
            loss_global_model.backward(retain_graph=True)

            for param in self.model.local_network.parameters():
                param.requires_grad = True
            for param in self.model.top_layer.parameters():
                param.requires_grad = True
            for param in self.model.global_network.parameters():
                param.requires_grad = False
            loss_local_and_top.backward()
            for param in self.model.global_network.parameters():
                param.requires_grad = True

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss_global_model.item()
            loss_list.append(loss_global_model.item())
            # wandb.log({'Training loss': loss_global_model.item(),
                       # 'Training local loss': loss_local_and_top.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)


class SteadyNetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(SteadyNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
            
class ALTERTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(ALTERTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

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

class AlzNetV3Trainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(AlzNetV3Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs, **kwargs):
        time_series = inputs['time_series']
        labels = inputs['labels']

        # pdb.set_trace()
        # node_feature = extract_eeg_connectivity_features(time_series, sfreq=250)
        node_feature = extract_simple_eeg_features(time_series, sfreq=250, epoch_length=1)
        
        return {"node_feature": node_feature.to(self.device),
                "labels": labels.float().to(self.device)}

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, step=step)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())

        return losses / len(loss_list)

class GCDGCNTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(GCDGCNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

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
    
    def train_epoch(self, epoch):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, epoch)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        if self.args.visualize:
            self.visualize()

        from utils.early_stopping import EarlyStopping
        early_stopper = EarlyStopping(
            patience=self.args.early_stop_patience,
            min_delta=self.args.early_stop_min_delta,
            mode='max' if self.args.early_stop_metric != 'Loss' else 'min',
        )

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss = self.train_epoch(epoch)
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2

            val_result = self.evaluate(dataloader_key='val')
            monitor_score = val_result.get(self.args.early_stop_metric, 0.0)

            improved = early_stopper.step(monitor_score)
            if improved:
                self.best_result = val_result
                self.save_model()
                logger.info("Best model saved (val_%s=%.4f)",
                            self.args.early_stop_metric, early_stopper.best_score)

            msg = (f"Epoch: {epoch}, Train loss: {train_loss:.5f}, "
                   f"Val loss: {val_result['Loss']:.5f}, "
                   f"Epoch time = {(end_time - start_time):.3f}s")
            print(msg)
            logger.info(msg)

            if early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.load_model()
        self.test_result = self.evaluate(dataloader_key='test')
        logger.info("=== Final test result ===")
        for k, v in self.test_result.items():
            if v is not None:
                logger.info(f"  {k}: {v:.5f}")

class CEEDNetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(CEEDNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

class LDDETrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(LDDETrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        SFC = inputs['correlation']
        DFC = inputs['DFC']
        labels = inputs['labels']
        gender = inputs['gender']
        age = inputs['age']
        education = inputs['education']
        m_label = inputs['m_label']
        # spects = inputs['spects']

        return {"time_series": time_series.float().to(self.device),
                "SFC": SFC.float().to(self.device),
                "DFC": DFC.float().to(self.device),
                # "spects": spects.float().to(self.device),
                "gender": gender.float().to(self.device),
                "age": age.float().to(self.device),
                "education": education.float().to(self.device),
                "labels": labels.float().to(self.device),
                "m_label": m_label.float().to(self.device)}

    
    def train_epoch(self, epoch):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses1 = 0
        loss1_list = []
        losses2 = 0
        loss2_list = []
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            cla_loss, gre_loss = outputs.loss
            loss = cla_loss + gre_loss
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            
            losses1 += cla_loss.item()
            losses2 += gre_loss.item()
            loss1_list.append(cla_loss.item())
            loss2_list.append(gre_loss.item())
            

        return losses1 / len(loss1_list), losses2 / len(loss2_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        if self.args.visualize:
            self.visualize()

        from utils.early_stopping import EarlyStopping
        early_stopper = EarlyStopping(
            patience=self.args.early_stop_patience,
            min_delta=self.args.early_stop_min_delta,
            mode='max' if self.args.early_stop_metric != 'Loss' else 'min',
        )

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss1, train_loss2 = self.train_epoch(epoch)
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2

            val_result = self.evaluate(dataloader_key='val')
            monitor_score = val_result.get(self.args.early_stop_metric, 0.0)

            improved = early_stopper.step(monitor_score)
            if improved:
                self.best_result = val_result
                self.save_model()
                logger.info("Best model saved (val_%s=%.4f)",
                            self.args.early_stop_metric, early_stopper.best_score)

            msg = (f"Epoch: {epoch}, Train Cla: {train_loss1:.5f}, "
                   f"Train Reg: {train_loss2:.5f}, "
                   f"Val {self.args.early_stop_metric}: {monitor_score:.4f}, "
                   f"Best: {early_stopper.best_score:.4f}, "
                   f"Time: {(end_time - start_time):.1f}s")
            print(msg)
            logger.info(msg)

            if early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.load_model()
        self.test_result = self.evaluate(dataloader_key='test')
        logger.info("=== Final test result ===")
        for k, v in self.test_result.items():
            if v is not None:
                logger.info(f"  {k}: {v:.5f}")
            
    def multiple_evaluate(self, dataloader_key='test', inference_mode=None):
        logger.info(f"***** Running evaluation on {dataloader_key}{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders[dataloader_key]
        losses1 = 0
        losses2 = 0
        loss1_list = []
        loss2_list = []
        labels = []
        result = {}
        preds = None
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                if inference_mode is not None:
                    input_kwargs['inference_mode'] = inference_mode
                outputs = self.model(**input_kwargs)
                loss1 = outputs.loss[0]
                loss2 = outputs.loss[1]
                losses1 += loss1.item()
                losses2 += loss2.item()
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())
                # print(f"Evaluate loss: {loss.item():.5f}")
                if preds is None:
                    preds = F.softmax(outputs.logits.float(), dim=1).cpu().numpy()
                else:
                    preds = np.append(preds, F.softmax(outputs.logits.float(), dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            try:
                result['AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}; "
                               f"n_classes_in_labels={len(set(labels))}, "
                               f"n_pred_classes={len(set(np.array(preds).argmax(axis=1)))}, "
                               f"preds_shape={preds.shape}, preds_dtype={preds.dtype}, "
                               f"preds_has_nan={np.isnan(preds).any()}, "
                               f"preds_has_inf={np.isinf(preds).any()}")
                result['AUC'] = 0.
            preds_labels = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds_labels)
            preds_arr, labels_arr = np.array(preds_labels), np.array(labels)
            metric = precision_recall_fscore_support(
                labels_arr, preds_arr, average='macro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]
            result['Classification Loss'] = losses1 / len(loss1_list)
            result['Regression Loss'] = losses2 / len(loss2_list)

            print(f'\n{dataloader_key}{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, Precision:{result["Precision"]:.5f}, '
                  f'AUC:{result["AUC"]:.5f}, Recall:{result["Recall"]:.5f}, F_score:{result["F_score"]:.5f}, Classification Loss:{result["Classification Loss"]:.5f}, Regression Loss:{result["Regression Loss"]:.5f}, '
                  , end=',')

        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        # wandb.log(result)
        return result
    
    
class LDDE2thTrainer(LDDETrainer):
    """LDDE2th 训练器 — 简化版：BNC 特征提取 → LLM 编码 → 分类。

    与父类 LDDETrainer 的关键区别：
      - 损失为单一 CE loss（非 5 元组）
      - 无 SFC 输入（DFC-only）
    """

    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(LDDE2thTrainer, self).__init__(args, local_rank=local_rank,
                                              task_id=task_id, subject_id=subject_id)

        if args.deepspeed:
            import deepspeed
            import json
            with open(args.deepspeed_config) as f:
                ds_config = json.load(f)

            if ds_config.get("bf16", {}).get("enabled"):
                self.model = self.model.bfloat16()

            self.model = self.model.to(self.device)
            self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=self.model,
                config=ds_config,
                model_parameters=[p for p in self.model.parameters() if p.requires_grad],
            )
            self.model = self.engine
            self.model_config.self_train_log_steps = self.args.save_steps
        else:
            self.scaler = torch.cuda.amp.GradScaler()
            self.model_config.self_train_log_steps = self.args.save_steps

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        DFC = inputs['DFC']
        labels = inputs['labels']
        gender = inputs.get('gender', None)
        age = inputs.get('age', None)
        education = inputs.get('education', None)

        if labels.dim() > 1 and labels.shape[-1] > 1:
            labels = labels.argmax(dim=-1)

        kwargs = {"time_series": time_series.float().to(self.device),
                  "DFC": DFC.float().to(self.device),
                  "labels": labels.long().to(self.device)}
        if gender is not None:
            kwargs["gender"] = gender.float().to(self.device)
        if age is not None:
            kwargs["age"] = age.float().to(self.device)
        if education is not None:
            kwargs["education"] = education.float().to(self.device)

        return kwargs

    # ── 单一 CE loss 训练 ──
    def train_epoch(self, epoch):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs)

            if self.args.deepspeed:
                outputs = self.model(**input_kwargs)
            else:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**input_kwargs)

            loss = outputs.loss  # scalar CE loss

            if self.args.deepspeed:
                self.model.backward(loss)
                self.model.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            else:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

            losses += loss.item()
            loss_list.append(loss.item())

            if self.args.max_steps > 0 and step + 1 >= self.args.max_steps:
                break

        return losses / len(loss_list)

    # ── 模型保存 ──
    def save_model(self, path=None, save_optimizer=False):
        if path is None:
            path = os.path.join(self.args.model_dir, self.args.model)
        os.makedirs(path, exist_ok=True)

        do_dp = self.args.do_parallel or self.args.deepspeed
        model = self.model.module if do_dp else self.model

        if self.args.deepspeed:
            self._save_state_dict_zeRO(model, path)
        else:
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            buffer_names = {n for n, _ in model.named_buffers()}
            state_dict = {}
            for k, v in model.state_dict().items():
                if k in trainable_names or k in buffer_names:
                    state_dict[k] = v.cpu()
            torch.save(state_dict,
                       os.path.join(path, f'{self.args.model}-{self.task_id}.bin'))

        do_save = True
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            do_save = False
        if do_save:
            lora_dir = os.path.join(path, f'lora-{self.task_id}')
            model.llm.save_pretrained(lora_dir)

        if save_optimizer and self.optimizer is not None:
            ckpt = {
                'epoch': getattr(self, '_current_epoch', None),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            }
            torch.save(ckpt,
                       os.path.join(path, f'optimizer-{self.task_id}.bin'))

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        args_dict = {k: v for k, v in self.args.__dict__.items()}
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(args_dict, f, indent=2)

        logger.info("Model saved to %s", path)

    def _save_state_dict_zeRO(self, model, path):
        import deepspeed
        trainable = [p for p in model.parameters() if p.requires_grad]
        with deepspeed.zero.GatheredParameters(trainable, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
                buffer_names = {n for n, _ in model.named_buffers()}
                state_dict = {}
                for k, v in model.state_dict().items():
                    if k in trainable_names or k in buffer_names:
                        state_dict[k] = v.cpu()
                torch.save(state_dict,
                           os.path.join(path, f'{self.args.model}-{self.task_id}.bin'))

    def load_model(self, path=None):
        if path is None:
            path = os.path.join(self.args.model_dir, self.args.model)

        do_dp = self.args.do_parallel or self.args.deepspeed
        model = self.model.module if do_dp else self.model
        bin_path = os.path.join(path, f'{self.args.model}-{self.task_id}.bin')

        if not os.path.exists(bin_path):
            logger.info("Model checkpoint doesn't exist: %s — skip loading", bin_path)
            return

        saved_state = torch.load(bin_path, map_location=self.device, weights_only=False)
        current_state = model.state_dict()

        missing, unexpected = [], []
        for k, v in saved_state.items():
            if k in current_state:
                if current_state[k].shape == v.shape:
                    current_state[k].copy_(v)
                else:
                    logger.warning("  Shape mismatch for %s: saved %s vs current %s — skipped",
                                   k, tuple(v.shape), tuple(current_state[k].shape))
            else:
                unexpected.append(k)
        for k in current_state:
            if k not in saved_state:
                missing.append(k)

        n_loaded = len(saved_state) - len(unexpected)
        if missing:
            logger.info("  %d keys not in checkpoint (frozen LLM base, kept from init)", len(missing))
        if unexpected:
            logger.warning("  %d keys in checkpoint but not in model — skipped", len(unexpected))
        logger.info("Model loaded from %s (%d/%d keys)", path, n_loaded, len(saved_state))

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(self.data_loaders['train']))
            logger.info("  Num Epochs = %d", self.args.num_epochs)
            logger.info("  Total train batch size = %d", self.args.batch_size)
            logger.info("  warmup steps = %d", self.args.warmup_steps)
            logger.info("  Total optimization steps = %d", total)
            logger.info("  Save steps = %d", self.args.save_steps)

        if not self.args.deepspeed:
            self.init_components()
        else:
            from utils.schedule import init_deepspeed_schedule
            self.scheduler = init_deepspeed_schedule(self.optimizer, self.args, total)
        if self.args.visualize:
            self.visualize()

        from utils.early_stopping import EarlyStopping
        early_stopper = EarlyStopping(
            patience=self.args.early_stop_patience,
            min_delta=self.args.early_stop_min_delta,
            mode='max' if self.args.early_stop_metric != 'Loss' else 'min',
        )

        best_accuracy = 0.0
        stop_requested = False

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            if self.args.deepspeed:
                train_sampler = getattr(self.data_loaders['train'].sampler, 'set_epoch', None)
                if train_sampler is not None:
                    train_sampler(epoch)

            start_time = timer()
            self._current_epoch = epoch
            train_loss = self.train_epoch(epoch)
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2

            val_result = self.evaluate(dataloader_key='val')

            is_rank_0 = (not torch.distributed.is_initialized()
                         or torch.distributed.get_rank() == 0)
            if is_rank_0:
                monitor_score = val_result.get(self.args.early_stop_metric, 0.0)
                current_acc = val_result.get('Accuracy', 0.0)

                improved = early_stopper.step(monitor_score)
                if improved:
                    best_accuracy = early_stopper.best_score
                    self.best_result = val_result
                    self.save_model()
                    logger.info("Best model saved (val_%s=%.4f)",
                                self.args.early_stop_metric, best_accuracy)

                if self.args.save_steps > 0 and epoch % self.args.save_steps == 0:
                    ckpt_dir = os.path.join(
                        self.args.model_dir, self.args.model,
                        f'checkpoint-epoch-{epoch}')
                    self.save_model(path=ckpt_dir, save_optimizer=True)
                    logger.info("Checkpoint saved at epoch %d", epoch)

                msg = (f"Epoch: {epoch}, Loss: {train_loss:.5f}, "
                       f"Val Acc: {current_acc:.4f}, Best: {early_stopper.best_score:.4f}, "
                       f"Time: {(end_time - start_time):.1f}s")
                print(msg)
                logger.info(msg)

                if early_stopper.early_stop:
                    stop_requested = True

            if self.args.deepspeed and torch.distributed.is_initialized():
                stop_tensor = torch.tensor([1 if stop_requested else 0],
                                           device=self.device, dtype=torch.int)
                torch.distributed.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            elif stop_requested:
                break

        is_rank_0 = (not torch.distributed.is_initialized()
                     or torch.distributed.get_rank() == 0)

        if is_rank_0:
            self.load_model()
            logger.info("=== Final test ===")

        if self.args.deepspeed and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.test_result = self.evaluate(dataloader_key='test')

        if is_rank_0:
            logger.info("=== Final test result ===")
            for k, v in self.test_result.items():
                if v is not None:
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        logger.info(f"  {k}: {v:.5f}")
                    else:
                        logger.info(f"  {k}: {v}")

            final_dir = os.path.join(
                self.args.model_dir, self.args.model,
                f'final-epoch-{self.args.num_epochs}')
            self.save_model(path=final_dir)
            logger.info("Final model saved at epoch %d", self.args.num_epochs)

    # ── 评估：单一 CE loss ──
    def multiple_evaluate(self, dataloader_key='test', inference_mode=None):
        import torch.distributed as dist
        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1

        if rank == 0:
            logger.info(f"***** Running evaluation on {dataloader_key}{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders[dataloader_key]
        losses = 0
        loss_list = []
        labels_local = []
        result = {}
        preds_local = None

        iterator = tqdm(evaluate_dataloader, desc=f"{dataloader_key}-eval-R{rank}",
                        ncols=0)

        with torch.no_grad():
            for inputs in iterator:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                if self.args.deepspeed:
                    outputs = self.model(**input_kwargs)
                else:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**input_kwargs)

                loss = outputs.loss  # scalar
                losses += loss.item()
                loss_list.append(loss.item())

                batch_preds = F.softmax(outputs.logits.float(), dim=1).cpu()
                if preds_local is None:
                    preds_local = batch_preds
                else:
                    preds_local = torch.cat([preds_local, batch_preds], dim=0)
                labels_local += input_kwargs['labels'].cpu().tolist()

        # ── 分布式汇总 ──
        if is_dist and world_size > 1:
            local_count = preds_local.shape[0]
            counts = [torch.zeros(1, dtype=torch.long, device=self.device)
                      for _ in range(world_size)]
            t = torch.tensor([local_count], dtype=torch.long, device=self.device)
            dist.all_gather(counts, t)
            counts = [c.item() for c in counts]
            max_count = max(counts)

            n_classes = preds_local.shape[1]
            if local_count < max_count:
                pad = torch.zeros(max_count - local_count, n_classes,
                                  dtype=preds_local.dtype, device=self.device)
                preds_padded = torch.cat([preds_local.to(self.device), pad], dim=0)
            else:
                preds_padded = preds_local.to(self.device)
            preds_list = [torch.zeros(max_count, n_classes, dtype=preds_padded.dtype,
                                      device=self.device) for _ in range(world_size)]
            dist.all_gather(preds_list, preds_padded)

            labels_t = torch.tensor(labels_local, dtype=torch.long, device=self.device)
            if local_count < max_count:
                pad = torch.full((max_count - local_count,), -1,
                                 dtype=torch.long, device=self.device)
                labels_padded = torch.cat([labels_t, pad], dim=0)
            else:
                labels_padded = labels_t
            labels_list = [torch.zeros(max_count, dtype=torch.long, device=self.device)
                           for _ in range(world_size)]
            dist.all_gather(labels_list, labels_padded)

            all_preds_list = [preds_list[i][:counts[i]].cpu().numpy()
                              for i in range(world_size)]
            preds = np.concatenate(all_preds_list, axis=0)
            all_labels_list = [labels_list[i][:counts[i]].cpu().tolist()
                               for i in range(world_size)]
            labels = [l for sub in all_labels_list for l in sub]

            loss_avg = sum(loss_list) / len(loss_list) if len(loss_list) > 0 else 0.0
            local_losses = torch.tensor([loss_avg, float(local_count)], device=self.device)
            all_losses = [torch.zeros(2, device=self.device) for _ in range(world_size)]
            dist.all_gather(all_losses, local_losses)

            if rank == 0:
                w_sum = 0.0
                w_loss = 0.0
                for i in range(world_size):
                    n = all_losses[i][1].item()
                    w_loss += all_losses[i][0].item() * n
                    w_sum += n
                result['Loss'] = w_loss / w_sum if w_sum > 0 else 0.0
        else:
            preds = preds_local.numpy()
            labels = labels_local
            result['Loss'] = losses / len(loss_list) if len(loss_list) > 0 else 0.0

        # ── 仅 rank 0 计算指标 ──
        if rank == 0:
            try:
                result['AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}; "
                               f"n_classes_in_labels={len(set(labels))}, "
                               f"n_pred_classes={len(set(np.array(preds).argmax(axis=1)))}, "
                               f"preds_shape={preds.shape}, preds_dtype={preds.dtype}, "
                               f"preds_has_nan={np.isnan(preds).any()}, "
                               f"preds_has_inf={np.isinf(preds).any()}")
                result['AUC'] = 0.
            preds_labels = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds_labels)
            preds_arr, labels_arr = np.array(preds_labels), np.array(labels)
            metric = precision_recall_fscore_support(
                labels_arr, preds_arr, average='macro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]

            print(f'\n{dataloader_key}{self.task_id} : Acc:{result["Accuracy"]:.5f}, '
                  f'AUC:{result["AUC"]:.5f}, F1:{result["F_score"]:.5f}, '
                  f'Loss:{result["Loss"]:.5f}',
                  end=', ')

            for k, v in result.items():
                if v is not None:
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        logger.info(f"{k}: {v:.5f}")
                    else:
                        logger.info(f"{k}: {v}")
        else:
            result = {'Accuracy': 0.0}

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# TimeLLMTrainer — Time-LLM based DFC classification (single-stage, no LoRA)
# ═══════════════════════════════════════════════════════════════════════════════

class TimeLLMTrainer(Trainer):
    """TimeLLM v2 trainer for static-FC-based dementia classification.

    Architecture: SFC → adjacency norm → GCN node encoder → Reprogramming
    (cross-attention to text prototypes) → frozen ChatGLM-6B (19 tokens)
    → mean-pool → classifier.

    Supports:
      - Single GPU (AMP)
      - Multi-GPU DDP (``--do_parallel``)
      - Multi-GPU DeepSpeed ZeRO-2 (``--deepspeed``)

    Key differences from LDDE2thTrainer:
      - No LoRA (pure Time-LLM reprogramming paradigm)
      - Static FC input (correlation) only (no DFC, no time_series)
      - No gender/age/education
      - Single-stage joint training
      - Simpler state_dict save/load (no LoRA adapter)
    """

    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super().__init__(args, local_rank=local_rank,
                         task_id=task_id, subject_id=subject_id)

        if args.deepspeed:
            import deepspeed
            import json
            with open(args.deepspeed_config) as f:
                ds_config = json.load(f)

            if ds_config.get("bf16", {}).get("enabled"):
                self.model = self.model.bfloat16()

            self.model = self.model.to(self.device)
            self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=self.model,
                config=ds_config,
                model_parameters=[p for p in self.model.parameters()
                                  if p.requires_grad],
            )
            self.model = self.engine
        else:
            self.scaler = torch.cuda.amp.GradScaler()

    # ── Input preparation ──────────────────────────────────────────

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

    # ── Training loop (AMP / DeepSpeed) ────────────────────────────

    def train_epoch(self, epoch):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(
                tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs)

            if self.args.deepspeed:
                outputs = self.model(**input_kwargs)
            else:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**input_kwargs)

            loss = outputs.loss

            if self.args.deepspeed:
                self.model.backward(loss)
                self.model.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            else:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

            losses += loss.item()
            loss_list.append(loss.item())

            if self.args.max_steps > 0 and step + 1 >= self.args.max_steps:
                break

        return losses / len(loss_list)

    # ── Full training orchestration ────────────────────────────────

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(self.data_loaders['train']))
            logger.info("  Num Epochs = %d", self.args.num_epochs)
            logger.info("  Total train batch size = %d", self.args.batch_size)
            logger.info("  warmup steps = %d", self.args.warmup_steps)
            logger.info("  Total optimization steps = %d", total)
            logger.info("  Save steps = %d", self.args.save_steps)

        if not self.args.deepspeed:
            self.init_components()
        else:
            from utils.schedule import init_deepspeed_schedule
            self.scheduler = init_deepspeed_schedule(self.optimizer, self.args, total)

        from utils.early_stopping import EarlyStopping
        early_stopper = EarlyStopping(
            patience=self.args.early_stop_patience,
            min_delta=self.args.early_stop_min_delta,
            mode='max' if self.args.early_stop_metric != 'Loss' else 'min',
        )

        stop_requested = False

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            if self.args.deepspeed:
                train_sampler = getattr(self.data_loaders['train'].sampler,
                                        'set_epoch', None)
                if train_sampler is not None:
                    train_sampler(epoch)

            start_time = timer()
            self._current_epoch = epoch
            train_loss = self.train_epoch(epoch)
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2

            val_result = self.evaluate(dataloader_key='val')

            is_rank_0 = (not torch.distributed.is_initialized()
                         or torch.distributed.get_rank() == 0)
            if is_rank_0:
                monitor_score = val_result.get(self.args.early_stop_metric, 0.0)
                current_acc = val_result.get('Accuracy', 0.0)

                improved = early_stopper.step(monitor_score)
                if improved:
                    self.best_result = val_result
                    self.save_model()
                    logger.info("Best model saved (val_%s=%.4f)",
                                self.args.early_stop_metric, early_stopper.best_score)

                if self.args.save_steps > 0 and epoch % self.args.save_steps == 0:
                    ckpt_dir = os.path.join(
                        self.args.model_dir, self.args.model,
                        f'checkpoint-epoch-{epoch}')
                    self.save_model(path=ckpt_dir, save_optimizer=True)
                    logger.info("Checkpoint saved at epoch %d", epoch)

                msg = (f"Epoch: {epoch}, Loss: {train_loss:.5f}, "
                       f"Val Acc: {current_acc:.4f}, "
                       f"Best: {early_stopper.best_score:.4f}, "
                       f"Time: {(end_time - start_time):.1f}s")
                print(msg)
                logger.info(msg)

                if early_stopper.early_stop:
                    stop_requested = True

            # Broadcast early-stop decision to all ranks (DeepSpeed / DDP)
            if self.args.deepspeed and torch.distributed.is_initialized():
                stop_tensor = torch.tensor([1 if stop_requested else 0],
                                           device=self.device, dtype=torch.int)
                torch.distributed.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            elif stop_requested:
                break

        is_rank_0 = (not torch.distributed.is_initialized()
                     or torch.distributed.get_rank() == 0)

        if is_rank_0:
            self.load_model()
            logger.info("=== Final test ===")

        if self.args.deepspeed and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.test_result = self.evaluate(dataloader_key='test')

        if is_rank_0:
            logger.info("=== Final test result ===")
            for k, v in self.test_result.items():
                if v is not None:
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        logger.info(f"  {k}: {v:.5f}")
                    else:
                        logger.info(f"  {k}: {v}")

            final_dir = os.path.join(
                self.args.model_dir, self.args.model,
                f'final-epoch-{self.args.num_epochs}')
            self.save_model(path=final_dir)
            logger.info("Final model saved at epoch %d", self.args.num_epochs)

    # ── Evaluation with distributed gathering ──────────────────────

    def multiple_evaluate(self, dataloader_key='test', inference_mode=None):
        import torch.distributed as dist
        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1

        if rank == 0:
            logger.info(f"***** Running evaluation on {dataloader_key}{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders[dataloader_key]
        losses = 0
        loss_list = []
        labels_local = []
        result = {}
        preds_local = None

        iterator = tqdm(evaluate_dataloader,
                        desc=f"{dataloader_key}-eval-R{rank}", ncols=0)

        with torch.no_grad():
            for inputs in iterator:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                if self.args.deepspeed:
                    outputs = self.model(**input_kwargs)
                else:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**input_kwargs)

                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())

                batch_preds = F.softmax(outputs.logits.float(), dim=1).cpu()
                if preds_local is None:
                    preds_local = batch_preds
                else:
                    preds_local = torch.cat([preds_local, batch_preds], dim=0)
                labels_local += input_kwargs['labels'].cpu().tolist()

        # ── Distributed gather across ranks ──
        if is_dist:
            local_count = preds_local.shape[0] if preds_local is not None else 0
            counts = [torch.zeros(1, dtype=torch.long, device=self.device)
                      for _ in range(world_size)]
            t = torch.tensor([local_count], dtype=torch.long, device=self.device)
            dist.all_gather(counts, t)
            counts = [c.item() for c in counts]
            max_count = max(counts)

            n_classes = preds_local.shape[1] if preds_local is not None else 4
            if local_count < max_count and preds_local is not None:
                pad = torch.zeros(max_count - local_count, n_classes,
                                  dtype=preds_local.dtype, device=self.device)
                preds_padded = torch.cat([preds_local.to(self.device), pad], dim=0)
            elif preds_local is not None:
                preds_padded = preds_local.to(self.device)
            else:
                preds_padded = torch.zeros(max_count, n_classes, device=self.device)

            preds_list = [torch.zeros(max_count, n_classes, dtype=preds_padded.dtype,
                                      device=self.device) for _ in range(world_size)]
            dist.all_gather(preds_list, preds_padded)

            labels_t = torch.tensor(labels_local, dtype=torch.long, device=self.device)
            if local_count < max_count:
                pad = torch.full((max_count - local_count,), -1,
                                 dtype=torch.long, device=self.device)
                labels_padded = torch.cat([labels_t, pad], dim=0)
            else:
                labels_padded = labels_t
            labels_list = [torch.zeros(max_count, dtype=torch.long, device=self.device)
                           for _ in range(world_size)]
            dist.all_gather(labels_list, labels_padded)

            all_preds_list = [preds_list[i][:counts[i]].cpu().numpy()
                              for i in range(world_size)]
            preds = np.concatenate(all_preds_list, axis=0)
            all_labels_list = [labels_list[i][:counts[i]].cpu().tolist()
                               for i in range(world_size)]
            labels = [l for sub in all_labels_list for l in sub]

            loss_avg = sum(loss_list) / len(loss_list) if len(loss_list) > 0 else 0.0
            local_losses = torch.tensor([loss_avg, float(local_count)], device=self.device)
            all_losses = [torch.zeros(2, device=self.device) for _ in range(world_size)]
            dist.all_gather(all_losses, local_losses)

            if rank == 0:
                w_sum = 0.0
                w_loss = 0.0
                for i in range(world_size):
                    n = all_losses[i][1].item()
                    w_loss += all_losses[i][0].item() * n
                    w_sum += n
                result['Loss'] = w_loss / w_sum if w_sum > 0 else 0.0
        else:
            preds = preds_local.numpy() if preds_local is not None else np.zeros((0, 4))
            labels = labels_local
            result['Loss'] = losses / len(loss_list) if len(loss_list) > 0 else 0.0

        # ── Metrics (rank 0 only) ──
        if rank == 0:
            try:
                result['AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}; "
                               f"n_classes_in_labels={len(set(labels))}, "
                               f"n_pred_classes={len(set(np.array(preds).argmax(axis=1)))}, "
                               f"preds_shape={preds.shape}")
                result['AUC'] = 0.
            preds_labels = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds_labels)
            preds_arr, labels_arr = np.array(preds_labels), np.array(labels)
            metric = precision_recall_fscore_support(
                labels_arr, preds_arr, average='macro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]

            print(f'\n{dataloader_key}{self.task_id} : Acc:{result["Accuracy"]:.5f}, '
                  f'AUC:{result["AUC"]:.5f}, F1:{result["F_score"]:.5f}, '
                  f'Loss:{result["Loss"]:.5f}',
                  end=', ')

            for k, v in result.items():
                if v is not None:
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        logger.info(f"{k}: {v:.5f}")
                    else:
                        logger.info(f"{k}: {v}")
        else:
            result = {'Accuracy': 0.0}

        return result

    # ── Model persistence (state_dict only, no LoRA) ───────────────

    def save_model(self, path=None, save_optimizer=False):
        if path is None:
            path = os.path.join(self.args.model_dir, self.args.model)
        os.makedirs(path, exist_ok=True)

        do_dp = self.args.do_parallel or self.args.deepspeed
        model = self.model.module if do_dp else self.model

        if self.args.deepspeed:
            self._save_state_dict_zeRO(model, path)
        else:
            trainable_names = {n for n, p in model.named_parameters()
                              if p.requires_grad}
            buffer_names = {n for n, _ in model.named_buffers()}
            state_dict = {}
            for k, v in model.state_dict().items():
                if k in trainable_names or k in buffer_names:
                    state_dict[k] = v.cpu()
            torch.save(state_dict,
                       os.path.join(path, f'{self.args.model}-{self.task_id}.bin'))

        if save_optimizer and self.optimizer is not None:
            ckpt = {
                'epoch': getattr(self, '_current_epoch', None),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            }
            torch.save(ckpt,
                       os.path.join(path, f'optimizer-{self.task_id}.bin'))

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        args_dict = {k: v for k, v in self.args.__dict__.items()}
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(args_dict, f, indent=2)

        logger.info("Model saved to %s", path)

    def _save_state_dict_zeRO(self, model, path):
        """Gather trainable parameters across ZeRO shards, save on rank 0."""
        import deepspeed
        trainable = [p for p in model.parameters() if p.requires_grad]
        with deepspeed.zero.GatheredParameters(trainable, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                trainable_names = {n for n, p in model.named_parameters()
                                  if p.requires_grad}
                buffer_names = {n for n, _ in model.named_buffers()}
                state_dict = {}
                for k, v in model.state_dict().items():
                    if k in trainable_names or k in buffer_names:
                        state_dict[k] = v.cpu()
                torch.save(state_dict,
                           os.path.join(path, f'{self.args.model}-{self.task_id}.bin'))

    def load_model(self, path=None):
        if path is None:
            path = os.path.join(self.args.model_dir, self.args.model)

        do_dp = self.args.do_parallel or self.args.deepspeed
        model = self.model.module if do_dp else self.model
        bin_path = os.path.join(path, f'{self.args.model}-{self.task_id}.bin')

        if not os.path.exists(bin_path):
            logger.info("Model checkpoint doesn't exist: %s — skip loading", bin_path)
            return

        saved_state = torch.load(bin_path, map_location=self.device,
                                weights_only=False)
        current_state = model.state_dict()

        missing, unexpected = [], []
        for k, v in saved_state.items():
            if k in current_state:
                if current_state[k].shape == v.shape:
                    current_state[k].copy_(v)
                else:
                    logger.warning("  Shape mismatch for %s: saved %s vs current %s — skipped",
                                   k, tuple(v.shape), tuple(current_state[k].shape))
            else:
                unexpected.append(k)
        for k in current_state:
            if k not in saved_state:
                missing.append(k)

        n_loaded = len(saved_state) - len(unexpected)
        if missing:
            logger.info("  %d keys not in checkpoint (frozen LLM, kept from init)", len(missing))
        if unexpected:
            logger.warning("  %d keys in checkpoint but not in model — skipped", len(unexpected))
        logger.info("Model loaded from %s (%d/%d keys)", path, n_loaded, len(saved_state))
