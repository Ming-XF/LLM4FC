import json
import os
from timeit import default_timer as timer
# import wandb
import logging
import torch
import numpy as np
from abc import abstractmethod
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

from config import init_model_config
from .optimizer import init_optimizer
from .schedule import init_schedule
from .accuracy import accuracy
from data import *
from data.dataloader import init_deepspeed_dataloader

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, args, local_rank=0, task_id=0):
        self.task_id = task_id
        self.args = args
        self.local_rank = local_rank
        self.data_config = DataConfig(args)
        self.data_loaders = self.load_datasets()

        model, self.model_config = init_model_config(args, self.data_config)
        if args.deepspeed:
            import deepspeed
            import json as _json
            self.device = f'cuda:{self.local_rank}' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            with open(args.deepspeed_config) as f:
                ds_config = _json.load(f)
            self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=model.to(self.device),
                config=ds_config,
                model_parameters=[p for p in model.parameters() if p.requires_grad],
            )
            self.model = self.engine
            self._ds_config = ds_config
        elif args.do_parallel:
            self.device = f'cuda:{self.local_rank}' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   find_unused_parameters=True)
        else:
            self.device = f'cuda' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
            self.scaler = torch.cuda.amp.GradScaler()

        self.scheduler = None if args.deepspeed else self.scheduler
        self.best_result = None
        self.test_result = None

    @abstractmethod
    def prepare_inputs_kwargs(self, inputs):
        return {}

    def load_datasets(self):
        datasets = eval(
            f"{self.args.dataset}Dataset")(self.data_config, k=self.task_id)

        if self.args.deepspeed:
            data_loaders = init_deepspeed_dataloader(self.data_config, datasets)
        elif self.args.do_parallel:
            data_loaders = init_distributed_dataloader(self.data_config, datasets)
        else:
            data_loaders = init_StratifiedKFold_dataloader(self.data_config, datasets)
        return data_loaders

    def init_components(self):
        if self.args.deepspeed:
            return  # engine manages optimizer & scheduler
        total = self.args.num_epochs * len(self.data_loaders['train'])
        self.optimizer = init_optimizer(self.model, self.args)
        self.scheduler = init_schedule(self.optimizer, self.args, total)

    def _forward(self, input_kwargs):
        """Return model outputs, using autocast for mixed precision."""
        if self.args.deepspeed:
            return self.model(**input_kwargs)
        else:
            with torch.cuda.amp.autocast():
                return self.model(**input_kwargs)

    def _early_stop_enabled(self, epoch):
        """Hook for subclasses to skip early stopping in certain phases (e.g. pretrain)."""
        return True

    def _backward_and_step(self, loss):
        """Backward + step, dispatching to DeepSpeed or AMP scaler."""
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

    def train_epoch(self, epoch=None):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self._forward(input_kwargs)
            loss = outputs.loss
            self._backward_and_step(loss)
            losses += loss.item()
            loss_list.append(loss.item())
            if self.args.max_steps > 0 and step + 1 >= self.args.max_steps:
                break
        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        is_rank_0 = (not torch.distributed.is_initialized()
                     or torch.distributed.get_rank() == 0)
        if is_rank_0:
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
        metric_name = self.args.early_stop_metric
        mode = 'min' if metric_name == 'Loss' else 'max'
        early_stopper = EarlyStopping(
            patience=self.args.early_stop_patience,
            min_delta=self.args.early_stop_min_delta,
            mode=mode,
        )

        best_test_result = None
        stop_requested = False

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch", ncols=0):
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
            test_result = self.evaluate(dataloader_key='test')

            if is_rank_0:
                val_loss = val_result.get('Loss', float('inf'))
                val_metric = val_result.get(metric_name, float('inf') if mode == 'min' else 0.0)

                if self._early_stop_enabled(epoch):
                    improved = early_stopper.step(val_metric)
                    if improved:
                        self.best_result = val_result
                        best_test_result = test_result
                        self.save_model()
                        logger.info("Best model saved (val_%s=%.4f, test_acc=%.4f)",
                                    metric_name, early_stopper.best_score,
                                    test_result.get('Accuracy', 0.0))

                    if early_stopper.early_stop:
                        stop_requested = True
                else:
                    improved = False

                if self.args.save_steps > 0 and epoch % self.args.save_steps == 0:
                    ckpt_dir = os.path.join(
                        self.args.model_dir, self.args.model,
                        f'checkpoint-epoch-{epoch}')
                    self.save_model(path=ckpt_dir, save_optimizer=True)
                    logger.info("Checkpoint saved at epoch %d", epoch)

                msg = (f"Epoch: {epoch}, Loss: {train_loss:.5f}, "
                       f"Val loss: {val_loss:.5f}, Test loss: {test_result.get('Loss', float('inf')):.5f}, "
                       f"Val {metric_name}: {val_metric:.4f}, Best val {metric_name}: {early_stopper.best_score:.4f}, "
                       f"No improve: {early_stopper.counter}/{early_stopper.patience}, "
                       f"Time: {(end_time - start_time):.1f}s")
                print(msg)
                logger.info(msg)

            if self.args.deepspeed and torch.distributed.is_initialized():
                stop_tensor = torch.tensor([1 if stop_requested else 0],
                                           device=self.device, dtype=torch.int)
                torch.distributed.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            elif stop_requested:
                break

        if is_rank_0:
            self.load_model()
            self.test_result = best_test_result
            logger.info("=== Best epoch test result ===")
            if self.test_result is not None:
                for k, v in self.test_result.items():
                    if v is not None:
                        if isinstance(v, (int, float, np.floating, np.integer)):
                            logger.info(f"  {k}: {v:.5f}")
                        else:
                            logger.info(f"  {k}: {v}")
            else:
                logger.info("  (no test result recorded)")

            final_dir = os.path.join(
                self.args.model_dir, self.args.model,
                f'final-epoch-{self.args.num_epochs}')
            self.save_model(path=final_dir)
            logger.info("Final model saved at epoch %d", self.args.num_epochs)

    def evaluate(self, dataloader_key='test'):
        if self.data_config.num_class == 2:
            result = self.binary_evaluate(dataloader_key)
        else:
            result = self.multiple_evaluate(dataloader_key)
        return result

    def binary_evaluate(self, dataloader_key='test'):
        import torch.distributed as dist
        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1

        if rank == 0:
            logger.info(f"***** Running evaluation on {dataloader_key} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders[dataloader_key]
        losses = 0
        loss_list = []
        labels_local = []
        preds_local = None
        result = {}

        iterator = tqdm(evaluate_dataloader,
                        desc=f"{dataloader_key}-eval-R{rank}", ncols=0)

        with torch.no_grad():
            for inputs in iterator:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self._forward(input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())

                batch_preds = F.softmax(outputs.logits.float(), dim=1)[:, 1]  # (B,) 正类概率
                if preds_local is None:
                    preds_local = batch_preds
                else:
                    preds_local = torch.cat([preds_local, batch_preds], dim=0)

                lbl = input_kwargs['labels']
                if lbl.dim() == 1:
                    labels_local += lbl.tolist()           # class indices
                else:
                    labels_local += lbl[:, 1].tolist()     # one-hot → 正类

        # ── Distributed gather across ranks ──
        if is_dist:
            local_count = preds_local.shape[0] if preds_local is not None else 0
            counts = [torch.zeros(1, dtype=torch.long, device=self.device)
                      for _ in range(world_size)]
            t = torch.tensor([local_count], dtype=torch.long, device=self.device)
            dist.all_gather(counts, t)
            counts = [c.item() for c in counts]
            max_count = max(counts)

            if local_count < max_count and preds_local is not None:
                pad = torch.zeros(max_count - local_count,
                                  dtype=preds_local.dtype, device=self.device)
                preds_padded = torch.cat([preds_local.to(self.device), pad], dim=0)
            elif preds_local is not None:
                preds_padded = preds_local.to(self.device)
            else:
                preds_padded = torch.zeros(max_count, device=self.device)

            preds_list = [torch.zeros(max_count, dtype=preds_padded.dtype,
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
            preds = preds_local.numpy() if preds_local is not None else np.zeros(0)
            labels = labels_local
            result['Loss'] = losses / len(loss_list) if len(loss_list) > 0 else 0.0

        # ── Metrics (rank 0 only) ──
        if rank == 0:
            try:
                result['AUC'] = roc_auc_score(labels, preds)
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}")
                result['AUC'] = 0.
            preds_bin = (np.array(preds) > 0.5).astype(int)
            labels_arr = np.array(labels)
            result['Accuracy'] = (preds_bin == labels_arr).sum() / len(labels)
            result['Precision'] = precision_recall_fscore_support(
                labels_arr, preds_bin, average="binary")[0]

            report = classification_report(
                labels_arr, preds_bin, output_dict=True, zero_division=0)
            recall = [0, 0]
            for k, v in report.items():
                if k.isdigit():
                    recall[int(float(k))] = v['recall']
            result['Specificity'] = recall[0]
            result['Sensitivity'] = recall[1]

            print()
            print(f'{dataloader_key}{self.task_id} : Loss:{result["Loss"]:.5f}, '
                  f'Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}, '
                  f'Precision:{result["Precision"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}, '
                  f'Specificity:{result["Specificity"]:.5f}')
            for k, v in result.items():
                if v is not None:
                    logger.info(f"{k}: {v:.5f}")
        else:
            result = {'Accuracy': 0.0}

        return result

    def multiple_evaluate(self, dataloader_key='test'):
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
                outputs = self._forward(input_kwargs)

                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())

                batch_preds = F.softmax(outputs.logits.float(), dim=1).cpu()
                if preds_local is None:
                    preds_local = batch_preds
                else:
                    preds_local = torch.cat([preds_local, batch_preds], dim=0)
                lbl = input_kwargs['labels']
                if lbl.dim() == 2 and lbl.shape[-1] > 1:
                    labels_local += lbl.argmax(dim=-1).cpu().tolist()
                else:
                    labels_local += lbl.cpu().tolist()

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

            print()
            print(f'{dataloader_key}{self.task_id} : Acc:{result["Accuracy"]:.5f}, '
                  f'AUC:{result["AUC"]:.5f}, F1:{result["F_score"]:.5f}, '
                  f'Precision:{result["Precision"]:.5f}, Recall:{result["Recall"]:.5f}, '
                  f'Loss:{result["Loss"]:.5f}')

            for k, v in result.items():
                if v is not None:
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        logger.info(f"{k}: {v:.5f}")
                    else:
                        logger.info(f"{k}: {v}")
        else:
            result = {'Accuracy': 0.0}

        return result

    def save_model(self, path=None, save_optimizer=False):
        if path is None:
            path = os.path.join(self.args.model_dir, self.args.model)
        os.makedirs(path, exist_ok=True)

        do_dp = self.args.do_parallel or self.args.deepspeed
        model = self.model.module if hasattr(self.model, 'module') else self.model

        if self.args.deepspeed:
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

    def load_model(self, path=None):
        if path is None:
            path = os.path.join(self.args.model_dir, self.args.model)

        do_dp = self.args.do_parallel or self.args.deepspeed
        model = self.model.module if hasattr(self.model, 'module') else self.model
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
            logger.info("  %d keys not in checkpoint (kept from init)", len(missing))
        if unexpected:
            logger.warning("  %d keys in checkpoint but not in model — skipped", len(unexpected))
        logger.info("Model loaded from %s (%d/%d keys)", bin_path, n_loaded, len(saved_state))

    def visualize(self):
        self.model.eval()
        inputs = (torch.rand((self.data_config.batch_size, self.data_config.node_size, self.data_config.time_series_size)),
                  torch.rand((self.data_config.batch_size, self.data_config.node_size, self.data_config.node_size)),
                  F.one_hot(torch.randint(0, self.model_config.num_classes, (self.data_config.batch_size,))))
        input_kwargs = self.prepare_inputs_kwargs(inputs)
        self.model.config.dict_output = False
        torch.onnx.export(self.model,
                          tuple([v for k, v in input_kwargs.items()]),
                          'model.onnx')
        self.model.config.dict_output = True
