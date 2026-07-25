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
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        self.task_id = task_id
        self.args = args
        self.local_rank = local_rank
        self.subject_id = subject_id
        self.data_config = DataConfig(args)
        self.data_loaders = self.load_datasets()

        model, self.model_config = init_model_config(args, self.data_config)
        if args.deepspeed:
            self.device = f'cuda:{self.local_rank}' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            # Keep model as-is; LDDE2thTrainer handles device placement
            # before DeepSpeed.initialize().
            self.model = model
        elif args.do_parallel:
            # self.model = torch.nn.DataParallel(self.model)
            self.device = f'cuda:{self.local_rank}' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   find_unused_parameters=True)
        else:
            self.device = f'cuda' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
        # self.model = torch.compile(model, dynamic=True)

        self.optimizer = None
        self.scheduler = None

        self.best_result = None
        self.test_result = None

    @abstractmethod
    def prepare_inputs_kwargs(self, inputs):
        return {}

    def load_datasets(self):
        # datasets = eval(
        #     f"load_{self.args.dataset}_data")(self.data_config)
        datasets = eval(
            f"{self.args.dataset}Dataset")(self.data_config, k=self.task_id, subject_id=self.subject_id)

        if self.args.deepspeed:
            data_loaders = init_deepspeed_dataloader(self.data_config, datasets)
        elif self.args.do_parallel:
            data_loaders = init_distributed_dataloader(self.data_config, datasets)
        else:
            data_loaders = init_StratifiedKFold_dataloader(self.data_config, datasets)
        return data_loaders

    def init_components(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        self.optimizer = init_optimizer(self.model, self.args)
        self.scheduler = init_schedule(self.optimizer, self.args, total)

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []
        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss

            if self.data_config.dataset == "ZuCo":
                loss.backward()
                if step % self.data_config.batch_size == self.data_config.batch_size - 1:

                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(), 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

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

        self.init_components()
        if self.args.visualize:
            self.visualize()

        # ── Early stopping — monitors val Loss (mode='min') ──
        from utils.early_stopping import EarlyStopping
        early_stopper = EarlyStopping(
            patience=self.args.early_stop_patience,
            min_delta=self.args.early_stop_min_delta,
            mode='min',
        )

        best_test_result = None

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch", ncols=0):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.5 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.5

            # ── Validation ──
            val_result = self.evaluate(dataloader_key='val')
            val_loss = val_result.get('Loss', float('inf'))

            # ── Test (evaluate every epoch) ──
            test_result = self.evaluate(dataloader_key='test')

            # ── Early stop based on val Loss ──
            improved = early_stopper.step(val_loss)
            if improved:
                self.best_result = val_result
                best_test_result = test_result
                self.save_model()
                logger.info("Best model saved (val_loss=%.4f, test_acc=%.4f)",
                            early_stopper.best_score,
                            test_result.get('Accuracy', 0.0))

            msg = (f" Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}, "
                   f"Test loss: {test_result['Loss']:.5f}, "
                   f"Best val loss: {early_stopper.best_score:.5f}, "
                   f"No improve: {early_stopper.counter}/{early_stopper.patience}, "
                   f"Time: {(end_time - start_time):.1f}s")
            print(msg)
            logger.info(msg)

            if early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # ── Load best model & report test result from best epoch ──
        self.load_model()
        self.test_result = best_test_result
        logger.info("=== Best epoch test result ===")
        if self.test_result is not None:
            for k, v in self.test_result.items():
                if v is not None:
                    logger.info(f"  {k}: {v:.5f}")
        else:
            logger.info("  (no test result recorded)")

    def evaluate(self, dataloader_key='test', inference_mode=None):
        if self.data_config.num_class == 2:
            result = self.binary_evaluate(dataloader_key, inference_mode=inference_mode)
        else:
            result = self.multiple_evaluate(dataloader_key, inference_mode=inference_mode)
        return result

    def binary_evaluate(self, dataloader_key='test', **kwargs):
        logger.info(f"***** Running evaluation on {dataloader_key} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders[dataloader_key]
        losses = 0
        loss_list = []
        labels = []
        result = {}
        preds = []
        acc = []
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())
                # print(f"Evaluate loss: {loss.item():.5f}")

                # top1 = accuracy(outputs.logits, input_kwargs['labels'][:, 1])[0]
                # acc.append([top1*input_kwargs['labels'].shape[0], input_kwargs['labels'].shape[0]])
                preds += F.softmax(outputs.logits, dim=1)[:, 1].tolist()
                labels += input_kwargs['labels'][:, 1].tolist()
            
            # acc = np.array(acc).sum(axis=0)
            # result['Accuracy'] = acc[0] / acc[1]
            result['AUC'] = roc_auc_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            preds[preds > 0.5] = 1
            preds[preds <= 0.5] = 0
            result['Accuracy'] = (preds == labels).sum() / len(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average="binary")
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]

            report = classification_report(
                labels, preds, output_dict=True, zero_division=0)

            recall = [0, 0]
            for k, v in report.items():
                if '.' in k:
                    recall[int(float(k))] = v['recall']

            result['Specificity'] = recall[0]
            result['Sensitivity'] = recall[1]
            result['Loss'] = losses / len(loss_list)
        print(f'\n{dataloader_key}{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, Precision:{result["Precision"]:.5f}, '
              f'AUC:{result["AUC"]:.5f}, Recall:{result["Recall"]:.5f}, F_score:{result["F_score"]:.5f}, '
              f'Loss:{result["Loss"]:.5f}')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        # wandb.log(result)
        return result

    def multiple_evaluate(self, dataloader_key='test', **kwargs):
        logger.info(f"***** Running evaluation on {dataloader_key}{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders[dataloader_key]
        losses = 0
        loss_list = []
        labels = []
        result = {}
        preds = None
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())
                # print(f"Evaluate loss: {loss.item():.5f}")
                if preds is None:
                    preds = F.softmax(outputs.logits, dim=1).cpu().numpy()
                else:
                    preds = np.append(preds, F.softmax(outputs.logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            try:
                result['AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            except:
                result['AUC'] = 0.
            preds = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average='macro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]

            # report = classification_report(
            #     labels, preds, output_dict=True, zero_division=0)
            #
            # recall = [0, 0]
            # for k, v in report.items():
            #     if '.' in k:
            #         recall[int(float(k))] = v['recall']
            #
            # result['Specificity'] = recall[0]
            # result['Sensitivity'] = recall[1]
            result['Loss'] = losses / len(loss_list)
        # if self.args.within_subject:
        #     print(f'Test{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}', end=',')
        # else:
        #     print(f'Test{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}', end=',')
        
        print(f'\n{dataloader_key}{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, Precision:{result["Precision"]:.5f}, '
              f'AUC:{result["AUC"]:.5f}, Recall:{result["Recall"]:.5f}, F_score:{result["F_score"]:.5f}, '
              f'Loss:{result["Loss"]:.5f}')

        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        # wandb.log(result)
        return result

    def save_model(self):
        # Save model checkpoint (Overwrite)
        path = os.path.join(self.args.model_dir, self.args.model)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = self.model.module if self.args.do_parallel else self.model
        torch.save(model_to_save, os.path.join(path, f'{self.args.model}-{self.task_id}.bin'))

        # Save training arguments together with the trained model
        args_dict = {k: v for k, v in self.args.__dict__.items()}
        with open(os.path.join(path, "config.json"), 'w') as f:
            f.write(json.dumps(args_dict))
        logger.info("Saving model checkpoint to %s", path)

    def load_model(self, path=None):
        """Load the best checkpoint saved during training.

        When ``path`` is None, loads from the default output directory
        using the current ``task_id``.
        """
        if path is None:
            path = os.path.join(self.args.model_dir, self.args.model)
        bin_path = os.path.join(path, f'{self.args.model}-{self.task_id}.bin')

        if not os.path.exists(bin_path):
            logger.info("Model checkpoint doesn't exist: %s — skip loading", bin_path)
            return

        saved = torch.load(bin_path, map_location=self.device, weights_only=False)

        if hasattr(saved, 'state_dict'):
            # ── 旧格式：保存了整个模型对象 ──
            self.model = saved.to(self.device)
            logger.info("Model loaded from %s (full model)", bin_path)
        elif isinstance(saved, dict):
            # ── 新格式：保存了 state_dict 子集 ──
            if self.args.do_parallel or self.args.deepspeed:
                model = self.model.module if hasattr(self.model, 'module') else self.model
            else:
                model = self.model

            current_state = model.state_dict()
            missing, unexpected = [], []
            for k, v in saved.items():
                if k in current_state:
                    if current_state[k].shape == v.shape:
                        current_state[k].copy_(v)
                    else:
                        logger.warning("  Shape mismatch for %s: saved %s vs current %s — skipped",
                                       k, tuple(v.shape), tuple(current_state[k].shape))
                else:
                    unexpected.append(k)
            for k in current_state:
                if k not in saved:
                    missing.append(k)

            n_loaded = len(saved) - len(unexpected)
            if missing:
                logger.info("  %d keys not in checkpoint (kept from init)", len(missing))
            logger.info("Model loaded from %s (%d/%d keys)", bin_path, n_loaded, len(saved))
        else:
            logger.warning("Unknown checkpoint format in %s — skip loading", bin_path)

    def visualize(self):
        self.model.eval()
        inputs = (torch.rand((self.data_config.batch_size, self.data_config.node_size, self.data_config.time_series_size)),
                  torch.rand((self.data_config.batch_size, self.data_config.node_size, self.data_config.node_size)),
                  F.one_hot(torch.randint(0, self.model_config.num_classes, (self.data_config.batch_size,))))
        input_kwargs = self.prepare_inputs_kwargs(inputs)
        # save_path = os.path.join(self.args.model_dir, self.args.model, 'model.onnx')
        self.model.config.dict_output = False
        torch.onnx.export(self.model,
                          tuple([v for k, v in input_kwargs.items()]),
                          'model.onnx')
        # wandb.save('model.onnx')
        self.model.config.dict_output = True
