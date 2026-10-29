"""EEG-Deformer training harness for the LLM4FC disease matrix.

Protocol (identical to the OFA reproduction):
  - 10 s segments (first 10 s of each 60 s window, per-sample per-channel
    z-score baked into the npz), project subject-grouped splits
  - EPOCHS=200 / LR=1e-4 / PATIENCE=20 / val-AUC model selection / seed 42

Run: python main_LLM4FC.py --dataset DiseaseCAUEEG --data-dir llm4fc_data_10s
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '.'))
from models.model_handler import DLModel


def load_npz(data_dir, split):
    d = np.load(osp.join(data_dir, f'{split}.npz'), allow_pickle=True)
    return (torch.from_numpy(d['X'].astype(np.float32)),
            torch.from_numpy(d['y'].astype(np.int64)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=str, default='DiseaseCAUEEG')
    ap.add_argument('--data-dir', type=str, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max-epoch', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save-path', default='./save_llm4fc')
    ap.add_argument('--model', type=str, default='Deformer')
    ap.add_argument('--num-chan', type=int, default=19)
    ap.add_argument('--num-time', type=int, default=2000)
    ap.add_argument('--kernel-length', type=int, default=21)  # Odd[0.1*200Hz]
    ap.add_argument('--num-class', type=int, default=2)
    ap.add_argument('--T', type=int, default=64)      # num_kernel
    ap.add_argument('--AT', type=int, default=16)     # heads/mlp_dim/dim_head
    ap.add_argument('--num-layers', type=int, default=6)
    ap.add_argument('--dropout', type=float, default=0.5)
    args = ap.parse_args()

    # ── torch 2.6+ 的 ckpt 加载默认 weights_only=True，
    #    hparams 中的 argparse.Namespace 需显式白名单 ──
    torch.serialization.add_safe_globals([argparse.Namespace])

    # ── 确定性（与主框架/其他基线一致）──
    pl.seed_everything(args.seed, workers=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    X_tr, y_tr = load_npz(args.data_dir, 'train')
    X_va, y_va = load_npz(args.data_dir, 'val')
    X_te, y_te = load_npz(args.data_dir, 'test')
    print(f'train {tuple(X_tr.shape)} val {tuple(X_va.shape)} '
          f'test {tuple(X_te.shape)}')

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, persistent_workers=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=4)
    test_loader = DataLoader(TensorDataset(X_te, y_te),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=4)

    model = DLModel(config=args)

    out_dir = osp.join(args.save_path, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        monitor='val_auc', mode='max', dirpath=out_dir,
        filename='best-{epoch:02d}-{val_auc:.4f}', save_last=True)
    early_stop_cb = EarlyStopping(
        monitor='val_auc', mode='max', patience=args.patience,
        verbose=True)
    logger = CSVLogger(save_dir=osp.join(args.save_path, 'logs'),
                       name=args.dataset)

    trainer = pl.Trainer(
        accelerator='gpu', devices=[args.gpu], max_epochs=args.max_epoch,
        callbacks=[checkpoint_cb, early_stop_cb], logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    print('=== BEST checkpoint:', checkpoint_cb.best_model_path)
    results = trainer.test(model, dataloaders=test_loader,
                           ckpt_path='best')
    print('=== TEST results:', results[0] if results else None)


if __name__ == '__main__':
    main()
