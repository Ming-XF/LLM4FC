import os
from random import shuffle

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CAUEEG-compatible channel order (19 EEG, 10-20 system)
# DS raw .set may have a different order; we reorder to this for cross-dataset
# compatibility with CAUEEG.
# ═══════════════════════════════════════════════════════════════════════════════

_DS_CHANNEL_ORDER = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',   # left hemisphere
    'Fp2', 'F4', 'C4', 'P4', 'O2',   # right hemisphere
    'F7', 'T3', 'T5',                  # left temporal
    'F8', 'T4', 'T6',                  # right temporal
    'Fz', 'Cz', 'Pz',                  # midline
]

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class DiseaseDSDataset(BaseDataset):
    """DS dementia dataset — AD vs CN binary classification.

    Task: given a 1‑minute sliding window of 19‑channel resting‑state EEG,
    classify the subject as Alzheimerʼs disease (AD) or healthy control (CN).

    - AD  (label 1): Group A in participants.tsv
    - CN  (label 0): Group C in participants.tsv

    Uses preprocessed derivatives/ .set files, resampled to 200 Hz,
    with channels reordered to match the CAUEEG standard order.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(DiseaseDSDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                               episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        raw = np.load("../data/DS/ds_disease.npz", allow_pickle=True)
        data = dict(raw) if hasattr(raw, 'files') else raw.item()
        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = data["hz"]

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.output_dim = 2
        self.data_config.task_type = DataConfig.TASK_CLASSIFICATION

        self.data_config.class_weight = [1, 1]
        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        if 'split_train_index' in data:
            self.train_index = data['split_train_index']
            self.val_index = data['split_val_index']
            self.test_index = data['split_test_index']
        else:
            print("  [WARN] 未找到预计算划分，回退到 _create_splits")
            self._create_splits(labels, self.all_data['subject_id'])
        self.all_data['labels'] = F.one_hot(
            torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(
            self.all_data['time_series'][idx[item]]).float()
        labels = torch.from_numpy(
            self.all_data['labels'][idx[item]]).to(torch.int64)

        # ── EEGNet fallback：仅时域信号（在线 FC/PSD 死代码已删）──
        return {
                'time_series': time_series,
                'labels': labels,
        }
