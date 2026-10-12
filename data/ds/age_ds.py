import os
from random import shuffle

import mne
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CAUEEG-compatible channel order (19 EEG, 10-20 system)
# ═══════════════════════════════════════════════════════════════════════════════

_DS_CHANNEL_ORDER = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7', 'T3', 'T5',
    'F8', 'T4', 'T6',
    'Fz', 'Cz', 'Pz',
]

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AgeDSDataset(BaseDataset):
    """DS age regression dataset — predict chronological age from EEG.

    Task: given a 1‑minute window of 19‑channel resting‑state EEG,
    predict the subject's age (continuous value in years).

    Uses the same subjects as the disease classification task (AD + CN from
    participants.tsv).  Age labels are float32.

    Note: ``output_dim`` is set to 1 (regression).
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(AgeDSDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                           episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        raw_path = "../data/DS/ds_age.npz"
        raw = np.load(raw_path, allow_pickle=True)
        data = dict(raw) if hasattr(raw, 'files') else raw.item()
        time_series = data["timeseries"]
        labels, labels_raw, self.label_min, self.label_max = prepare_age_labels(
            data, source=raw_path)
        subject_id = data["subject_id"]
        self.hz = data["hz"]

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.output_dim = 1  # regression
        self.data_config.task_type = DataConfig.TASK_REGRESSION

        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['labels_raw'] = labels_raw
        self.all_data['subject_id'] = subject_id

        # ── 使用预处理阶段预计算的划分（随机，无分层）──
        if 'split_train_index' in data:
            self.train_index = data['split_train_index']
            self.val_index = data['split_val_index']
            self.test_index = data['split_test_index']
        else:
            # 向后兼容：旧 .npz 无预计算划分时回退到分层划分
            print("  [WARN] 未找到预计算划分，回退到 _create_splits 分层划分")
            self._create_splits(labels, self.all_data['subject_id'])
        # No one_hot — labels are continuous floats
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(
            self.all_data['time_series'][idx[item]]).float()
        labels = torch.tensor(
            self.all_data['labels'][idx[item]], dtype=torch.float32)

        # ── EEGNet fallback：仅时域信号（在线 FC/PSD 死代码已删）──
        return {
                'time_series': time_series,
                'labels': labels,
        }
