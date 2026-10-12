import os
from datetime import date, datetime, time, timedelta
from random import shuffle

import mne
import numpy as np
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
# Seizure annotation times from Preprocessing_Scripts/Seizure_times.py
# Format: {patient: {record_num: [(hour, min, sec, duration_sec), ...], ...}, ...}
# (hour, min, sec) are absolute times of day.
# Patients 11–15 only (patient 10 excluded).
# ═══════════════════════════════════════════════════════════════════════════════

_SEIZURE_TIMES = {
    15: {1: [(17, 18, 8, 50)],
         2: [(22, 49, 24, 47)],
         3: [(2, 57, 4, 13)],
         4: [(5, 3, 26, 56),
             (6, 23, 29, 20)]},
    14: {1: [(14, 32, 2, 28),
             (15, 34, 32, 134)],
         2: [(16, 20, 58, 32),
             (17, 50, 56, 10)],
         3: [(20, 20, 46, 31),
             (21, 2, 4, 26),
             (21, 27, 49, 40),
             (21, 50, 24, 40)]},
    13: {1: [(2, 31, 9, 52)],
         2: [(3, 33, 4, 25),
             (4, 38, 59, 16)],
         3: [(6, 45, 51, 18)],
         4: [(10, 51, 41, 30),
             (12, 18, 22, 24)]},
    12: {1: [(1, 56, 14, 76),
             (2, 20, 12, 104)],
         2: [(5, 50, 55, 118)],
         3: [(6, 40, 30, 82),
             (7, 21, 36, 164),
             (8, 37, 48, 113)]},
    11: {1: [(15, 8, 55, 64)],
         2: [(18, 11, 38, 51)],
         3: [(19, 18, 38, 91),
             (20, 6, 38, 83),
             (20, 53, 22, 76),
             (21, 27, 24, 73)],
         4: [(23, 55, 35, 1358)]},
}

_FILES_MAP = {
    15: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
    14: ["Record1.edf", "Record2.edf", "Record3.edf"],
    13: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
    12: ["Record1.edf", "Record2.edf", "Record3.edf"],
    11: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
}

# EEG channel order for reordering raw EDF — ends up with 19 EEG + 2 ref + EKG + Manual
_CHANNEL_ORDER = [
    'EEG Fp1-Ref', 'EEG F3-Ref', 'EEG C3-Ref', 'EEG P3-Ref', 'EEG O1-Ref',
    'EEG Fp2-Ref', 'EEG F4-Ref', 'EEG C4-Ref', 'EEG P4-Ref', 'EEG O2-Ref',
    'EEG F7-Ref', 'EEG T3-Ref', 'EEG T5-Ref', 'EEG F8-Ref', 'EEG T4-Ref',
    'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref',
    'EEG A2-Ref', 'EEG A1-Ref', 'ECG EKG', 'Manual',
]

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class DiseaseBeirutDataset(BaseDataset):
    """Beirut seizure-classification dataset.

    Task: given a 1‑minute sliding window of 19‑channel EEG time series,
    classify the window as positive (pre‑ictal) or negative (inter‑ictal).

    - Positive (label 1): seizure onset inside [window_end, window_end + 10 min]
    - Negative (label 0): ≥ 10 min away from any seizure
    - Ictal windows and buffer-zone windows are discarded.

    Patients 11–15, patient 10 excluded.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(DiseaseBeirutDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                   episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        data = np.load("../data/Beirut/beirut_disease.npy", allow_pickle=True).item()

        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = data.get("hz", 200)

        self.data_config.node_size = time_series[0].shape[0]
        self.data_config.node_feature_size = time_series[0].shape[0]
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
            groups = np.arange(len(self.all_data['labels']))
            self._create_splits(self.all_data['labels'], groups)

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
