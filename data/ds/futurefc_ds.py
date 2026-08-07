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

# Default parameters for future FC prediction
_N_INPUT_WINDOWS = 6   # use first 6 DFC windows as input
_N_TOTAL_WINDOWS = 10  # total DFC windows per 1-min segment


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class FutureFCDSDataset(BaseDataset):
    """DS future FC prediction dataset — self-supervised time-series forecasting.

    Task: given the first ``k`` dynamic FC windows (computed from a 1‑minute
    EEG segment), predict the remaining ``T−k`` future FC matrices.

    Input: first k DFC matrices  (k, 19, 19)
    Target: remaining T-k DFC matrices  (T-k, 19, 19)

    Uses the same subjects as the disease classification task (AD + CN).
    Labels in the .npz are dummy zeros (the true supervision signal is the
    future FC matrices themselves).

    N.B. This task requires a regression-style trainer with MSE/MAE loss
    on the predicted FC matrices; the existing classification trainer will
    not work.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None,
                 n_input_windows=_N_INPUT_WINDOWS,
                 n_total_windows=_N_TOTAL_WINDOWS):
        self.n_input_windows = n_input_windows
        self.n_total_windows = n_total_windows
        super(FutureFCDSDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        raw = np.load(self.data_config.data_dir, allow_pickle=True)
        data = dict(raw) if hasattr(raw, 'files') else raw.item()
        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = data["hz"]

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.task_type = DataConfig.TASK_MULTI_OUTPUT_REGRESSION
        n_out = self.n_total_windows - self.n_input_windows
        self.data_config.output_dim = n_out * self.data_config.node_size ** 2

        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        # ── 使用预处理阶段预计算的划分（随机，无分层）──
        if 'split_train_index' in data:
            self.train_index = data['split_train_index']
            self.val_index = data['split_val_index']
            self.test_index = data['split_test_index']
        else:
            print("  [WARN] 未找到预计算划分，回退到 _create_splits")
            self._create_splits(labels, self.all_data['subject_id'])
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(
            self.all_data['time_series'][idx[item]]).float()
        window_size = 12 * self.hz
        step_size = (60 * self.hz - window_size) // (self.n_total_windows - 1)
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)
        DFC = self.sparsify_fc(DFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)
        SFC = self.connectivity(time_series)
        SFC = self.sparsify_fc(SFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)

        # ── labels 即 DFC 的未来窗口 ──
        return {
                'DFC': DFC,
                'correlation': SFC,
                'labels': DFC[self.n_input_windows:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════



def futurefc_ds_preprocess(path="../data/DS", hz=200,
                           max_windows_per_subject=None,
                           train_split=0.7, val_split=0.15):
    """Preprocess the DS dataset for future FC prediction.

    Identical signal processing to ``ds_preprocess()`` but saves dummy labels
    (all zeros).  The prediction target (future FC matrices) is computed
    on-the-fly in ``FutureFCDSDataset.__getitem__``.

    Parameters
    ----------
    path : str
        Path to the DS dataset root directory.
    hz : int
        Resampling rate in Hz.  Default 200.
    """
    participants_path = os.path.join(path, "participants.tsv")
    derivatives_dir = os.path.join(path, "derivatives")
    output_path = os.path.join(path, "ds_futurefc.npz")

    participants = pd.read_csv(participants_path, sep='\t')
    target_participants = participants[participants['Group'].isin(['A', 'C'])]

    print(f"Target subjects (AD + CN): {len(target_participants)}")

    ts_list, lbl_list, subj_list = [], [], []

    for _, row in tqdm(target_participants.iterrows(),
                       total=len(target_participants),
                       desc="Processing subjects"):
        subj_id = row['participant_id']

        set_path = os.path.join(
            derivatives_dir, subj_id, 'eeg',
            f'{subj_id}_task-eyesclosed_eeg.set')

        if not os.path.exists(set_path):
            print(f"  [SKIP] {set_path} not found")
            continue

        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

        available = set(raw.info['ch_names'])
        missing = [ch for ch in _DS_CHANNEL_ORDER if ch not in available]
        if missing:
            print(f"  [WARN] {subj_id}: missing channels: {missing}")
            continue

        raw.pick(_DS_CHANNEL_ORDER, verbose=False)
        raw = raw.copy().resample(sfreq=hz, verbose=False)
        data = raw.get_data()

        n_total = data.shape[1]
        window_samples = hz * 60
        n_windows = n_total // window_samples
        if n_windows == 0:
            print(f"  [SKIP] {subj_id}: too short ({n_total / hz:.1f}s)")
            continue

        data = data[:, :n_windows * window_samples]
        data = data.reshape(data.shape[0], n_windows, window_samples)
        data = np.transpose(data, (1, 0, 2))

        labels_arr = np.zeros(data.shape[0], dtype=np.int8)  # dummy
        subj_ids_arr = np.full(data.shape[0], int(subj_id.split('-')[1]))

        ts_list.append(data)
        lbl_list.append(labels_arr)
        subj_list.append(subj_ids_arr)

    time_series = np.concatenate(ts_list, axis=0)
    labels = np.concatenate(lbl_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)

    # ── Per-subject window cap (evenly spaced sampling along time axis) ──
    if max_windows_per_subject is not None:
        unique_subjs = np.unique(subject_ids)
        keep_mask = np.ones(len(subject_ids), dtype=bool)
        n_capped = 0
        for subj in unique_subjs:
            idx = np.where(subject_ids == subj)[0]
            cap = max_windows_per_subject
            if cap is not None and len(idx) > cap:
                sample_idx = np.linspace(0, len(idx) - 1, cap, dtype=int)
                keep_mask[idx] = False
                keep_mask[idx[sample_idx]] = True
                n_capped += 1
        time_series = time_series[keep_mask]
        labels = labels[keep_mask]
        subject_ids = subject_ids[keep_mask]
        print(f"Capped {n_capped} subjects (evenly spaced)")

    # ── Per-subject random split (no stratification, reproducible) ──
    unique_subjs = np.unique(subject_ids)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_subjs)

    n_total = len(unique_subjs)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_subjs = unique_subjs[:n_train]
    val_subjs = unique_subjs[n_train:n_train + n_val]
    test_subjs = unique_subjs[n_train + n_val:]

    split_train_index = np.where(np.isin(subject_ids, train_subjs))[0]
    split_val_index = np.where(np.isin(subject_ids, val_subjs))[0]
    split_test_index = np.where(np.isin(subject_ids, test_subjs))[0]

    print(f"\nSplit: train={n_train} subj ({len(split_train_index)} samples), "
          f"val={n_val} subj ({len(split_val_index)} samples), "
          f"test={n_total - n_train - n_val} subj ({len(split_test_index)} samples)")

    time_series = time_series.astype(np.float32)
    labels = labels.astype(np.int8)

    print(f"\nTotal samples: {len(labels)}")
    print(f"  Shape: {time_series.shape}")

    np.savez(output_path,
             timeseries=time_series,
             labels=labels, subject_id=subject_ids, hz=hz,
             split_train_index=split_train_index,
             split_val_index=split_val_index,
             split_test_index=split_test_index)
    print(f"Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    futurefc_ds_preprocess("../data/DS", hz=200)
