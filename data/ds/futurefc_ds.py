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
_N_INPUT_WINDOWS = 8   # use first 8 DFC windows as input
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
        super(FutureFCDSDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                episode_seed=episode_seed)
        self.n_input_windows = n_input_windows
        self.n_total_windows = n_total_windows

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

        self._create_splits(labels, self.all_data['subject_id'])
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(
            self.all_data['time_series'][idx[item]]).float()
        window_size = 6 * self.hz
        step_size = (60 * self.hz - window_size) // (self.n_total_windows - 1)
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)
        SFC = self.connectivity(time_series)

        # ── Split into input (history) and target (future) ──
        dfc_input = DFC[:self.n_input_windows]   # (k, 19, 19)
        dfc_target = DFC[self.n_input_windows:]  # (T-k, 19, 19)

        return {'time_series': time_series,
                'DFC': DFC,
                'DFC_input': dfc_input,
                'DFC_target': dfc_target,
                'correlation': SFC,
                'labels': dfc_target,
                'sample_idx': idx[item]}


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def futurefc_ds_preprocess(path="../data/DS", hz=200):
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

    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)

    time_series = time_series.astype(np.float32)
    labels = labels.astype(np.int8)

    print(f"\nTotal samples: {len(labels)}")
    print(f"  Shape: {time_series.shape}")

    np.savez(output_path, timeseries=time_series,
             labels=labels, subject_id=subject_ids, hz=hz)
    print(f"Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    futurefc_ds_preprocess("../data/DS", hz=200)
