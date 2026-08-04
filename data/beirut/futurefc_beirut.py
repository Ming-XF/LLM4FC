import os
from random import shuffle

import mne
import numpy as np
import torch
from tqdm import tqdm

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# EEG channel order for reordering raw EDF — 19 EEG channels
# ═══════════════════════════════════════════════════════════════════════════════

_CHANNEL_ORDER = [
    'EEG Fp1-Ref', 'EEG F3-Ref', 'EEG C3-Ref', 'EEG P3-Ref', 'EEG O1-Ref',
    'EEG Fp2-Ref', 'EEG F4-Ref', 'EEG C4-Ref', 'EEG P4-Ref', 'EEG O2-Ref',
    'EEG F7-Ref', 'EEG T3-Ref', 'EEG T5-Ref', 'EEG F8-Ref', 'EEG T4-Ref',
    'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref',
    'EEG A2-Ref', 'EEG A1-Ref', 'ECG EKG', 'Manual',
]

_N_INPUT_WINDOWS = 8
_N_TOTAL_WINDOWS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class FutureFCBeirutDataset(BaseDataset):
    """Beirut future FC prediction dataset.

    Task: given the first ``k`` dynamic FC windows from a 1‑minute EEG
    segment, predict the remaining ``T−k`` future FC matrices.

    Input: first k DFC matrices  (k, 19, 19)
    Target: remaining T-k DFC matrices  (T-k, 19, 19)
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None,
                 n_input_windows=_N_INPUT_WINDOWS,
                 n_total_windows=_N_TOTAL_WINDOWS):
        super(FutureFCBeirutDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                    episode_seed=episode_seed)
        self.n_input_windows = n_input_windows
        self.n_total_windows = n_total_windows

    def load_data(self, one_hot=True):
        data = np.load(self.data_config.data_dir, allow_pickle=True).item()

        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = data.get("hz", 200)

        self.data_config.node_size = time_series[0].shape[0]
        self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = time_series[0].shape[0]

        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        groups = np.arange(len(self.all_data['labels']))
        self._create_splits(self.all_data['labels'], groups)
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(
            self.all_data['time_series'][idx[item]]).float()
        dummy_label = torch.tensor(
            self.all_data['labels'][idx[item]], dtype=torch.int64)

        SFC = self.connectivity(time_series)

        window_size = 6 * self.hz
        step_size = (time_series.size(-1) - window_size) // (self.n_total_windows - 1)
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)

        dfc_input = DFC[:self.n_input_windows]
        dfc_target = DFC[self.n_input_windows:]

        return {'time_series': time_series,
                'correlation': SFC,
                'DFC': DFC,
                'DFC_input': dfc_input,
                'DFC_target': dfc_target,
                'labels': dummy_label}


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def futurefc_beirut_preprocess(path_beirut="../data/Beirut", hz=200):
    """Preprocess the Beirut dataset for future FC prediction.

    Reads all EDF files for patients 11–15, extracts 19 EEG channels,
    resamples, and segments into non‑overlapping 1‑minute windows.

    Dummy labels (all zeros) are saved — actual prediction targets are
    computed on-the-fly from the time series.

    Parameters
    ----------
    path_beirut : str
        Path to the Beirut dataset directory.
    hz : int
        Resampling rate in Hz.  Default 200.
    """
    output_path = os.path.join(path_beirut, "beirut_futurefc.npy")

    time_series_all = []
    labels_all = []
    subject_ids_all = []

    for patient in [11, 12, 13, 14, 15]:
        patient_dir = os.path.join(path_beirut, "Raw_EDF_Files")
        patient_files = sorted([
            f for f in os.listdir(patient_dir)
            if f.startswith(f"p{patient}_") and f.endswith(".edf")
        ])

        for file_name in tqdm(patient_files, desc=f"Patient {patient}"):
            file_path = os.path.join(patient_dir, file_name)
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

            raw.reorder_channels(_CHANNEL_ORDER)
            raw.set_eeg_reference('average', verbose=False)
            raw = raw.copy().resample(sfreq=hz, verbose=False)

            eeg, _ = raw[:, :]
            eeg = eeg[:19, :]

            window_samples = hz * 60
            n_total = eeg.shape[1]
            n_windows = n_total // window_samples
            if n_windows == 0:
                continue

            eeg = eeg[:, :n_windows * window_samples]
            eeg = eeg.reshape(eeg.shape[0], n_windows, window_samples)
            eeg = np.transpose(eeg, (1, 0, 2))

            for window in eeg:
                time_series_all.append(window)
                labels_all.append(0)
                subject_ids_all.append(float(patient))

    time_series_all = np.array(time_series_all)
    labels_all = np.array(labels_all, dtype=np.int8)
    subject_ids_all = np.array(subject_ids_all)

    time_series_all = data_norm(time_series_all)
    time_series_all = preprocess_ea(time_series_all)

    print(f"Total samples: {len(labels_all)}")
    print(f"  Shape: {time_series_all.shape}")

    np.save(output_path, {
        "timeseries": time_series_all,
        "labels": labels_all,
        "subject_id": subject_ids_all,
        "hz": hz,
    })
    print(f"Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    futurefc_beirut_preprocess(path_beirut="../data/Beirut", hz=200)
