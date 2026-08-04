import os
from random import shuffle

import mne
import numpy as np
import torch
import torch.nn.functional as F
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

# Gender mapping from Seizures_Information.xlsx
# Patient → Gender: F=False(0), M=True(1)
_BEIRUT_PATIENT_GENDER = {
    10: 0,   # F
    11: 0,   # F
    12: 1,   # M
    13: 1,   # M
    14: 1,   # M
    15: 1,   # M
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class GenderBeirutDataset(BaseDataset):
    """Beirut gender classification dataset — Female vs Male binary classification.

    Task: given a 1‑minute window of 19‑channel EEG time series,
    classify the subject's gender as Female (0) or Male (1).

    Includes all 6 patients (10–15).  Gender labels are derived from
    Seizures_Information.xlsx.

    Note: statistical power is limited (6 subjects, 2F/4M, all pediatric).
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(GenderBeirutDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                  episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        data = np.load(self.data_config.data_dir, allow_pickle=True).item()

        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = data.get("hz", 200)

        self.data_config.node_size = time_series[0].shape[0]
        self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 2
        self.data_config.class_weight = [1, 1]

        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        # ── Split via parent _create_splits (K‑Fold or train/val/test) ──
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

        correlation = self.connectivity(time_series)

        window_size = 6 * self.hz
        step_size = (time_series.size(-1) - window_size) // 9
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)

        return {'time_series': time_series,
                'correlation': correlation,
                'DFC': DFC,
                'labels': labels}


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def gender_beirut_preprocess(path_beirut="../data/Beirut", hz=200):
    """Preprocess the Beirut epilepsy dataset for gender (M/F) classification.

    Reads all EDF files for patients 10–15, extracts the 19 EEG channels,
    resamples to ``hz`` Hz, and segments into non‑overlapping 1‑minute windows.

    Labels: Female=0, Male=1 (from Seizures_Information.xlsx).

    Unlike the seizure prediction task, this uses ALL available data
    (non-overlapping windows) regardless of seizure state, since gender
    is a static subject attribute.

    Parameters
    ----------
    path_beirut : str
        Path to the Beirut dataset directory.
    hz : int
        Resampling rate in Hz.  Default 200.
    """
    output_path = os.path.join(path_beirut, "beirut_gender.npy")

    time_series_all = []
    labels_all = []
    subject_ids_all = []

    for patient in [10, 11, 12, 13, 14, 15]:
        gender_label = _BEIRUT_PATIENT_GENDER[patient]
        patient_dir = os.path.join(path_beirut, "Raw_EDF_Files")
        patient_files = sorted([
            f for f in os.listdir(patient_dir)
            if f.startswith(f"p{patient}_") and f.endswith(".edf")
        ])

        for file_name in tqdm(patient_files, desc=f"Patient {patient}"):
            file_path = os.path.join(patient_dir, file_name)
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

            # ── Channel re‑order, average reference, resample ──
            raw.reorder_channels(_CHANNEL_ORDER)
            raw.set_eeg_reference('average', verbose=False)
            raw = raw.copy().resample(sfreq=hz, verbose=False)

            eeg, _ = raw[:, :]
            eeg = eeg[:19, :]          # keep 19 EEG channels only

            # ── Non-overlapping 1‑minute windows ──
            window_samples = hz * 60
            n_total = eeg.shape[1]
            n_windows = n_total // window_samples
            if n_windows == 0:
                continue

            eeg = eeg[:, :n_windows * window_samples]
            eeg = eeg.reshape(eeg.shape[0], n_windows, window_samples)
            eeg = np.transpose(eeg, (1, 0, 2))  # (n_windows, 19, hz*60)

            for window in eeg:
                time_series_all.append(window)
                labels_all.append(gender_label)
                subject_ids_all.append(float(patient))

    # ── Stack and normalize ──
    time_series_all = np.array(time_series_all)
    labels_all = np.array(labels_all)
    subject_ids_all = np.array(subject_ids_all)

    time_series_all = data_norm(time_series_all)
    time_series_all = preprocess_ea(time_series_all)

    n_f = int((labels_all == 0).sum())
    n_m = int((labels_all == 1).sum())
    print(f"Total samples: {len(labels_all)}")
    print(f"  Female (label=0): {n_f}")
    print(f"  Male   (label=1): {n_m}")
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
    gender_beirut_preprocess(path_beirut="../data/Beirut", hz=200)
