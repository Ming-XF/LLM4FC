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

# Age mapping from Seizures_Information.xlsx (years, parsed)
_BEIRUT_PATIENT_AGE = {
    10: 5.0,
    11: 4.5,
    12: 8.0,
    13: 9.0,
    14: 7.0,
    15: 8.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AgeBeirutDataset(BaseDataset):
    """Beirut age regression dataset — predict chronological age from EEG.

    Task: given a 1‑minute window of 19‑channel EEG time series,
    predict the subject's age (continuous value in years).

    Includes all 6 patients (10–15), all pediatric (4.5–9 years).
    Age values from Seizures_Information.xlsx.

    Note: statistical power is extremely limited (6 subjects).
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(AgeBeirutDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                               episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        data = np.load(self.data_config.data_dir, allow_pickle=True).item()

        time_series = data["timeseries"]
        labels = data["labels"].astype(np.float32)
        subject_id = data["subject_id"]
        self.hz = data.get("hz", 200)

        self.data_config.node_size = time_series[0].shape[0]
        self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 1  # regression

        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        # ── Binned labels for stratified splits ──
        age_binned = (labels // 5).astype(int)  # wider bins for pediatric range
        groups = np.arange(len(self.all_data['labels']))
        self._create_splits(age_binned, groups)
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(
            self.all_data['time_series'][idx[item]]).float()
        labels = torch.tensor(
            self.all_data['labels'][idx[item]], dtype=torch.float32)

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

def age_beirut_preprocess(path_beirut="../data/Beirut", hz=200):
    """Preprocess the Beirut epilepsy dataset for age regression.

    Reads all EDF files for patients 10–15, extracts the 19 EEG channels,
    resamples to ``hz`` Hz, and segments into non‑overlapping 1‑minute windows.

    Labels: continuous age in years (float32), from Seizures_Information.xlsx.

    Parameters
    ----------
    path_beirut : str
        Path to the Beirut dataset directory.
    hz : int
        Resampling rate in Hz.  Default 200.
    """
    output_path = os.path.join(path_beirut, "beirut_age.npy")

    time_series_all = []
    labels_all = []
    subject_ids_all = []

    for patient in [10, 11, 12, 13, 14, 15]:
        age = _BEIRUT_PATIENT_AGE[patient]
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
            eeg = eeg[:19, :]

            # ── Non-overlapping 1‑minute windows ──
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
                labels_all.append(age)
                subject_ids_all.append(float(patient))

    # ── Stack and normalize ──
    time_series_all = np.array(time_series_all)
    labels_all = np.array(labels_all, dtype=np.float32)
    subject_ids_all = np.array(subject_ids_all)

    time_series_all = data_norm(time_series_all)
    time_series_all = preprocess_ea(time_series_all)

    print(f"Total samples: {len(labels_all)}")
    print(f"  Age range: {labels_all.min():.1f}–{labels_all.max():.1f} "
          f"(mean={labels_all.mean():.1f})")
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
    age_beirut_preprocess(path_beirut="../data/Beirut", hz=200)
