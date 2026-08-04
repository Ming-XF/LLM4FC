import os
from random import shuffle

import mne
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Standard 19-channel 10-20 EEG order (CAUEEG-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

_TUEP_CHANNEL_ORDER = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7', 'T3', 'T5',
    'F8', 'T4', 'T6',
    'Fz', 'Cz', 'Pz',
]

_CHANNEL_NORM_MAP = {
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
}


def _normalize_channel(name):
    """Normalize an EDF channel name to the 10-20 short form."""
    for prefix in ['EEG ']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ['-REF', '-LE']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return _CHANNEL_NORM_MAP.get(name, name)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class GenderTUEPDataset(BaseDataset):
    """TUEP gender classification — Female vs Male binary classification.

    Task: given a 1‑minute window of 19‑channel EEG time series,
    classify the subject's gender as Female (0) or Male (1).

    Uses all subjects from the TUH EEG Epilepsy Corpus regardless of
    epilepsy/no-epilepsy label.  Gender is extracted from the EDF file header.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(GenderTUEPDataset, self).__init__(data_config, k, train, one_hot=one_hot,
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
        self.data_config.output_dim = 2
        self.data_config.task_type = DataConfig.TASK_CLASSIFICATION

        self.data_config.class_weight = [1, 1]
        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

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

        SFC = self.connectivity(time_series)
        window_size = 6 * self.hz
        step_size = (60 * self.hz - window_size) // 9
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)

        return {'time_series': time_series,
                'DFC': DFC,
                'correlation': SFC,
                'labels': labels,
                'sample_idx': idx[item]}


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — worker (module-level for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════════

def _process_tuep_file_gender(args):
    """Process a single TUEP EDF file into 1‑minute windows with gender label.

    Gender is extracted from the EDF file header (Patient ID field).
    MNE maps sex=1 → Male, sex=2 → Female; unknown/0 files are skipped.

    Parameters
    ----------
    args : tuple
        (edf_path, hz, subject_id)

    Returns
    -------
    data : ndarray (n_windows, 19, hz*60) or None
    labels : ndarray (n_windows,) or None
    subj_ids : ndarray (n_windows,) or None
    """
    edf_path, hz, subject_id = args

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None, None, None

    # ── Extract gender from EDF header ──
    sex = raw.info.get('subject_info')
    if sex is not None:
        sex_val = sex.get('sex', 0) if hasattr(sex, 'get') else getattr(sex, 'sex', 0)
    else:
        sex_val = 0

    if sex_val not in (1, 2):
        return None, None, None

    gender_label = 0 if sex_val == 2 else 1  # F=0, M=1

    # ── Build mapping from normalised channel name → EDF index ──
    edf_ch_names = raw.info['ch_names']
    ch_index = {}
    for i, ch in enumerate(edf_ch_names):
        norm = _normalize_channel(ch)
        ch_index[norm] = i

    # ── Check all 19 standard channels exist ──
    missing = [ch for ch in _TUEP_CHANNEL_ORDER if ch not in ch_index]
    if missing:
        return None, None, None

    # ── Pick and reorder the 19 EEG channels ──
    pick_idx = [ch_index[ch] for ch in _TUEP_CHANNEL_ORDER]
    raw.pick([edf_ch_names[i] for i in pick_idx], verbose=False)

    # ── Average reference, resample ──
    raw.set_eeg_reference('average', verbose=False)
    raw = raw.copy().resample(sfreq=hz, verbose=False)
    data = raw.get_data()  # (19, n_samples)

    # ── Truncate to whole minutes, reshape to 1‑minute windows ──
    n_total = data.shape[1]
    window_samples = hz * 60
    n_windows = n_total // window_samples
    if n_windows == 0:
        return None, None, None

    data = data[:, :n_windows * window_samples]
    data = data.reshape(data.shape[0], n_windows, window_samples)
    data = np.transpose(data, (1, 0, 2))  # (n_windows, 19, hz*60)

    labels_arr = np.full(data.shape[0], gender_label, dtype=np.int8)
    subj_arr = np.full(data.shape[0], subject_id, dtype=np.int32)

    return data, labels_arr, subj_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def gender_tuep_preprocess(path="../data/TUEP", hz=200):
    """Preprocess the TUH EEG Epilepsy Corpus for gender (M/F) classification.

    Reads EDF files from ``00_epilepsy/`` and ``01_no_epilepsy/`` using the
    ``01_tcp_ar`` montage.  Extracts the standard 19‑channel 10-20 EEG,
    resamples to ``hz`` Hz, and segments into non‑overlapping 1‑minute windows.

    Gender labels are extracted from the EDF file header (Patient ID field):
    Female=0, Male=1.  Files with unknown/missing sex are skipped.

    Parameters
    ----------
    path : str
        Path to the TUEP dataset root.
    hz : int
        Resampling rate in Hz.  Default 200.
    """
    output_path = os.path.join(path, "tuep_gender.npz")

    # ── Collect EDF files ──
    file_list = []
    subject_map = {}

    for cls_dir_name in ['00_epilepsy', '01_no_epilepsy']:
        cls_dir = os.path.join(path, cls_dir_name)
        if not os.path.isdir(cls_dir):
            print(f"  [SKIP] directory not found: {cls_dir}")
            continue

        for subj_name in sorted(os.listdir(cls_dir)):
            subj_dir = os.path.join(cls_dir, subj_name)
            if not os.path.isdir(subj_dir):
                continue

            for sess_name in sorted(os.listdir(subj_dir)):
                sess_dir = os.path.join(subj_dir, sess_name)
                if not os.path.isdir(sess_dir):
                    continue

                montage_dir = os.path.join(sess_dir, '01_tcp_ar')
                if not os.path.isdir(montage_dir):
                    continue

                for fname in sorted(os.listdir(montage_dir)):
                    if not fname.endswith('.edf'):
                        continue
                    edf_path = os.path.join(montage_dir, fname)
                    file_list.append((edf_path, subj_name))

    print(f"Found {len(file_list)} EDF files")

    # ── Assign sequential integer subject IDs ──
    for _, subj_str in file_list:
        if subj_str not in subject_map:
            subject_map[subj_str] = len(subject_map) + 1  # 1-indexed

    print(f"Unique subjects: {len(subject_map)}")

    # ── Build multiprocessing task args ──
    task_args = [
        (edf_path, hz, subject_map[subj_str])
        for edf_path, subj_str in file_list
    ]

    n_workers = min(cpu_count(), len(task_args), 8)
    print(f"Processing {len(task_args)} files with {n_workers} workers...")

    ts_list, lbl_list, subj_list = [], [], []
    skipped = 0
    with Pool(processes=n_workers) as pool:
        for data, labels_arr, subj_arr in tqdm(
                pool.imap_unordered(_process_tuep_file_gender, task_args),
                total=len(task_args),
                desc="Processing TUEP gender"):
            if data is None:
                skipped += 1
                continue
            ts_list.append(data)
            lbl_list.append(labels_arr)
            subj_list.append(subj_arr)

    print(f"Skipped {skipped} files (missing channels, too short, or unknown sex)")

    if not ts_list:
        raise RuntimeError("No valid EDF files processed — check dataset path.")

    # ── Concatenate ──
    time_series = np.concatenate(ts_list, axis=0)
    labels = np.concatenate(lbl_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)

    # ── Normalize ──
    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)

    time_series = time_series.astype(np.float32)
    labels = labels.astype(np.int8)

    # ── Report ──
    n_f = int((labels == 0).sum())
    n_m = int((labels == 1).sum())
    print(f"\nTotal samples: {len(labels)}")
    print(f"  Female (label=0): {n_f}")
    print(f"  Male   (label=1): {n_m}")
    print(f"  Shape: {time_series.shape}")

    np.savez(output_path, timeseries=time_series,
             labels=labels, subject_id=subject_ids, hz=hz)
    print(f"Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    gender_tuep_preprocess("../data/TUEP", hz=200)
