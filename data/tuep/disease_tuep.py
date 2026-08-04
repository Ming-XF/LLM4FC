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
    'Fp1', 'F3', 'C3', 'P3', 'O1',          # left hemisphere
    'Fp2', 'F4', 'C4', 'P4', 'O2',          # right hemisphere
    'F7', 'T3', 'T5',                         # left temporal
    'F8', 'T4', 'T6',                         # right temporal
    'Fz', 'Cz', 'Pz',                         # midline
]

# Mapping from EDF channel-name variants to standard 10-20 short names.
_CHANNEL_NORM_MAP = {
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
}


def _normalize_channel(name):
    """Normalize an EDF channel name to the 10-20 short form.

    ``EEG FP1-REF`` → ``Fp1``, ``EEG C3-LE`` → ``C3``.
    """
    for prefix in ['EEG ']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ['-REF', '-LE']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return _CHANNEL_NORM_MAP.get(name, name)


def _resolve_param(val, pos):
    """Resolve a per-class parameter that may be an int (both classes) or
    tuple ``(pos_val, neg_val)``.

    Parameters
    ----------
    val : int, tuple, or None
    pos : bool
        ``True`` for positive class, ``False`` for negative.

    Returns
    -------
    int or None
    """
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return val
    return val[0] if pos else val[1]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class DiseaseTUEPDataset(BaseDataset):
    """TUH EEG Epilepsy Corpus — epilepsy vs no-epilepsy binary classification.

    Task: given a 1‑minute window of 19‑channel EEG time series,
    classify the subject as having epilepsy (1) or not (0).

    The dataset is preprocessed by ``disease_tuep_preprocess()`` and stored as a
    single ``.npz`` file.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(DiseaseTUEPDataset, self).__init__(data_config, k, train, one_hot=one_hot,
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

def _process_tuep_file(args):
    """Process a single TUEP EDF file into 1-minute windows.

    Parameters
    ----------
    args : tuple
        (edf_path, label, hz, subject_id)

    Returns
    -------
    data : ndarray (n_windows, 19, hz*60) or None
    labels : ndarray (n_windows,) or None
    subj_ids : ndarray (n_windows,) or None
    """
    edf_path, label, hz, subject_id = args

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None, None, None

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

    labels_arr = np.full(data.shape[0], label, dtype=np.int8)
    subj_arr = np.full(data.shape[0], subject_id, dtype=np.int32)

    return data, labels_arr, subj_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def disease_tuep_preprocess(path="../data/TUEP", hz=200, max_windows_per_subject=60,
                            max_subjects=None):
    """Preprocess the TUH EEG Epilepsy Corpus for epilepsy classification.

    Reads EDF files from ``00_epilepsy/`` (label=1) and ``01_no_epilepsy/``
    (label=0).  For each session the ``01_tcp_ar`` montage is preferred;
    sessions without it are skipped.

    Each file is resampled to ``hz`` Hz, reduced to the standard 19-channel
    10-20 EEG, and segmented into non‑overlapping 1‑minute windows.

    Parameters
    ----------
    path : str
        Path to the TUEP dataset root (contains ``00_epilepsy/`` and
        ``01_no_epilepsy/`` subdirectories).
    hz : int
        Resampling rate in Hz.  Default 200.
    max_windows_per_subject : int, tuple, or None
        Max windows retained per subject.  ``int`` → same for both classes.
        ``(pos, neg)`` → label=1 / label=0 separately.
        ``None`` to disable.  Default 60.
    max_subjects : int, tuple, or None
        Max subjects retained (stratified by class, deterministically).
        ``int`` → ``max_subjects // 2`` per class.
        ``(pos, neg)`` → explicit limits per class.  Default ``None``.
    """
    output_path = os.path.join(path, "tuep_disease.npz")

    # ── Collect EDF files ──
    # For each (class_dir, label) pair, walk subject/session/montage
    file_list = []       # list of (edf_path, label, subject_str)
    subject_map = {}     # subject_str → sequential int ID

    for cls_dir_name, label in [('00_epilepsy', 1), ('01_no_epilepsy', 0)]:
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

                # ── Prefer 01_tcp_ar montage ──
                montage_dir = os.path.join(sess_dir, '01_tcp_ar')
                if not os.path.isdir(montage_dir):
                    continue  # skip sessions without AR montage

                for fname in sorted(os.listdir(montage_dir)):
                    if not fname.endswith('.edf'):
                        continue
                    edf_path = os.path.join(montage_dir, fname)
                    file_list.append((edf_path, label, subj_name))

    print(f"Found {len(file_list)} EDF files")
    print(f"  Epilepsy:     {sum(1 for _, l, _ in file_list if l == 1)}")
    print(f"  No epilepsy:  {sum(1 for _, l, _ in file_list if l == 0)}")

    # ── Assign sequential integer subject IDs ──
    for _, _, subj_str in file_list:
        if subj_str not in subject_map:
            subject_map[subj_str] = len(subject_map) + 1  # 1-indexed

    print(f"Unique subjects: {len(subject_map)}")

    # ── Build multiprocessing task args ──
    task_args = [
        (edf_path, label, hz, subject_map[subj_str])
        for edf_path, label, subj_str in file_list
    ]

    n_workers = min(cpu_count(), len(task_args), 8)
    print(f"Processing {len(task_args)} files with {n_workers} workers...")

    ts_list, lbl_list, subj_list = [], [], []
    skipped = 0
    with Pool(processes=n_workers) as pool:
        for data, labels_arr, subj_arr in tqdm(
                pool.imap_unordered(_process_tuep_file, task_args),
                total=len(task_args),
                desc="Processing TUEP"):
            if data is None:
                skipped += 1
                continue
            ts_list.append(data)
            lbl_list.append(labels_arr)
            subj_list.append(subj_arr)

    if skipped:
        print(f"Skipped {skipped} files (missing channels or too short)")

    if not ts_list:
        raise RuntimeError("No valid EDF files processed — check dataset path.")

    # ── Concatenate ──
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
            # Determine this subject's label (all windows share the same label)
            subj_label = labels[idx[0]]
            cap = _resolve_param(max_windows_per_subject, pos=(subj_label == 1))
            if cap is not None and len(idx) > cap:
                sample_idx = np.linspace(0, len(idx) - 1, cap, dtype=int)
                keep_mask[idx] = False
                keep_mask[idx[sample_idx]] = True
                n_capped += 1
        time_series = time_series[keep_mask]
        labels = labels[keep_mask]
        subject_ids = subject_ids[keep_mask]
        print(f"Capped {n_capped} subjects (evenly spaced)")

    # ── Per-class stratified subject sampling (deterministic: keep first N) ──
    if max_subjects is not None:
        unique_subjs = np.unique(subject_ids)
        # Determine dominant label per subject
        subj_labels = np.array([
            np.bincount(labels[subject_ids == s].astype(int)).argmax()
            for s in unique_subjs
        ])
        pos_subjs = unique_subjs[subj_labels == 1]
        neg_subjs = unique_subjs[subj_labels == 0]

        # Resolve per-class limits
        n_pos_limit = _resolve_param(max_subjects, pos=True)
        n_neg_limit = _resolve_param(max_subjects, pos=False)

        # If an int was passed, split evenly
        if isinstance(max_subjects, (int, np.integer)):
            n_pos_limit = max_subjects // 2
            n_neg_limit = max_subjects // 2

        # Apply limits (None = keep all in that class)
        n_pos = len(pos_subjs) if n_pos_limit is None else min(n_pos_limit, len(pos_subjs))
        n_neg = len(neg_subjs) if n_neg_limit is None else min(n_neg_limit, len(neg_subjs))
        kept_pos = pos_subjs[:n_pos]
        kept_neg = neg_subjs[:n_neg]
        kept_subjs = np.concatenate([kept_pos, kept_neg])

        keep_mask = np.isin(subject_ids, kept_subjs)
        time_series = time_series[keep_mask]
        labels = labels[keep_mask]
        subject_ids = subject_ids[keep_mask]

        # Re-assign sequential IDs (1-indexed)
        _, subject_ids = np.unique(subject_ids, return_inverse=True)
        subject_ids = subject_ids + 1
        print(f"Sampled {n_pos} epilepsy + {n_neg} no-epilepsy = "
              f"{n_pos + n_neg} subjects from {len(unique_subjs)} total")

    # ── Normalize ──
    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)

    time_series = time_series.astype(np.float32)
    labels = labels.astype(np.int8)

    # ── Report ──
    n_epi = int(labels.sum())
    n_no_epi = int(len(labels) - n_epi)
    print(f"\nTotal samples: {len(labels)}")
    print(f"  Epilepsy     (label=1): {n_epi}")
    print(f"  No epilepsy  (label=0): {n_no_epi}")
    print(f"  Shape: {time_series.shape}")

    np.savez(output_path, timeseries=time_series,
             labels=labels, subject_id=subject_ids, hz=hz)
    print(f"Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    disease_tuep_preprocess("../data/TUEP", hz=200, max_windows_per_subject=(80, None), max_subjects=None)
