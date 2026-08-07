import os
from random import shuffle

import mne
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _init_worker():
    """Set BLAS threads to 1 so each worker is single-threaded."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


from sklearn.model_selection import train_test_split

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Standard 19-channel 10-20 EEG order (CAUEEG-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

_TUAB_CHANNEL_ORDER = [
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

class GenderTUABDataset(BaseDataset):
    """TUAB gender classification — Female vs Male binary classification.

    Task: given a 1‑minute window of 19‑channel EEG time series,
    classify the subject's gender as Female (0) or Male (1).

    Uses all subjects from the TUH Abnormal EEG Corpus regardless of
    normal/abnormal label.  Gender is extracted from the EDF file header.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(GenderTUABDataset, self).__init__(data_config, k, train, one_hot=one_hot,
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

        # ── 使用预处理阶段预计算的分层划分 ──
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

        SFC = self.connectivity(time_series)
        SFC = self.sparsify_fc(SFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)
        window_size = 12 * self.hz
        step_size = (60 * self.hz - window_size) // 9
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)
        DFC = self.sparsify_fc(DFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)

        return {
                'DFC': DFC,
                'correlation': SFC,
                'labels': labels,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — worker (module-level for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════════

def _process_tuab_file_gender(args):
    """Process a single TUAB EDF file into 1‑minute windows with gender label.

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
        # Unknown or missing sex — skip this file
        return None, None, None

    gender_label = 0 if sex_val == 2 else 1  # F=0, M=1

    # ── Build mapping from normalised channel name → EDF index ──
    edf_ch_names = raw.info['ch_names']
    ch_index = {}
    for i, ch in enumerate(edf_ch_names):
        norm = _normalize_channel(ch)
        ch_index[norm] = i

    # ── Check all 19 standard channels exist ──
    missing = [ch for ch in _TUAB_CHANNEL_ORDER if ch not in ch_index]
    if missing:
        return None, None, None

    # ── Pick and reorder the 19 EEG channels ──
    pick_idx = [ch_index[ch] for ch in _TUAB_CHANNEL_ORDER]
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

    return data.astype(np.float32), labels_arr, subj_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_param(val, pos):
    """Resolve a per-class parameter that may be an int (both classes) or
    tuple ``(pos_val, neg_val)``.
    """
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return val
    return val[0] if pos else val[1]


def gender_tuab_preprocess(path="../data/TUAB", hz=200,
                           max_windows_per_subject=4,
                           max_subjects=None,
                           train_split=0.7, val_split=0.15):
    """Preprocess the TUH Abnormal EEG Corpus for gender (M/F) classification.

    Reads all EDF files under ``edf/train/`` and ``edf/eval/``, extracts the
    standard 19‑channel 10-20 EEG, resamples to ``hz`` Hz, and segments into
    non‑overlapping 1‑minute windows.

    Gender labels are extracted from the EDF file header (Patient ID field):
    Female=0, Male=1.  Files with unknown/missing sex are skipped.

    Parameters
    ----------
    path : str
        Path to the TUAB dataset root (contains ``edf/`` subdirectory).
    hz : int
        Resampling rate in Hz.  Default 200.
    """
    edf_root = os.path.join(path, "edf")
    output_path = os.path.join(path, "tuab_gender.npz")

    # ── Collect all EDF files ──
    file_list = []
    subject_map = {}

    for split in ['train', 'eval']:
        for cls_name in ['normal', 'abnormal']:
            cls_dir = os.path.join(edf_root, split, cls_name, '01_tcp_ar')
            if not os.path.isdir(cls_dir):
                print(f"  [SKIP] directory not found: {cls_dir}")
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.endswith('.edf'):
                    continue
                edf_path = os.path.join(cls_dir, fname)
                subj_str = fname.split('_s')[0]
                file_list.append((edf_path, subj_str))

    print(f"Found {len(file_list)} EDF files in {edf_root}")

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

    n_workers = min(cpu_count(), len(task_args), 48)
    print(f"Processing {len(task_args)} files with {n_workers} workers...")

    ts_list, lbl_list, subj_list = [], [], []
    skipped = 0
    skipped_sex = 0
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        for data, labels_arr, subj_arr in tqdm(
                pool.imap_unordered(_process_tuab_file_gender, task_args),
                total=len(task_args),
                desc="Processing TUAB gender"):
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
    print("Concatenating arrays...")
    time_series = np.concatenate(ts_list, axis=0)
    labels = np.concatenate(lbl_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)
    print(f"Concatenated: time_series={time_series.shape}, labels={labels.shape}")

    # ── Per-subject window cap (evenly spaced sampling along time axis) ──
    if max_windows_per_subject is not None:
        unique_subjs = np.unique(subject_ids)
        keep_mask = np.ones(len(subject_ids), dtype=bool)
        n_capped = 0
        for subj in unique_subjs:
            idx = np.where(subject_ids == subj)[0]
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
        subj_labels = np.array([
            np.bincount(labels[subject_ids == s].astype(int)).argmax()
            for s in unique_subjs
        ])
        pos_subjs = unique_subjs[subj_labels == 1]
        neg_subjs = unique_subjs[subj_labels == 0]

        n_pos_limit = _resolve_param(max_subjects, pos=True)
        n_neg_limit = _resolve_param(max_subjects, pos=False)

        if isinstance(max_subjects, (int, np.integer)):
            n_pos_limit = max_subjects // 2
            n_neg_limit = max_subjects // 2

        n_pos = len(pos_subjs) if n_pos_limit is None else min(n_pos_limit, len(pos_subjs))
        n_neg = len(neg_subjs) if n_neg_limit is None else min(n_neg_limit, len(neg_subjs))
        kept_pos = pos_subjs[:n_pos]
        kept_neg = neg_subjs[:n_neg]
        kept_subjs = np.concatenate([kept_pos, kept_neg])

        keep_mask = np.isin(subject_ids, kept_subjs)
        time_series = time_series[keep_mask]
        labels = labels[keep_mask]
        subject_ids = subject_ids[keep_mask]

        _, subject_ids = np.unique(subject_ids, return_inverse=True)
        subject_ids = subject_ids + 1
        print(f"Sampled {n_pos} Male + {n_neg} Female = "
              f"{n_pos + n_neg} subjects from {len(unique_subjs)} total")

    # ── Per-subject stratified random split ──
    unique_subjs = np.unique(subject_ids)
    subj_labels = np.array([
        np.bincount(labels[subject_ids == s].astype(int)).argmax()
        for s in unique_subjs
    ])
    train_subjs, rest_subjs = train_test_split(
        unique_subjs, test_size=1.0 - train_split,
        stratify=subj_labels, random_state=42)
    rest_mask = np.isin(unique_subjs, rest_subjs)
    val_frac = val_split / (1.0 - train_split)
    val_subjs, test_subjs = train_test_split(
        rest_subjs, test_size=1.0 - val_frac,
        stratify=subj_labels[rest_mask], random_state=42)

    split_train_index = np.where(np.isin(subject_ids, train_subjs))[0]
    split_val_index = np.where(np.isin(subject_ids, val_subjs))[0]
    split_test_index = np.where(np.isin(subject_ids, test_subjs))[0]

    train_m = int(sum(subj_labels[np.isin(unique_subjs, train_subjs)] == 1))
    train_f = len(train_subjs) - train_m
    val_m = int(sum(subj_labels[np.isin(unique_subjs, val_subjs)] == 1))
    val_f = len(val_subjs) - val_m
    test_m = int(sum(subj_labels[np.isin(unique_subjs, test_subjs)] == 1))
    test_f = len(test_subjs) - test_m
    print(f"\nSplit: train={len(train_subjs)} subj ({train_f}F/{train_m}M, {len(split_train_index)} samples)")
    print(f"  val={len(val_subjs)} subj ({val_f}F/{val_m}M, {len(split_val_index)} samples)")
    print(f"  test={len(test_subjs)} subj ({test_f}F/{test_m}M, {len(split_test_index)} samples)")

    # ── Normalize ──

    labels = labels.astype(np.int8)

    # ── Report ──
    n_f = int((labels == 0).sum())
    n_m = int((labels == 1).sum())
    print(f"\nTotal samples: {len(labels)}")
    print(f"  Female (label=0): {n_f}")
    print(f"  Male   (label=1): {n_m}")
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
    gender_tuab_preprocess("../data/TUAB", hz=200,
                           max_subjects=None)
