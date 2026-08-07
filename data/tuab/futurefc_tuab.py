import os
from random import shuffle

import mne
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _init_worker():
    """Set BLAS threads to 1 so each worker is single-threaded."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


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


_N_INPUT_WINDOWS = 6
_N_TOTAL_WINDOWS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class FutureFCTUABDataset(BaseDataset):
    """TUAB future FC prediction dataset.

    Task: given the first ``k`` dynamic FC windows from a 1‑minute EEG
    segment, predict the remaining ``T−k`` future FC matrices.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None,
                 n_input_windows=_N_INPUT_WINDOWS,
                 n_total_windows=_N_TOTAL_WINDOWS):
        self.n_input_windows = n_input_windows
        self.n_total_windows = n_total_windows
        super(FutureFCTUABDataset, self).__init__(data_config, k, train, one_hot=one_hot,
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
        SFC = self.connectivity(time_series)
        SFC = self.sparsify_fc(SFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)
        window_size = 12 * self.hz
        step_size = (60 * self.hz - window_size) // (self.n_total_windows - 1)
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)
        DFC = self.sparsify_fc(DFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)

        return {
                'DFC': DFC,
                'correlation': SFC,
                'labels': DFC[self.n_input_windows:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — worker (module-level for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════════

def _process_tuab_file_futurefc(args):
    """Process a single TUAB EDF file for future FC prediction.

    Returns dummy labels (all zeros) — the prediction target is computed
    on-the-fly from the time series in __getitem__.

    Parameters
    ----------
    args : tuple
        (edf_path, hz, subject_id)

    Returns
    -------
    data : ndarray (n_windows, 19, hz*60) or None
    labels : ndarray (n_windows,) or None — dummy zeros
    subj_ids : ndarray (n_windows,) or None
    """
    edf_path, hz, subject_id = args

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None, None, None

    edf_ch_names = raw.info['ch_names']
    ch_index = {}
    for i, ch in enumerate(edf_ch_names):
        norm = _normalize_channel(ch)
        ch_index[norm] = i

    missing = [ch for ch in _TUAB_CHANNEL_ORDER if ch not in ch_index]
    if missing:
        return None, None, None

    pick_idx = [ch_index[ch] for ch in _TUAB_CHANNEL_ORDER]
    raw.pick([edf_ch_names[i] for i in pick_idx], verbose=False)

    raw.set_eeg_reference('average', verbose=False)
    raw = raw.copy().resample(sfreq=hz, verbose=False)
    data = raw.get_data()

    n_total = data.shape[1]
    window_samples = hz * 60
    n_windows = n_total // window_samples
    if n_windows == 0:
        return None, None, None

    data = data[:, :n_windows * window_samples]
    data = data.reshape(data.shape[0], n_windows, window_samples)
    data = np.transpose(data, (1, 0, 2))

    labels_arr = np.zeros(data.shape[0], dtype=np.int8)  # dummy
    subj_arr = np.full(data.shape[0], subject_id, dtype=np.int32)

    return data.astype(np.float32), labels_arr, subj_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — main entry point
# ═══════════════════════════════════════════════════════════════════════════════



def futurefc_tuab_preprocess(path="../data/TUAB", hz=200,
                             max_windows_per_subject=9,
                             train_split=0.7, val_split=0.15):
    """Preprocess the TUH Abnormal EEG Corpus for future FC prediction.

    Signal processing identical to ``tuab_preprocess()`` but saves dummy
    labels.  The prediction target is computed on-the-fly from the time series.

    Parameters
    ----------
    path : str
        Path to the TUAB dataset root.
    hz : int
        Resampling rate in Hz.  Default 200.
    """
    edf_root = os.path.join(path, "edf")
    output_path = os.path.join(path, "tuab_futurefc.npz")

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

    for _, subj_str in file_list:
        if subj_str not in subject_map:
            subject_map[subj_str] = len(subject_map) + 1

    print(f"Unique subjects: {len(subject_map)}")

    task_args = [
        (edf_path, hz, subject_map[subj_str])
        for edf_path, subj_str in file_list
    ]

    n_workers = min(cpu_count(), len(task_args), 48)
    print(f"Processing {len(task_args)} files with {n_workers} workers...")

    ts_list, lbl_list, subj_list = [], [], []
    skipped = 0
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        for data, labels_arr, subj_arr in tqdm(
                pool.imap_unordered(_process_tuab_file_futurefc, task_args),
                total=len(task_args),
                desc="Processing TUAB futurefc"):
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
    futurefc_tuab_preprocess("../data/TUAB", hz=200)
