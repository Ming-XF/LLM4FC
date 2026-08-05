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

import json
import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

_N_INPUT_WINDOWS = 6
_N_TOTAL_WINDOWS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class FutureFCCAUEEGDataset(BaseDataset):
    """CAUEEG future FC prediction dataset.

    Task: given the first ``k`` dynamic FC windows from a 1‑minute EEG
    segment, predict the remaining ``T−k`` future FC matrices.

    Uses all subjects from the CAUEEG dataset regardless of disease label.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None,
                 n_input_windows=_N_INPUT_WINDOWS,
                 n_total_windows=_N_TOTAL_WINDOWS):
        self.n_input_windows = n_input_windows
        self.n_total_windows = n_total_windows
        super(FutureFCCAUEEGDataset, self).__init__(data_config, k, train, one_hot=one_hot,
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
        window_size = 6 * self.hz
        step_size = (60 * self.hz - window_size) // (self.n_total_windows - 1)
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)
        DFC = self.sparsify_fc(DFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)

        return {
                'DFC': DFC,
                'correlation': SFC,
                'labels': DFC[self.n_input_windows:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — worker
# ═══════════════════════════════════════════════════════════════════════════════

def _process_one_sample_futurefc(args):
    """Process a single EDF sample for future FC prediction (multiprocessing worker).

    Returns dummy labels (all zeros).
    """
    sample, signal_folder, hz = args
    serial = sample['serial']
    subject_id = int(re.findall(r'\d+', serial)[0])
    edf_path = os.path.join(signal_folder, f"{serial}.edf")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, times = raw[:, :]
    data = data[:19, :]

    if data.shape[1] % (hz * 60) != 0:
        data = data[:, :-(data.shape[1] % (hz * 60))]
    data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
    data = np.transpose(data, (1, 0, 2))

    label = np.zeros(data.shape[0], dtype=np.int8)  # dummy
    subj_ids = np.full(data.shape[0], subject_id)

    return data.astype(np.float32), label, subj_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing — main entry point
# ═══════════════════════════════════════════════════════════════════════════════



def futurefc_caueeg_preprocess(path="../data/CAUEEG/", hz=200,
                               max_windows_per_subject=None,
                               train_split=0.7, val_split=0.15):
    """Preprocess the CAUEEG dataset for future FC prediction.

    Uses all subjects from annotation.json.  Saves dummy labels; the actual
    prediction target is computed on-the-fly from the time series.

    Parameters
    ----------
    path : str
        Path to the CAUEEG dataset root.
    hz : int
        Resampling rate in Hz.  Default 200.
    """
    annotation_file = os.path.join(path, "caueeg-dataset/annotation.json")
    signal_folder = os.path.join(path, 'caueeg-dataset/signal', "edf")
    output_path = os.path.join(path, "caueeg_futurefc.npz")

    with open(annotation_file, 'r') as f:
        annotation = json.load(f)

    target_samples = annotation['data']
    print(f"Total subjects: {len(target_samples)}")

    n_workers = min(cpu_count(), len(target_samples), 48)
    print(f"Processing {len(target_samples)} samples with {n_workers} workers...")

    task_args = [(sample, signal_folder, hz) for sample in target_samples]

    ts_list, lbl_list, subj_list = [], [], []
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        for data, label, subj_ids in tqdm(
                pool.imap_unordered(_process_one_sample_futurefc, task_args),
                total=len(task_args), desc="Loading EDF"):
            ts_list.append(data)
            lbl_list.append(label)
            subj_list.append(subj_ids)

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
    futurefc_caueeg_preprocess("../data/CAUEEG/", hz=200)
