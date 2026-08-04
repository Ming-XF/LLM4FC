import os
from random import shuffle

import mne
import numpy as np
import pandas as pd
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
# CAUEEG-compatible channel order (19 EEG, 10-20 system)
# DS raw .set may have a different order; we reorder to this for cross-dataset
# compatibility with CAUEEG2.
# ═══════════════════════════════════════════════════════════════════════════════

_DS_CHANNEL_ORDER = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',   # left hemisphere
    'Fp2', 'F4', 'C4', 'P4', 'O2',   # right hemisphere
    'F7', 'T3', 'T5',                  # left temporal
    'F8', 'T4', 'T6',                  # right temporal
    'Fz', 'Cz', 'Pz',                  # midline
]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class DiseaseDSDataset(BaseDataset):
    """DS dementia dataset — AD vs CN binary classification.

    Task: given a 1‑minute sliding window of 19‑channel resting‑state EEG,
    classify the subject as Alzheimerʼs disease (AD) or healthy control (CN).

    - AD  (label 1): Group A in participants.tsv
    - CN  (label 0): Group C in participants.tsv

    Uses preprocessed derivatives/ .set files, resampled to 200 Hz,
    with channels reordered to match the CAUEEG standard order.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(DiseaseDSDataset, self).__init__(data_config, k, train, one_hot=one_hot,
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

        window_size = 6 * self.hz
        step_size = (60 * self.hz - window_size) // 9
        SFC = self.connectivity(time_series)
        SFC = self.sparsify_fc(SFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)
        DFC = self.sparsify_fc(DFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)

        return {
                'DFC': DFC,
                'correlation': SFC,   # SFC, for backward compat
                'labels': labels,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
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


def disease_ds_preprocess(path="../data/DS", hz=200,
                          max_windows_per_subject=None,
                          max_subjects=None):
    """Preprocess the DS dementia dataset for AD-vs-CN classification.

    Reads preprocessed .set files from ``derivatives/``, selects the first
    19 EEG channels, reorders them to the CAUEEG-compatible standard order,
    resamples to ``hz`` Hz, and segments into non‑overlapping 1‑minute windows.

    Parameters
    ----------
    path : str
        Path to the DS dataset root directory (contains participants.tsv
        and derivatives/ subfolder).
    hz : int
        Resampling rate in Hz.  Default 200 (matches CAUEEG2).
    """
    participants_path = os.path.join(path, "participants.tsv")
    derivatives_dir = os.path.join(path, "derivatives")
    output_path = os.path.join(path, "ds_disease.npz")

    # ── Load participant metadata ──
    participants = pd.read_csv(participants_path, sep='\t')
    # Filter: AD (Group=A) and CN (Group=C) only, exclude FTD (Group=F)
    target_participants = participants[participants['Group'].isin(['A', 'C'])]

    print(f"Total subjects in participants.tsv: {len(participants)}")
    print(f"Target subjects (AD + CN): {len(target_participants)}")
    print(f"  AD: {len(target_participants[target_participants['Group'] == 'A'])}")
    print(f"  CN: {len(target_participants[target_participants['Group'] == 'C'])}")

    ts_list, lbl_list, subj_list = [], [], []

    for _, row in tqdm(target_participants.iterrows(),
                       total=len(target_participants),
                       desc="Processing subjects"):
        subj_id = row['participant_id']           # e.g. "sub-001"
        group = row['Group']                      # "A" or "C"
        label = 1 if group == 'A' else 0          # AD=1, CN=0

        set_path = os.path.join(
            derivatives_dir, subj_id, 'eeg',
            f'{subj_id}_task-eyesclosed_eeg.set')

        if not os.path.exists(set_path):
            print(f"  [SKIP] {set_path} not found — "
                  f"run 'datalad get' to download data files")
            continue

        # ── Read .set file ──
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

        # ── Pick and reorder 19 EEG channels to CAUEEG-compatible order ──
        available = set(raw.info['ch_names'])
        missing = [ch for ch in _DS_CHANNEL_ORDER if ch not in available]
        if missing:
            print(f"  [WARN] {subj_id}: missing channels: {missing}")
            continue

        raw.pick(_DS_CHANNEL_ORDER, verbose=False)

        # ── Resample ──
        raw = raw.copy().resample(sfreq=hz, verbose=False)
        data = raw.get_data()  # (19, n_samples)

        # ── Truncate to whole minutes, reshape to 1‑minute windows ──
        n_total = data.shape[1]
        window_samples = hz * 60
        n_windows = n_total // window_samples
        if n_windows == 0:
            print(f"  [SKIP] {subj_id}: too short ({n_total / hz:.1f}s)")
            continue

        data = data[:, :n_windows * window_samples]
        data = data.reshape(data.shape[0], n_windows, window_samples)
        data = np.transpose(data, (1, 0, 2))  # (n_windows, 19, hz*60)

        labels_arr = np.full(data.shape[0], label)
        subj_ids_arr = np.full(data.shape[0], int(subj_id.split('-')[1]))

        ts_list.append(data)
        lbl_list.append(labels_arr)
        subj_list.append(subj_ids_arr)

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
        print(f"Sampled {n_pos} AD + {n_neg} CN = "
              f"{n_pos + n_neg} subjects from {len(unique_subjs)} total")

    # ── Normalize ──
    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)

    time_series = time_series.astype(np.float32)
    labels = labels.astype(np.int8)

    print(f"\nTotal samples: {len(labels)}")
    print(f"  AD (label=1): {int(labels.sum())}")
    print(f"  CN (label=0): {int(len(labels) - labels.sum())}")
    print(f"  Shape: {time_series.shape}")

    np.savez(output_path, timeseries=time_series,
             labels=labels, subject_id=subject_ids, hz=hz)
    print(f"Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    disease_ds_preprocess("../data/DS", hz=200)
