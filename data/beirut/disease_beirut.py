import os
from datetime import date, datetime, time, timedelta
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
# Seizure annotation times from Preprocessing_Scripts/Seizure_times.py
# Format: {patient: {record_num: [(hour, min, sec, duration_sec), ...], ...}, ...}
# (hour, min, sec) are absolute times of day.
# Patients 11–15 only (patient 10 excluded).
# ═══════════════════════════════════════════════════════════════════════════════

_SEIZURE_TIMES = {
    15: {1: [(17, 18, 8, 50)],
         2: [(22, 49, 24, 47)],
         3: [(2, 57, 4, 13)],
         4: [(5, 3, 26, 56),
             (6, 23, 29, 20)]},
    14: {1: [(14, 32, 2, 28),
             (15, 34, 32, 134)],
         2: [(16, 20, 58, 32),
             (17, 50, 56, 10)],
         3: [(20, 20, 46, 31),
             (21, 2, 4, 26),
             (21, 27, 49, 40),
             (21, 50, 24, 40)]},
    13: {1: [(2, 31, 9, 52)],
         2: [(3, 33, 4, 25),
             (4, 38, 59, 16)],
         3: [(6, 45, 51, 18)],
         4: [(10, 51, 41, 30),
             (12, 18, 22, 24)]},
    12: {1: [(1, 56, 14, 76),
             (2, 20, 12, 104)],
         2: [(5, 50, 55, 118)],
         3: [(6, 40, 30, 82),
             (7, 21, 36, 164),
             (8, 37, 48, 113)]},
    11: {1: [(15, 8, 55, 64)],
         2: [(18, 11, 38, 51)],
         3: [(19, 18, 38, 91),
             (20, 6, 38, 83),
             (20, 53, 22, 76),
             (21, 27, 24, 73)],
         4: [(23, 55, 35, 1358)]},
}

_FILES_MAP = {
    15: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
    14: ["Record1.edf", "Record2.edf", "Record3.edf"],
    13: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
    12: ["Record1.edf", "Record2.edf", "Record3.edf"],
    11: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
}

# EEG channel order for reordering raw EDF — ends up with 19 EEG + 2 ref + EKG + Manual
_CHANNEL_ORDER = [
    'EEG Fp1-Ref', 'EEG F3-Ref', 'EEG C3-Ref', 'EEG P3-Ref', 'EEG O1-Ref',
    'EEG Fp2-Ref', 'EEG F4-Ref', 'EEG C4-Ref', 'EEG P4-Ref', 'EEG O2-Ref',
    'EEG F7-Ref', 'EEG T3-Ref', 'EEG T5-Ref', 'EEG F8-Ref', 'EEG T4-Ref',
    'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref',
    'EEG A2-Ref', 'EEG A1-Ref', 'ECG EKG', 'Manual',
]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class DiseaseBeirutDataset(BaseDataset):
    """Beirut seizure-classification dataset.

    Task: given a 1‑minute sliding window of 19‑channel EEG time series,
    classify the window as positive (pre‑ictal) or negative (inter‑ictal).

    - Positive (label 1): seizure onset inside [window_end, window_end + 10 min]
    - Negative (label 0): ≥ 10 min away from any seizure
    - Ictal windows and buffer-zone windows are discarded.

    Patients 11–15, patient 10 excluded.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(DiseaseBeirutDataset, self).__init__(data_config, k, train, one_hot=one_hot,
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
        self.data_config.output_dim = 2
        self.data_config.task_type = DataConfig.TASK_CLASSIFICATION
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
        correlation = self.sparsify_fc(correlation, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)

        window_size = 6 * self.hz
        step_size = (time_series.size(-1) - window_size) // 9
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)
        DFC = self.sparsify_fc(DFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)

        return {
                'correlation': correlation,
                'DFC': DFC,
                'labels': labels}


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


def disease_beirut_preprocess(path_beirut="../data/Beirut", hz=200,
                      window_sec=60, stride_sec=30,
                      predict_sec=600, buffer_sec=600,
                      max_windows_per_subject=None,
                      max_subjects=None):
    """Preprocess the Beirut epilepsy dataset for seizure‑prediction.

    Parameters
    ----------
    path_beirut : str
        Path to the Beirut dataset directory (which contains ``Raw_EDF_Files/``).
    hz : int
        Resampling rate in Hz.
    window_sec : int
        Duration of each input EEG window (seconds).  Default 60 s.
    stride_sec : int
        Sliding‑window stride (seconds).  Default 30 s → 50 % overlap.
    predict_sec : int
        Prediction horizon: a window is pre‑ictal if a seizure starts within
        ``predict_sec`` seconds after the window ends.  Default 600 s (10 min).
    buffer_sec : int
        Minimum distance from any seizure for an inter‑ictal window.  Default
        600 s (10 min).
    """
    output_path = os.path.join(path_beirut, "beirut_disease.npy")

    window_samples = window_sec * hz
    stride_samples = stride_sec * hz
    predict_samples = predict_sec * hz
    buffer_samples = buffer_sec * hz

    time_series_all = []
    labels_all = []
    subject_ids_all = []

    for patient in [15, 14, 13, 12, 11]:
        files_list = _FILES_MAP[patient]
        patient_seizures = _SEIZURE_TIMES[patient]

        for file_id, file_name in enumerate(tqdm(files_list, desc=f"Patient {patient}")):
            record_num = file_id + 1

            if record_num not in patient_seizures:
                continue

            file_path = os.path.join(
                path_beirut, "Raw_EDF_Files",
                f"p{patient}_{file_name}")
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

            # ── Channel re‑order, average reference, resample ──
            raw.reorder_channels(_CHANNEL_ORDER)
            raw.set_eeg_reference('average', verbose=False)
            raw = raw.copy().resample(sfreq=hz, verbose=False)

            eeg, _ = raw[:, :]
            eeg = eeg[:19, :]          # keep 19 EEG channels only

            # ── Record start time (for seizure‑time conversion) ──
            record_dt = raw.info['meas_date']
            if hasattr(record_dt, 'tzinfo') and record_dt.tzinfo is not None:
                record_dt = record_dt.replace(tzinfo=None)

            # ── Seizure intervals in sample indices ──
            seizure_intervals = []
            for (h, m, s, dur) in patient_seizures[record_num]:
                seizure_dt = datetime.combine(record_dt.date(),
                                              time(h, m, s))
                if seizure_dt < record_dt:
                    seizure_dt += timedelta(days=1)
                offset_sec = (seizure_dt - record_dt).total_seconds()
                start_samp = int(offset_sec * hz)
                end_samp = start_samp + int(dur * hz)
                seizure_intervals.append((start_samp, end_samp))

            total_samples = eeg.shape[1]

            # ── Per‑window classification ──
            preictal_samples = []    # (window, label=1)
            interictal_samples = []  # (window, label=0)

            for start in range(0, total_samples - window_samples,
                               stride_samples):
                end = start + window_samples

                # ── Skip ictal (overlaps seizure) ──
                is_ictal = any(
                    start < sz_end and end > sz_start
                    for sz_start, sz_end in seizure_intervals
                )
                if is_ictal:
                    continue

                # ── Pre‑ictal: seizure onset inside [end, end + predict] ──
                is_preictal = any(
                    end <= sz_start < end + predict_samples
                    for sz_start, _ in seizure_intervals
                )

                if is_preictal:
                    label = 1
                    preictal_samples.append(
                        (eeg[:, start:end], label))
                else:
                    # ── Buffer check for inter‑ictal ──
                    too_close = any(
                        end > sz_start - buffer_samples
                        and start < sz_end + buffer_samples
                        for sz_start, sz_end in seizure_intervals
                    )
                    if too_close:
                        continue

                    label = 0
                    interictal_samples.append(
                        (eeg[:, start:end], label))

            # ── 1:1 balance per record ──
            n_pre = len(preictal_samples)
            n_int = len(interictal_samples)
            n_keep = min(n_pre, n_int)
            if n_keep == 0:
                continue

            rng = np.random.RandomState(42)
            if n_pre > n_keep:
                idx = rng.choice(n_pre, n_keep, replace=False)
                preictal_samples = [preictal_samples[i] for i in idx]
            if n_int > n_keep:
                idx = rng.choice(n_int, n_keep, replace=False)
                interictal_samples = [interictal_samples[i] for i in idx]

            for window, label in preictal_samples + interictal_samples:
                time_series_all.append(window)
                labels_all.append(label)
                subject_ids_all.append(float(patient))

    # ── Stack and normalize ──
    time_series_all = np.array(time_series_all)
    labels_all = np.array(labels_all)
    subject_ids_all = np.array(subject_ids_all)

    # ── Per-subject window cap (evenly spaced sampling along time axis) ──
    if max_windows_per_subject is not None:
        unique_subjs = np.unique(subject_ids_all)
        keep_mask = np.ones(len(subject_ids_all), dtype=bool)
        n_capped = 0
        for subj in unique_subjs:
            idx = np.where(subject_ids_all == subj)[0]
            subj_label = labels_all[idx[0]]
            cap = _resolve_param(max_windows_per_subject, pos=(subj_label == 1))
            if cap is not None and len(idx) > cap:
                sample_idx = np.linspace(0, len(idx) - 1, cap, dtype=int)
                keep_mask[idx] = False
                keep_mask[idx[sample_idx]] = True
                n_capped += 1
        time_series_all = time_series_all[keep_mask]
        labels_all = labels_all[keep_mask]
        subject_ids_all = subject_ids_all[keep_mask]
        print(f"Capped {n_capped} subjects (evenly spaced)")

    # ── Per-class stratified subject sampling (deterministic: keep first N) ──
    if max_subjects is not None:
        unique_subjs = np.unique(subject_ids_all)
        subj_labels = np.array([
            np.bincount(labels_all[subject_ids_all == s].astype(int)).argmax()
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

        keep_mask = np.isin(subject_ids_all, kept_subjs)
        time_series_all = time_series_all[keep_mask]
        labels_all = labels_all[keep_mask]
        subject_ids_all = subject_ids_all[keep_mask]

        _, subject_ids_all = np.unique(subject_ids_all, return_inverse=True)
        subject_ids_all = subject_ids_all + 1
        print(f"Sampled {n_pos} pre-ictal + {n_neg} inter-ictal = "
              f"{n_pos + n_neg} subjects from {len(unique_subjs)} total")


    print(f"Total samples: {len(labels_all)}, "
          f"Pre-ictal: {int(labels_all.sum())}, "
          f"Inter-ictal: {int((1 - labels_all).sum())}")

    np.save(output_path, {
        "timeseries": time_series_all,
        "labels": labels_all,
        "subject_id": subject_ids_all,
        "hz": hz,
    })
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    disease_beirut_preprocess(
        path_beirut="../data/Beirut",
        hz=200,
        window_sec=60,
        stride_sec=30,
        predict_sec=600,
        buffer_sec=600,
    )
