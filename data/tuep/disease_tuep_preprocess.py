"""TUEP 疾病任务 v2 预处理(从 git 47544d1 恢复并按 v2 约定重建)。

与旧版差异(v2 修复点):
  1. 通道处理换统一管线 prepare_common_19_channel_raw
     (按名字选标准 19 导联 + average reference + 200Hz 重采样;
      缺失标准导联的文件跳过,与旧版行为一致);
  2. 多进程改有序收集(imap,结果按文件收集顺序拼接);
  3. NPZ 写 v2 元数据(preprocessing_version/channel_names/reference/unit);
  4. 原子写入 save_npz_atomic。

其余与旧版一致:caps=(30, 60)(癫痫/非癫痫)、max_subjects=None、
70/15/15 分层划分 seed 42、00_epilepsy/01_no_epilepsy 目录结构、
session 只取 01_tcp_ar montage(无则跳过)。

运行:python -m data.tuep.disease_tuep_preprocess
"""
import os
from multiprocessing import Pool, cpu_count

import mne
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..preprocess import (
    EEG_19_CHANNEL_ORDER,
    EEG_PREPROCESSING_VERSION,
    prepare_common_19_channel_raw,
    save_npz_atomic,
)


def _init_worker():
    """Set BLAS threads to 1 so each worker is single-threaded."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


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
        data = prepare_common_19_channel_raw(raw, hz)   # (19, n) @ hz, avgref
    except Exception:
        return None, None, None

    # ── Truncate to whole minutes, reshape to 1-minute windows ──
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

    return data.astype(np.float32), labels_arr, subj_arr


def _resolve_param(val, pos):
    """Resolve a per-class parameter that may be an int (both classes) or
    tuple ``(pos_val, neg_val)``."""
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return val
    return val[0] if pos else val[1]


def disease_tuep_preprocess(path="../data/TUEP", hz=200, max_windows_per_subject=60,
                            max_subjects=None,
                            train_split=0.7, val_split=0.15,
                            output_path=None):
    if output_path is None:
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

    # ── Build multiprocessing task args(按文件收集顺序;imap 有序)──
    task_args = [
        (edf_path, label, hz, subject_map[subj_str])
        for edf_path, label, subj_str in file_list
    ]

    n_workers = min(cpu_count(), len(task_args), 48)
    print(f"Processing {len(task_args)} files with {n_workers} workers...")

    ts_list, lbl_list, subj_list = [], [], []
    skipped = 0
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        for data, labels_arr, subj_arr in tqdm(
                pool.imap(_process_tuep_file, task_args),
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
        print(f"Sampled {n_pos} epilepsy + {n_neg} no-epilepsy = "
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

    labels = labels.astype(np.int8)

    train_ep = int(sum(subj_labels[np.isin(unique_subjs, train_subjs)] == 1))
    train_no = len(train_subjs) - train_ep
    val_ep = int(sum(subj_labels[np.isin(unique_subjs, val_subjs)] == 1))
    val_no = len(val_subjs) - val_ep
    test_ep = int(sum(subj_labels[np.isin(unique_subjs, test_subjs)] == 1))
    test_no = len(test_subjs) - test_ep
    print(f"\nSplit: train={len(train_subjs)} subj ({train_no}no-epi/{train_ep}epi, {len(split_train_index)} samples)")
    print(f"  val={len(val_subjs)} subj ({val_no}no-epi/{val_ep}epi, {len(split_val_index)} samples)")
    print(f"  test={len(test_subjs)} subj ({test_no}no-epi/{test_ep}epi, {len(split_test_index)} samples)")

    n_epi = int(labels.sum())
    n_no_epi = int(len(labels) - n_epi)
    print(f"\nTotal samples: {len(labels)}")
    print(f"  Epilepsy     (label=1): {n_epi}")
    print(f"  No epilepsy  (label=0): {n_no_epi}")
    print(f"  Shape: {time_series.shape}")

    save_npz_atomic(
        output_path,
        timeseries=time_series,
        labels=labels,
        subject_id=subject_ids,
        hz=hz,
        split_train_index=split_train_index,
        split_val_index=split_val_index,
        split_test_index=split_test_index,
        preprocessing_version=EEG_PREPROCESSING_VERSION,
        channel_names=np.asarray(EEG_19_CHANNEL_ORDER),
        reference='average_19_channels',
        unit='V',
    )
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    disease_tuep_preprocess("../data/TUEP", hz=200,
                            max_windows_per_subject=(30, 60), max_subjects=None)
