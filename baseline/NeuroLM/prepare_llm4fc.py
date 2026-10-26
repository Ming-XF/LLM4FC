"""
Convert LLM4FC project npz datasets (19-channel, 200 Hz, 60 s windows) into the
NeuroLM pickle format used by its downstream loaders.

Conventions (to match NeuroLM's dataset_maker exactly):
  - one pickle per 10 s segment (2000 samples), one file per segment
  - X stored in uV (NeuroLM loaders divide by 100 afterwards)
  - subject-level train/val/test split taken from the npz precomputed indices
  - fields: {"X": (19, 2000) float32, "y": int, "subject_id": ..., "window_id": int}

Run from the LLM4FC repo root:
    python baseline/NeuroLM/prepare_llm4fc.py --dataset CAUEEG
"""

import argparse
import json
import os
import pickle
import re

import numpy as np
from tqdm import tqdm

# Standard 19-channel 10-20 order used by LLM4FC (CAUEEG-compatible)
CHANNELS_19 = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2',
               'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz']

DATASET_PATHS = {
    'CAUEEG': '../data/CAUEEG/caueeg_disease.npz',
    'TUAB': '../data/TUAB/tuab_disease.npz',
    'TUEP': '../data/TUEP/tuep_disease.npz',
}

SAMPLE_RATE = 200          # Hz
SEG_SECS = 10              # segment length in seconds (matches NeuroLM TUAB protocol)
WINDOW_SECS = 60           # full window length in seconds


def sanitize(s):
    return re.sub(r'[^0-9A-Za-z_-]', '_', str(s))


def main():
    parser = argparse.ArgumentParser(description='LLM4FC npz -> NeuroLM pkl segments')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_PATHS),
                        help='which dataset to convert')
    parser.add_argument('--npz_path', default=None,
                        help='override npz path (default: ../data/<DATASET>/...)')
    parser.add_argument('--out_dir', default='llm4fc_data',
                        help='output root; segment files go to <out_dir>/<DATASET>/{train,val,test}/')
    args = parser.parse_args()

    npz_path = args.npz_path or DATASET_PATHS[args.dataset]
    raw = np.load(npz_path, allow_pickle=True)
    data = dict(raw) if hasattr(raw, 'files') else raw.item()

    time_series = data['timeseries']          # (N, 19, 12000)
    labels = np.asarray(data['labels']).astype(int)
    subject_id = data['subject_id']
    hz = int(np.asarray(data['hz']).item())
    assert hz == SAMPLE_RATE, f'expected {SAMPLE_RATE} Hz, got {hz}'
    unit = str(data['unit']) if 'unit' in data else '?'
    print(f'[{args.dataset}] {time_series.shape} windows, {len(set(subject_id))} subjects, '
          f'hz={hz}, unit={unit}, pos rate={labels.mean():.3f}')

    # channel sanity check
    ch_names = data['channel_names'] if 'channel_names' in data else None
    if ch_names is not None:
        norm = [re.sub(r'[^A-Za-z0-9]', '', c).lower() for c in ch_names]
        ref = [re.sub(r'[^A-Za-z0-9]', '', c).lower() for c in CHANNELS_19]
        assert norm == ref, f'channel mismatch: {ch_names}'
        print(f'  channels OK: {list(ch_names)[:5]}...')

    # uV conversion (npz unit is Volt); NeuroLM loaders divide by 100 afterwards
    if unit.lower().startswith('v'):
        time_series = time_series * 1e6
        print('  unit V -> uV (x1e6)')
    elif unit.lower().startswith('uv') or unit.lower().startswith('µv'):
        pass
    else:
        print(f'  [WARN] unknown unit {unit!r}, assuming uV already')

    # 新口径（与 LaBraM 对齐）：每 60s 窗只取第一个 10s 段作为样本
    seg_per_win = 1
    seg_len = SAMPLE_RATE * SEG_SECS

    out_root = os.path.join(args.out_dir, args.dataset)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_root, split), exist_ok=True)

    def dump(windows, split):
        count = 0
        for i in tqdm(windows, desc=f'{args.dataset}/{split}'):
            ts = time_series[i]
            for k in range(seg_per_win):
                pkl = {'X': np.ascontiguousarray(ts[:, k * seg_len:(k + 1) * seg_len]),
                       'y': int(labels[i]), 'subject_id': subject_id[i], 'window_id': int(i)}
                fname = f'{sanitize(subject_id[i])}-w{i}-s{k}.pkl'
                with open(os.path.join(out_root, split, fname), 'wb') as f:
                    pickle.dump(pkl, f)
            count += 1
        return count

    n_train = dump(data['split_train_index'], 'train')
    n_val = dump(data['split_val_index'], 'val')
    n_test = dump(data['split_test_index'], 'test')

    meta = {'dataset': args.dataset, 'npz': npz_path, 'unit': unit,
            'seg_secs': SEG_SECS, 'windows': {'train': n_train, 'val': n_val, 'test': n_test},
            'segments_per_window': seg_per_win, 'hz': hz,
            'channels': list(CHANNELS_19)}
    with open(os.path.join(out_root, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[{args.dataset}] done: {n_train}/{n_val}/{n_test} windows -> '
          f'{out_root}/{{train,val,test}} ({seg_per_win} segments each)')


if __name__ == '__main__':
    main()
