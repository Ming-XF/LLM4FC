"""Prepare disease NPZ data for OFA classification (npz format).

Reads a project-shared disease npz and writes train/val/test npz files
using the SAME subject-grouped 60/20/20 split indices stored in the source
npz (identical to the main framework's splits).

Per-sample per-channel z-score standardization is baked in at this stage
(standard EEG deep-learning preprocessing; the OFA harness normalization
path is not used).
"""
import argparse
import os

import numpy as np

SRC = '/root/autodl-tmp/data/CAUEEG/caueeg_disease.npz'
HERE = os.path.dirname(os.path.abspath(__file__))


def standardize_per_sample(X):
    """X: (N, C, L) float32 -> per (sample, channel) z-score."""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default=SRC,
                    help='source disease npz (default CAUEEG)')
    ap.add_argument('--seg_samples', type=int, default=12000,
                    help='segment length per window (12000=60s, 2000=10s)')
    ap.add_argument('--out_dir', default=HERE,
                    help='output directory for the three split npz files')
    args = ap.parse_args()

    d = np.load(args.src, allow_pickle=True)
    X = d['timeseries'].astype(np.float32)[:, :, :args.seg_samples]
    y = d['labels'].astype(np.int64)                # (N,)
    subj = d['subject_id'].astype(np.int64)         # (N,)
    splits = {
        'train': d['split_train_index'].astype(np.int64),
        'val': d['split_val_index'].astype(np.int64),
        'test': d['split_test_index'].astype(np.int64),
    }

    X = standardize_per_sample(X)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, idx in splits.items():
        out = os.path.join(args.out_dir, f'caueeg_disease_{name}.npz')
        np.savez(
            out,
            X=X[idx],
            y=y[idx],
            subject=subj[idx],
            split_index=idx,
            seg_samples=args.seg_samples,
        )
        print(f'{out}: X={X[idx].shape} y={y[idx].shape} '
              f'subjects={np.unique(subj[idx]).size}')
    print('done')


if __name__ == '__main__':
    main()
