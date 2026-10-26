"""Convert LLM4FC disease NPZ files into LaBraM fine-tuning pkl samples.

Only the FIRST 10 s of each 60 s window is kept as the sample input
(X: (19, 2000) uV float32, y: int label) — one sample per window, matching
the sample count of the 60 s-window baselines.  Subject-level splits are
taken from the NPZ.

Usage (from baseline/LaBraM):
    python dataset_maker/make_LLM4FC.py --datasets CAUEEG TUAB TUEP
"""

import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

DATASET_PATHS = {
    'CAUEEG': '../../../data/CAUEEG/caueeg_disease.npz',
    'TUAB': '../../../data/TUAB/tuab_disease.npz',
    'TUEP': '../../../data/TUEP/tuep_disease.npz',
}

SEG_SAMPLES = 2000  # 10 s @ 200 Hz, LaBraM TUAB protocol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', nargs='+',
                    default=['CAUEEG', 'TUAB', 'TUEP'],
                    choices=list(DATASET_PATHS))
    ap.add_argument('--out_dir', default='llm4fc_data')
    ap.add_argument('--npz_path', default=None,
                    help='override NPZ path (single dataset)')
    args = ap.parse_args()

    for ds in args.datasets:
        npz_path = args.npz_path or DATASET_PATHS[ds]
        raw = np.load(npz_path, allow_pickle=True)
        d = dict(raw) if hasattr(raw, 'files') else raw.item()
        ts = d['timeseries']
        labels = np.asarray(d['labels']).astype(int)
        subj = np.asarray(d['subject_id']).astype(int)
        n_seg = ts.shape[2] // SEG_SAMPLES
        print(f'[{ds}] {ts.shape}, pos rate {labels.mean():.3f}, '
              f'first-{SEG_SAMPLES // 200}s segment per window')

        for split, key in [('train', 'split_train_index'),
                           ('val', 'split_val_index'),
                           ('test', 'split_test_index')]:
            idx = np.asarray(d[key]).astype(int)
            out_root = os.path.join(args.out_dir, ds, split)
            os.makedirs(out_root, exist_ok=True)
            for i in tqdm(idx, desc=f'{ds}/{split}'):
                # 只取每个 60s 窗口的第一个 10s 段作为样本（丢弃剩余 50s）
                x = np.ascontiguousarray(
                    ts[i][:, 0:SEG_SAMPLES] * 1e6)
                pkl = {'X': x.astype(np.float32),
                       'y': int(labels[i]),
                       'subject_id': int(subj[i])}
                fname = f'{subj[i]}-w{i}.pkl'
                with open(os.path.join(out_root, fname), 'wb') as f:
                    pickle.dump(pkl, f)
            print(f'[{ds}/{split}] {len(idx)} windows')
    print('done')


if __name__ == '__main__':
    main()
