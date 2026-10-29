"""EEG-Deformer zero-shot / test evaluation for the LLM4FC matrix.

Loads a trained .ckpt and evaluates window-level AUC/Acc on a target
domain's test npz.

Run: python eval_LLM4FC_zs.py --ckpt <best.ckpt> --data-dir <target 10s dir>
"""
import argparse
import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from models.model_handler import DLModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data-dir', type=str, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    torch.serialization.add_safe_globals([argparse.Namespace])
    model = DLModel.load_from_checkpoint(args.ckpt, map_location='cpu')
    model = model.cuda(args.gpu).eval()

    d = np.load(osp.join(args.data_dir, 'test.npz'), allow_pickle=True)
    X = torch.from_numpy(d['X'].astype(np.float32))
    y = torch.from_numpy(d['y'].astype(np.int64))
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.cuda(args.gpu))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            labels.append(yb)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()

    auc = roc_auc_score(labels, probs)
    preds = (probs > 0.5).astype(np.int64)
    acc = accuracy_score(labels, preds)
    print(f'ZS/TEST RESULT: AUC={auc:.6f} Acc={acc:.6f} N={len(labels)}')

    # 持久化结果（conclusion.py 溯源用）
    import json
    import os
    out = osp.join(osp.dirname(osp.abspath(args.ckpt)), '..', 'results')
    os.makedirs(out, exist_ok=True)
    src = osp.basename(osp.dirname(osp.abspath(args.ckpt)))
    tgt = osp.basename(osp.normpath(args.data_dir)).replace('llm4fc_data_10s', '')
    tgt = {'': 'CAUEEG', '_tuab': 'TUAB', '_tuep': 'TUEP'}.get(tgt, tgt)
    rec = {'ckpt': osp.basename(args.ckpt), 'source': src, 'target': tgt,
           'auc': float(auc), 'accuracy': float(acc), 'n': int(len(labels))}
    fname = osp.join(out, f'{src}__{tgt}.json')
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(rec, f)
    print(f'结果已写入 {fname}')


if __name__ == '__main__':
    main()
