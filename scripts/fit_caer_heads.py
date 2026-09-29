"""CAER-v2: 源域闭式拟合双任务冻结读出头（18 维 FC+频谱证据）。

- 年龄任务: Ridge（alpha 由源域 val 选）→ output_dir/stats_ridge.pt
- 疾病任务: LogisticRegression（C 由源域 val AUC 选）→
  output_dir/stats_logistic.pt
- 两者共享同一确定性证据提取器（源域-free，无目标域泄漏）

运行: python3 scripts/fit_caer_heads.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from data.data_config import DataConfig
from data import AgeCAUEEGDataset, DiseaseCAUEEGDataset
from config import init_config
from model.TimeLLM.graph_evidence import compute_graph_evidence


def collect(dataset_cls, batch=64):
    args = init_config()
    args.dataset = dataset_cls.__name__.replace('Dataset', '')
    ds = dataset_cls(DataConfig(args))
    out = {'train': None, 'val': None}
    for split, mode in [('train', 'train'), ('val', 'val')]:
        ds.train = True if mode == 'train' else False
        ds._mode = mode
        idx = ds._active_index
        feats = []
        with torch.no_grad():
            for i in range(0, len(idx), batch):
                items = [ds[j] for j in range(i, min(i + batch, len(idx)))]
                dfc = torch.stack([b['DFC'] for b in items]).float()
                spec = torch.stack([b['spectral'] for b in items]).float()
                _, stats = compute_graph_evidence(dfc, spec)
                feats.append(stats.numpy())
        X = np.concatenate(feats)
        y = ds.all_data['labels'][idx].astype(np.float64)
        out[split] = (X, y)
    return out


def save_head(path, mu, sigma, w, b):
    torch.save({'mu': torch.tensor(mu), 'sigma': torch.tensor(sigma),
                'w': torch.tensor(w, dtype=torch.float32),
                'b': torch.tensor(b, dtype=torch.float32)}, path)


def main():
    # ── 年龄: ridge 头 ──
    d = collect(AgeCAUEEGDataset)
    Xtr, ytr = d['train']
    Xva, yva = d['val']
    mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xs, Xvs = (Xtr - mu) / sigma, (Xva - mu) / sigma
    best_a, best_r = 0.3, -9
    for a in [0.3, 1, 3, 10, 30, 100, 300, 1000]:
        m = Ridge(alpha=a).fit(Xs, ytr)
        r = pearsonr(m.predict(Xvs), yva)[0]
        print(f'age alpha={a:6}: val r={r:+.4f}')
        if r > best_r:
            best_r, best_a = r, a
    m = Ridge(alpha=best_a).fit(Xs, ytr)
    save_head('output_dir/stats_ridge.pt', mu, sigma, m.coef_, m.intercept_)
    print(f'age: best alpha={best_a}, val r={best_r:+.4f} -> stats_ridge.pt')

    # ── 疾病: logistic 头 ──
    d = collect(DiseaseCAUEEGDataset)
    Xtr, ytr = d['train']
    Xva, yva = d['val']
    mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xs, Xvs = (Xtr - mu) / sigma, (Xva - mu) / sigma
    best_c, best_auc = 0.01, 0.0
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
        m = LogisticRegression(C=C, max_iter=3000).fit(Xs, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xvs)[:, 1])
        print(f'disease C={C:6}: val AUC={auc:.4f}')
        if auc > best_auc:
            best_auc, best_c = auc, C
    m = LogisticRegression(C=best_c, max_iter=3000).fit(Xs, ytr)
    save_head('output_dir/stats_logistic.pt', mu, sigma,
              m.coef_[0], m.intercept_[0])
    print(f'disease: best C={best_c}, val AUC={best_auc:.4f} '
          f'-> stats_logistic.pt')


if __name__ == '__main__':
    main()
