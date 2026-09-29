"""E8c: 在源域训练集上闭式拟合统计量 ridge 头（跨域可迁移分支）。

- 用 AgeCAUEEG train 拟合、val 选 alpha（无目标域泄漏）
- 保存 (mu, sigma, w, b) 到 output_dir/stats_ridge.pt
- 预测: (stats - mu) / sigma @ w + b（归一化年龄单位）

运行: python3 scripts/fit_stats_ridge.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from data.data_config import DataConfig
from data import AgeCAUEEGDataset
from config import init_config
from model.TimeLLM.graph_evidence import compute_graph_evidence


def collect_stats(ds, idx, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            items = [ds[j] for j in range(i, min(i + batch, len(idx)))]
            dfc = torch.stack([b['DFC'] for b in items]).float()
            _, stats = compute_graph_evidence(dfc)
            out.append(stats.numpy())
    return np.concatenate(out)


def main():
    args = init_config()
    args.dataset = 'AgeCAUEEG'
    dc = DataConfig(args)
    ds = AgeCAUEEGDataset(dc)
    ds.train = True
    X = collect_stats(ds, ds.train_index)
    # npz 标签已是归一化值 [0,1]（与训练损失一致），无需再除以 100
    y = ds.all_data['labels'][ds.train_index].astype(np.float64)
    ds.train = False
    ds._mode = 'val'
    Xv = collect_stats(ds, ds.val_index)
    yv = ds.all_data['labels'][ds.val_index].astype(np.float64)

    mu, sigma = X.mean(0), X.std(0) + 1e-8
    Xs, Xvs = (X - mu) / sigma, (Xv - mu) / sigma

    best_alpha, best_r = 1.0, -9
    for alpha in [0.3, 1, 3, 10, 30, 100, 300, 1000]:
        m = Ridge(alpha=alpha).fit(Xs, y)
        r = pearsonr(m.predict(Xvs), yv)[0]
        print(f'alpha={alpha:6}: val r={r:+.4f}')
        if r > best_r:
            best_r, best_alpha = r, alpha
    m = Ridge(alpha=best_alpha).fit(Xs, y)
    torch.save({'mu': torch.tensor(mu), 'sigma': torch.tensor(sigma),
                'w': torch.tensor(m.coef_, dtype=torch.float32),
                'b': torch.tensor(m.intercept_, dtype=torch.float32)},
               'output_dir/stats_ridge.pt')
    print(f'best alpha={best_alpha}, val r={best_r:+.4f} -> '
          f'output_dir/stats_ridge.pt')


if __name__ == '__main__':
    main()
