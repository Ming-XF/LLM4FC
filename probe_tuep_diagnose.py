#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_tuep_diagnose.py
======================
诊断用线性探针: 判断各数据集 (CAUEEG/TUAB/TUEP) 的 EEG 信号与其疾病标签之间
是否存在可被简单分类器学到的信息。若 TUEP 的真实标签 AUC 与随机标签 AUC 无差别,
说明 TUEP 数据(信号-标签关联)有问题, 而非模型太弱。

特征: 每通道 5 个频带功率 (delta/theta/alpha/beta/gamma), 19 通道 = 95 维。
分类器: 标准化 + LogisticRegression (sklearn)。
划分: 直接使用各 npz 中预计算的 split_train/val/test_index (受试者级无泄漏)。
对照: 每数据集再做 3 次标签随机化, 取测试 AUC 均值。
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]  # Hz
FILES = {
    "CAUEEG": "/root/autodl-tmp/data/CAUEEG/caueeg_disease.npz",
    "TUAB": "/root/autodl-tmp/data/TUAB/tuab_disease.npz",
    "TUEP": "/root/autodl-tmp/data/TUEP/tuep_disease.npz",
}


def bandpower_features(ts, hz):
    """(N, C, T) -> (N, C * n_bands) 每通道频带功率 (相对总功率)。"""
    n, c, t = ts.shape
    spec = np.abs(np.fft.rfft(ts, axis=2)) ** 2          # (N, C, T//2+1)
    freqs = np.fft.rfftfreq(t, d=1.0 / hz)
    total = spec.sum(axis=2, keepdims=True) + 1e-12
    feats = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        feats.append(spec[:, :, m].sum(axis=2) / total[:, :, 0])
    return np.concatenate(feats, axis=1)                 # (N, C*5)


def probe(path, seed=42):
    d = np.load(path, allow_pickle=True)
    ts, lab = d["timeseries"], d["labels"].astype(int)
    hz = int(d["hz"])
    tr = d["split_train_index"].astype(int)
    va = d["split_val_index"].astype(int)
    te = d["split_test_index"].astype(int)
    X = bandpower_features(ts, hz)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    X = (X - mu) / sd
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(X[tr], lab[tr])
    out = {
        "val_auc": roc_auc_score(lab[va], clf.predict_proba(X[va])[:, 1]),
        "test_auc": roc_auc_score(lab[te], clf.predict_proba(X[te])[:, 1]),
        "train_auc": roc_auc_score(lab[tr], clf.predict_proba(X[tr])[:, 1]),
    }
    # 标签随机化对照 (3 次)
    rng = np.random.default_rng(seed)
    shuffle_aucs = []
    for _ in range(3):
        lab_s = rng.permutation(lab[tr])
        clf_s = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        clf_s.fit(X[tr], lab_s)
        shuffle_aucs.append(roc_auc_score(lab[te], clf_s.predict_proba(X[te])[:, 1]))
    out["shuffled_test_auc"] = float(np.mean(shuffle_aucs))
    return out


if __name__ == "__main__":
    print(f"{'dataset':<10}{'train_auc':>12}{'val_auc':>10}{'test_auc':>10}"
          f"{'shuffled_test_auc':>19}")
    print("-" * 61)
    for name, path in FILES.items():
        r = probe(path)
        print(f"{name:<10}{r['train_auc']:>12.4f}{r['val_auc']:>10.4f}"
              f"{r['test_auc']:>10.4f}{r['shuffled_test_auc']:>19.4f}")
