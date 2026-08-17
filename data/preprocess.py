import numpy as np
from scipy.linalg import sqrtm


# ── 年龄回归的固定先验归一化边界 ──
# 不从数据拟合 min/max（避免 test 泄漏），且所有域统一量纲，
# 便于 few-shot / 跨域迁移时 head 输出量纲一致。
AGE_LABEL_MIN = 0.0
AGE_LABEL_MAX = 100.0


def data_norm(data):
    data_copy = np.copy(data)
    for i in range(len(data)):
        data_copy[i] = data_copy[i] / np.maximum(np.max(abs(data[i])), 1e-8)

    return data_copy


def preprocess_ea(data):
    R_bar = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(data)):
        R_bar += np.dot(data[i], data[i].T)
    R_bar_mean = R_bar / len(data)
    R_bar_mean += 1e-6 * np.eye(R_bar_mean.shape[0])

    for i in range(len(data)):
        data[i] = np.dot(np.linalg.inv(sqrtm(R_bar_mean)), data[i])
    return data
