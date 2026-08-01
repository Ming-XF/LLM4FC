import numpy as np
import torch
from scipy.linalg import sqrtm


def continues_mixup_data(*xs, y1=None, y2=None, alpha=1.0, beta=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1
    batch_size = y1.size()[0]
    index = torch.randperm(batch_size)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y1 = lam * y1 + (1-lam) * y1[index]
    y2 = lam * y2 + (1-lam) * y2[index] if y2 is not None else y2
    return *new_xs, y1, y2


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
