import numpy as np
from scipy.linalg import sqrtm


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
