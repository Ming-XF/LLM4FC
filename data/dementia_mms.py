import os
from random import shuffle, randrange
import mne
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy import signal
from nilearn import connectome

from .data_config import DataConfig
from .dataset import BaseDataset
from .preprocess import *

import mat73
import scipy.io as sio
import h5py
import re

import pandas
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pdb


class Dementia_MMSDataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(Dementia_MMSDataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

    def load_data(self, one_hot=True):
        raw = np.load(self.data_config.data_dir, allow_pickle=True)
        data = dict(raw) if hasattr(raw, 'files') else raw.item()  # 兼容 .npz / .npy
        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        m_input = data["m_input"]
        m_label = data["m_label"]
        self.hz = data["hz"]
        # spects = data['spects']

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 4

        self.data_config.class_weight = [1, 1, 1, 1]
        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id
        self.all_data['gender'] = m_input[:, 0]
        self.all_data['age'] = m_input[:, 1]
        self.all_data['education'] = m_input[:, 2]
        self.all_data['m_label'] = m_label
        # self.all_data['spects'] = spects
        if 'dfc' in data:
            self.all_data['dfc'] = data['dfc']

        if self.subject_id:
            self.select_subject()
        self._create_splits(labels, self.all_data['subject_id'])
        self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()
        self.all_data['gender'] = F.one_hot(torch.from_numpy(self.all_data['gender'] - 1).to(torch.int64)).numpy()
        self.all_data['education'] = F.one_hot(torch.from_numpy(self.all_data['education'] - 1).to(torch.int64)).numpy()
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(self.all_data['time_series'][idx[item]]).float()
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).to(torch.int64)
        gender = torch.from_numpy(self.all_data['gender'][idx[item]]).to(torch.int64)
        age = torch.tensor(self.all_data['age'][idx[item]]).float()
        education = torch.from_numpy(self.all_data['education'][idx[item]]).to(torch.int64)
        m_label = torch.from_numpy(self.all_data['m_label'][idx[item]]).float()
        # spects = torch.from_numpy(self.all_data['spects'][idx[item]]).float()

        sampling_init = (randrange(time_series.size(-1) - self.data_config.time_series_size)) \
            if self.data_config.dynamic else 0
        time_series = time_series[:, sampling_init:sampling_init + self.data_config.time_series_size]
        SFC = self.connectivity(time_series, activate=False)
        if not self.data_config.dynamic and 'dfc' in self.all_data:
            DFC = torch.from_numpy(self.all_data['dfc'][idx[item]]).float()
        else:
            DFC = dynamic_connectivity(time_series.numpy(), 3 * self.hz, 1 * self.hz)
            DFC = torch.from_numpy(DFC).float()

        return {'time_series': time_series,
                'DFC': DFC,
                'correlation': SFC,  # SFC, for backward compat
                'labels': labels,
                'gender': gender,
                'age': age,
                'education': education,
                'm_label': m_label}
                # 'spects': spects}

    def select_subject(self):
        self.selected = [self.subject_id]
        index = np.sum(self.all_data["subject_id"] == i for i in self.selected) == 1
        self.all_data['time_series'] = self.all_data['time_series'][index]
        self.all_data['labels'] = self.all_data['labels'][index]
        self.all_data['subject_id'] = self.all_data['subject_id'][index]
        self.all_data['gender'] = self.all_data['gender'][index]
        self.all_data['age'] = self.all_data['age'][index]
        self.all_data['education'] = self.all_data['education'][index]
        self.all_data['m_label'] = self.all_data['m_label'][index]
        if 'dfc' in self.all_data:
            self.all_data['dfc'] = self.all_data['dfc'][index]
        # self.all_data['spects'] = self.all_data['spects'][index]


def _compute_dfc(args):
    """计算单个 segment 的 DFC，供 multiprocessing worker 调用"""
    segment, hz = args
    return dynamic_connectivity(segment, 3 * hz, 1 * hz)


def dementia_preprocess(path="../data/Dementia_MMS", hz=250):
    mms_path = path + "/MMS.txt"
    df = pd.read_csv(mms_path, sep='\t')
    nor = DementiaDataNormalizer()
    df = nor.normalize(df)

    # pdb.set_trace()

    time_series = labels = subject_ids = tags = m_input = m_label = spects = None

    Nor_path = os.path.join(path, "Normal")
    for filename in os.listdir(Nor_path):
        subject_id = int(re.findall(r'\d+', filename)[0])
        mm = df[df['受试编号'] == "sub" + str(subject_id)]
        
        mat = sio.loadmat(os.path.join(Nor_path, filename))
        data = mat['Value']
        if data.shape[1] % (hz * 60) != 0:
            data = data[:, :-(data.shape[1] % (hz * 60))]
        data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
        data = np.transpose(data, (1, 0, 2))

        m1 = mm.values[:, 1:4].astype(np.float32).repeat(data.shape[0], axis=0)
        m2 = mm.values[:, 5:].astype(np.float32).repeat(data.shape[0], axis=0)

        label = np.full(data.shape[0], 0)

        time_series = data if time_series is None else np.append(time_series, data, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
        subject_ids = np.ones(label.shape[0]) * subject_id if subject_ids is None \
            else np.append(subject_ids, np.ones(label.shape[0]) * subject_id, axis=0)
        m_input = m1 if m_input is None else np.append(m_input, m1, axis=0)
        m_label = m2 if m_label is None else np.append(m_label, m2, axis=0)
        # spects = spect if spects is None else np.append(spects, spect, axis=0)


    DSC_path = os.path.join(path, "DSC")
    for filename in os.listdir(DSC_path):
        subject_id = int(re.findall(r'\d+', filename)[0])
        mm = df[df['受试编号'] == "sub" + str(subject_id)]
        
        mat = sio.loadmat(os.path.join(DSC_path, filename))
        data = mat['Value']
        if data.shape[1] % (hz * 60) != 0:
            data = data[:, :-(data.shape[1] % (hz * 60))]
        data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
        data = np.transpose(data, (1, 0, 2))

        m1 = mm.values[:, 1:4].astype(np.float32).repeat(data.shape[0], axis=0)
        m2 = mm.values[:, 5:].astype(np.float32).repeat(data.shape[0], axis=0)

        label = np.full(data.shape[0], 1)

        time_series = data if time_series is None else np.append(time_series, data, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
        subject_ids = np.ones(label.shape[0]) * subject_id if subject_ids is None \
            else np.append(subject_ids, np.ones(label.shape[0]) * subject_id, axis=0)
        m_input = m1 if m_input is None else np.append(m_input, m1, axis=0)
        m_label = m2 if m_label is None else np.append(m_label, m2, axis=0)

    MCI_path = os.path.join(path, "MCI")
    for filename in os.listdir(MCI_path):
        subject_id = int(re.findall(r'\d+', filename)[0])
        mm = df[df['受试编号'] == "sub" + str(subject_id)]
        
        mat = sio.loadmat(os.path.join(MCI_path, filename))
        data = mat['Value']
        if data.shape[1] % (hz * 60) != 0:
            data = data[:, :-(data.shape[1] % (hz * 60))]
        data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
        data = np.transpose(data, (1, 0, 2))

        m1 = mm.values[:, 1:4].astype(np.float32).repeat(data.shape[0], axis=0)
        m2 = mm.values[:, 5:].astype(np.float32).repeat(data.shape[0], axis=0)

        label = np.full(data.shape[0], 2)

        time_series = data if time_series is None else np.append(time_series, data, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
        subject_ids = np.ones(label.shape[0]) * subject_id if subject_ids is None \
            else np.append(subject_ids, np.ones(label.shape[0]) * subject_id, axis=0)
        m_input = m1 if m_input is None else np.append(m_input, m1, axis=0)
        m_label = m2 if m_label is None else np.append(m_label, m2, axis=0)
    
    

    AD_path = os.path.join(path, "AD")
    for filename in os.listdir(AD_path):
        subject_id = int(re.findall(r'\d+', filename)[0])
        mm = df[df['受试编号'] == "sub" + str(subject_id)]

        mat = sio.loadmat(os.path.join(AD_path, filename))
        data = mat['Value']
        if data.shape[1] % (hz * 60) != 0:
            data = data[:, :-(data.shape[1] % (hz * 60))]
        data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
        data = np.transpose(data, (1, 0, 2))

        m1 = mm.values[:, 1:4].astype(np.float32).repeat(data.shape[0], axis=0)
        m2 = mm.values[:, 5:].astype(np.float32).repeat(data.shape[0], axis=0)

        label = np.full(data.shape[0], 3)

        time_series = data if time_series is None else np.append(time_series, data, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
        subject_ids = np.ones(label.shape[0]) * subject_id if subject_ids is None \
            else np.append(subject_ids, np.ones(label.shape[0]) * subject_id, axis=0)
        m_input = m1 if m_input is None else np.append(m_input, m1, axis=0)
        m_label = m2 if m_label is None else np.append(m_label, m2, axis=0)
        # spects = spect if spects is None else np.append(spects, spect, axis=0)

    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)

    n_workers = min(cpu_count(), len(time_series), 8)
    print(f"Precomputing DFC ({len(time_series)} segments, {n_workers} workers)...")
    dfc_task_args = [(time_series[i], hz) for i in range(len(time_series))]
    with Pool(processes=n_workers) as pool:
        dfc_list = list(tqdm(pool.imap_unordered(_compute_dfc, dfc_task_args),
                             total=len(dfc_task_args), desc="DFC"))
    dfc = np.stack(dfc_list, axis=0)

    # pdb.set_trace()

    print(time_series.shape, dfc.shape)
    np.savez(os.path.join(path, f"Dementia_MMS.npz"),
             timeseries=time_series,
             labels=labels,
             subject_id=subject_ids,
             m_input=m_input,
             m_label=m_label,
             hz=hz,
             dfc=dfc)


        
def dynamic_connectivity(time_series, window_size, step_size, activate=False, use_oas=False):
    """
    计算动态脑功能连接图
    
    Parameters:
    -----------
    time_series : array-like, shape (N, L)
        EEG数据，N为通道个数，L为采样长度
    window_size : int
        滑动窗口大小
    step_size : int
        滑动步长
    activate : bool
        是否应用arctanh变换和去对角线操作
    use_oas : bool
        是否使用OAS协方差估计器
    
    Returns:
    --------
    dynamic_conn : torch.Tensor, shape (num_windows, N, N)
        动态功能连接矩阵
    window_indices : list
        每个窗口的起始和结束索引
    """
    
    # 转换为numpy数组（如果需要）
    # if torch.is_tensor(time_series):
    #     time_series = time_series.numpy()
    
    N, L = time_series.shape
    
    # 计算滑动窗口数量
    num_windows = (L - window_size) // step_size + 1
    
    # 初始化连接图估计器
    if use_oas:
        conn_measure = connectome.ConnectivityMeasure(
            kind='correlation', 
            cov_estimator=OAS(store_precision=False)
        )
    else:
        conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    
    # 存储所有窗口的连接矩阵
    dynamic_conn = []
    window_indices = []
    
    for i in range(num_windows):
        # 计算当前窗口的起始和结束索引
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # 提取当前窗口的数据
        window_data = time_series[:, start_idx:end_idx]
        
        # 计算功能连接
        # 需要将数据reshape为 (n_samples, n_features) 格式
        # 这里n_samples是时间点，n_features是通道数
        window_data_reshaped = window_data.T  # shape: (window_size, N)
        
        # 添加一个维度作为样本维度 (n_subjects, n_samples, n_features)
        # 这里我们只有一个"被试"，所以添加一个维度
        data_for_connectivity = window_data_reshaped[np.newaxis, :, :]
        
        # 计算连接矩阵；窗口内通道方差为 0 时 nilearn 内部会触发
        # divide-by-zero → NaN，用 errstate 抑制 RuntimeWarning，
        # 随后 np.nan_to_num 将结果干净填充为 0
        with np.errstate(divide='ignore', invalid='ignore'):
            conn_matrix = conn_measure.fit_transform(data_for_connectivity)[0]
        conn_matrix = np.nan_to_num(conn_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为torch张量
        # conn_matrix = torch.from_numpy(conn_matrix).float()
        
        # 应用后处理（如果需要）
        if activate:
            conn_matrix = np.arctanh(conn_matrix)
            conn_matrix = np.clip(conn_matrix, -1.0, 1.0)
            # 将对角线设为0
            np.fill_diagonal(conn_matrix, 0)
        
        dynamic_conn.append(conn_matrix)
        # window_indices.append((start_idx, end_idx))
    
    # 堆叠所有窗口的连接矩阵
    dynamic_conn = np.stack(dynamic_conn, axis=0)
    
    # print(dynamic_conn.shape)
    
    # return dynamic_conn, window_indices
    return dynamic_conn


if __name__ == '__main__':
    dementia_preprocess("../data/Dementia_MMS", hz=250)
