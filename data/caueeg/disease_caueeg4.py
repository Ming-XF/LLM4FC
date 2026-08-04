import os
from random import shuffle
import mne
import numpy as np
import torch
import torch.nn.functional as F

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import *

import json
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class DiseaseCAUEEG4Dataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(DiseaseCAUEEG4Dataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                    episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        raw = np.load(self.data_config.data_dir, allow_pickle=True)
        data = dict(raw) if hasattr(raw, 'files') else raw.item()  # 兼容 .npz / .npy
        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = data["hz"]

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.output_dim = 4
        self.data_config.task_type = DataConfig.TASK_CLASSIFICATION

        self.data_config.class_weight = [1, 1, 1, 1]
        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        self._create_splits(labels, self.all_data['subject_id'])
        self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(self.all_data['time_series'][idx[item]]).float()
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).to(torch.int64)

        SFC = self.connectivity(time_series)
        window_size = 6 * self.hz
        step_size = (60 * self.hz - window_size) // 9
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)

        return {'time_series': time_series,
                'DFC': DFC,
                'correlation': SFC,  # SFC, for backward compat
                'labels': labels,
                'sample_idx': idx[item]}



def _process_one_sample(args):
    """处理单个 EDF 样本，供 multiprocessing worker 调用（模块级函数以支持 pickle）"""
    sample, signal_folder, hz = args
    serial = sample['serial']
    subject_id = int(re.findall(r'\d+', serial)[0])
    edf_path = os.path.join(signal_folder, f"{serial}.edf")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, times = raw[:, :]
    data = data[:19, :]

    if data.shape[1] % (hz * 60) != 0:
        data = data[:, :-(data.shape[1] % (hz * 60))]
    data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
    data = np.transpose(data, (1, 0, 2))

    if 'ad' in sample['symptom']:
        label = np.full(data.shape[0], 3)
    elif 'mci_amnestic' in sample['symptom']:
        label = np.full(data.shape[0], 2)
    elif 'smi' in sample['symptom']:
        label = np.full(data.shape[0], 1)
    else:
        label = np.full(data.shape[0], 0)

    subj_ids = np.full(data.shape[0], subject_id)

    return data, label, subj_ids


def disease_caueeg4_preprocess(path="../data/CAUEEG/", hz=200):
    # 配置路径
    annotation_file = os.path.join(path, "caueeg-dataset/annotation.json")
    signal_folder = os.path.join(path, os.path.join('caueeg-dataset/signal', "edf"))
    output_path = os.path.join(path, "caueeg4_disease.npz")

    # 读取标注文件
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)

    # 筛选目标样本
    target_samples = [s for s in annotation['data']
                      if 'ad' in s['symptom'] or 'cb_normal' in s['symptom']
                      or 'smi' in s['symptom'] or 'mci_amnestic' in s['symptom']]

    n_workers = min(cpu_count(), len(target_samples), 8)

    print(f"并行处理 {len(target_samples)} 个样本，使用 {n_workers} 个进程...")

    # 构建参数列表
    task_args = [(sample, signal_folder, hz) for sample in target_samples]

    # 多进程并行处理
    ts_list, lbl_list, subj_list = [], [], []
    with Pool(processes=n_workers) as pool:
        for data, label, subj_ids in tqdm(pool.imap_unordered(_process_one_sample, task_args),
                                          total=len(task_args), desc="加载EDF"):
            ts_list.append(data)
            lbl_list.append(label)
            subj_list.append(subj_ids)

    # 一次性拼接，避免循环中 np.append 的 O(n²) 开销
    time_series = np.concatenate(ts_list, axis=0)
    labels = np.concatenate(lbl_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)

    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)

    time_series = time_series.astype(np.float32)
    labels = labels.astype(np.int8)

    print(time_series.shape)
    np.savez(output_path, timeseries=time_series, labels=labels, subject_id=subject_ids, hz=hz)


if __name__ == '__main__':
    disease_caueeg4_preprocess('../data/CAUEEG/', hz=200)
