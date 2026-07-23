import os
from random import shuffle, randrange
import mne
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from .data_config import DataConfig
from .dataset import BaseDataset
from .preprocess import *

import json
import re
from tqdm import tqdm

import pdb


class Dementia2000Dataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(Dementia2000Dataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

    def load_data(self, one_hot=True):
        data = np.load(self.data_config.data_dir, allow_pickle=True).item()
        time_series = data["timeseries"]
        correlation = data["corr"]
        labels = data["labels"]
        subject_id = data["subject_id"]

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 2

        self.data_config.class_weight = [1, 1]
        self.all_data['time_series'] = time_series
        self.all_data['correlation'] = correlation
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        if self.subject_id:
            self.select_subject()
        groups = np.array([f"{int(s)}_{int(l)}" for s, l in zip(self.all_data['subject_id'], labels)])
        self.train_index, self.test_index = list(self.k_fold.split(self.all_data['time_series'], groups))[self.k]
        self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self.train_index if self.train else self.test_index
        time_series = torch.from_numpy(self.all_data['time_series'][idx[item]]).float()
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).to(torch.int64)

        sampling_init = (randrange(time_series.size(-1) - self.data_config.time_series_size)) \
            if self.data_config.dynamic else 0
        time_series = time_series[:, sampling_init:sampling_init + self.data_config.time_series_size]
        correlation = self.connectivity(time_series, activate=False)

        return {'time_series': time_series,
                'correlation': correlation,
                'labels': labels}

    def select_subject(self):
        self.selected = [self.subject_id]
        index = np.sum(self.all_data["subject_id"] == i for i in self.selected) == 1
        self.all_data['time_series'] = self.all_data['time_series'][index]
        self.all_data['correlation'] = self.all_data['correlation'][index]
        self.all_data['labels'] = self.all_data['labels'][index]
        self.all_data['subject_id'] = self.all_data['subject_id'][index]
        # self.all_data['tags'] = self.all_data['tags'][index]


def dementia_preprocess(path="../data/Dementia2000/", hz=200):
    time_series = pearson = labels = subject_ids = tags = None
    
     # 配置路径
    annotation_file = os.path.join(path, "caueeg-dataset/annotation.json")
    signal_folder = os.path.join(path, os.path.join('caueeg-dataset/signal', "edf"))
    output_path = os.path.join(path, "Dementia2000.npy")
    
    # 读取标注文件
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)

    # 筛选目标样本
    target_samples = [s for s in annotation['data'] if 'ad' in s['symptom'] or 'cb_normal' in s['symptom']]
    
    # 处理每个样本
    for sample in tqdm(target_samples):
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

        corr = np.array([np.corrcoef(t) for t in data])

        label = np.ones(data.shape[0]) if 'ad' in sample['symptom'] else np.zeros(data.shape[0])

        time_series = data if time_series is None else np.append(time_series, data, axis=0)
        pearson = corr if pearson is None else np.append(pearson, corr, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
        subject_ids = np.ones(label.shape[0]) * subject_id if subject_ids is None \
            else np.append(subject_ids, np.ones(label.shape[0]) * subject_id, axis=0)

    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)

    print(time_series.shape)
    np.save(output_path, {"timeseries": time_series, "corr": pearson, "labels": labels, "subject_id": subject_ids})


if __name__ == '__main__':
    dementia_preprocess("../data/Dementia2000", hz=200)
