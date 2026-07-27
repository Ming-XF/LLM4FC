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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 或者抑制所有警告
warnings.filterwarnings("ignore")

import pdb


class BeirutDataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(BeirutDataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

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
        self.train_index, self.test_index = list(self.k_fold.split(self.all_data['time_series'], labels, groups))[self.k]
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


def beirut_preprocess(path_NC="../data/Dementia4000/", NC_num=300, path_beirut="../data/Beirut", AB_num=-1, hz=200):
    time_series = pearson = labels = subject_ids = tags = None
    output_path = os.path.join(path_beirut, "Beirut.npy")

     # 配置路径
    annotation_file = os.path.join(path_NC, "caueeg-dataset/annotation.json")
    signal_folder = os.path.join(path_NC, os.path.join('caueeg-dataset/signal', "edf"))
    
    # 读取标注文件
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)

    # 筛选目标样本
    target_samples = [s for s in annotation['data'] if 'cb_normal' in s['symptom']][:NC_num]
    
    # 处理每个样本
    for sample in tqdm(target_samples):
        serial = sample['serial']
        subject_id = int(re.findall(r'\d+', serial)[0])
        edf_path = os.path.join(signal_folder, f"{serial}.edf")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        # 通道名称： ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'O1-AVG', 'Fp2-AVG', 'F4-AVG', 'C4-AVG', 'P4-AVG', 'O2-AVG', 'F7-AVG', 'T3-AVG', 'T5-AVG', 'F8-AVG', 'T4-AVG', 'T6-AVG', 'FZ-AVG', 'CZ-AVG', 'PZ-AVG', 'EKG', 'Photic']
        data, times = raw[:, :]
        data = data[:19, :]

        if data.shape[1] % (hz * 60) != 0:
            data = data[:, :-(data.shape[1] % (hz * 60))]
        data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
        data = np.transpose(data, (1, 0, 2))

        corr = np.array([np.corrcoef(t) for t in data])

        label = np.full(data.shape[0], 0)

        time_series = data if time_series is None else np.append(time_series, data, axis=0)
        pearson = corr if pearson is None else np.append(pearson, corr, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
        subject_ids = np.ones(label.shape[0]) * subject_id if subject_ids is None \
            else np.append(subject_ids, np.ones(label.shape[0]) * subject_id, axis=0)

    print(time_series.shape)

    for patient in tqdm([15, 14, 13, 12, 11]):
        if patient in (15, 13, 11):
            files_list=["Record1.edf","Record2.edf","Record3.edf","Record4.edf"]
        elif patient in (14, 12):
            files_list=["Record1.edf","Record2.edf","Record3.edf"]

        for file_id, file in enumerate(files_list):
            file=os.path.join(os.path.join(path_beirut, "Raw_EDF_Files"), "p"+str(patient)+"_"+file)   
            subject_id = patient
            raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
           
            raw.reorder_channels(['EEG Fp1-Ref', 'EEG F3-Ref', 'EEG C3-Ref', 'EEG P3-Ref', 'EEG O1-Ref', 'EEG Fp2-Ref', 'EEG F4-Ref', 'EEG C4-Ref', 'EEG P4-Ref', 'EEG O2-Ref', 'EEG F7-Ref', 'EEG T3-Ref', 'EEG T5-Ref', 'EEG F8-Ref', 'EEG T4-Ref', 'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref', 'EEG A2-Ref', 'EEG A1-Ref', 'ECG EKG', 'Manual'])
            raw.set_eeg_reference('average', verbose=False)
            raw = raw.copy().resample(sfreq=hz, verbose=False)

            data, times = raw[:, :]
            data = data[:19, :]

            if data.shape[1] % (hz * 60) != 0:
                data = data[:, :-(data.shape[1] % (hz * 60))]
            data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
            data = np.transpose(data, (1, 0, 2))[:AB_num]
    
            corr = np.array([np.corrcoef(t) for t in data])
    
            label = np.full(data.shape[0], 1)
    
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
    beirut_preprocess(path_NC="../data/Dementia4000/", NC_num=40, path_beirut="../data/Beirut", AB_num=20, hz=200)
