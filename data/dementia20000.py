import numpy as np
from scipy.spatial.distance import cdist
from random import shuffle, randrange
import torch
import torch.nn.functional as F

import scipy.io as sio
import scipy.signal as signal
import mne
import os
import re
from tqdm import tqdm

from .preprocess import *
from .data_config import DataConfig
from .dataset import BaseDataset

import pdb

class Dementia20000Dataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(Dementia20000Dataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

    def load_data(self, one_hot=True):
        data1 = np.load(self.data_config.data_dir, allow_pickle=True).item()
        data2 = np.load(self.data_config.data2_dir, allow_pickle=True).item()

        data = {}
        for key in data1.keys():
            data[key] = np.concatenate([data1[key], data2[key]], axis=0)
        # data = np.concatenate([data1, data2], axis=0)
        
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
        # groups = np.array([f"{int(s)}_{int(l)}" for s, l in zip(self.all_data['subject_id'], labels)])
        # self.train_index, self.test_index = list(self.k_fold.split(self.all_data['time_series'], groups))[self.k]
        self.train_index = np.arange(len(data1["timeseries"]))
        self.test_index = np.arange(len(data2["timeseries"])) + len(data1["timeseries"])

        # train/val/test mode: further split train portion by subject
        if self.k_fold is None:
            from sklearn.model_selection import train_test_split
            train_subj_ids = self.all_data['subject_id'][self.train_index]
            train_labels_raw = labels[self.train_index]
            unique_subjs = np.unique(train_subj_ids)
            subj_labels = np.array([
                np.bincount(train_labels_raw[train_subj_ids == s].astype(int)).argmax()
                for s in unique_subjs
            ])
            val_ratio = self.data_config.val_set / self.data_config.train_set
            train_subjs, val_subjs = train_test_split(
                unique_subjs, test_size=val_ratio,
                stratify=subj_labels, random_state=42 + self.k)
            train_mask = np.isin(train_subj_ids, train_subjs)
            val_mask = np.isin(train_subj_ids, val_subjs)
            self.val_index = self.train_index[val_mask]
            self.train_index = self.train_index[train_mask]
        self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
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

# 示例数据 - 请替换为你的实际数据
electrode_10_20 = {
    "Fp1": [-21.5, 70.2, -0.1],
    "F3": [-35.5, 49.4, 32.4],
    "C3": [-52.2, -16.4, 57.8],
    "P3": [-39.5, -76.3, 47.4],
    "O1": [-26.8, -100.2, 12.8],
    "Fp2": [28.4, 69.1, -0.4],
    "F4": [40.2, 47.6, 32.1],
    "C4": [54.1, -18, 57.5],
    "P4": [36.8, -74.9, 49.2],
    "O2": [24.1, -100.5, 14.1],
    "F7": [-54.8, 33.9, -3.5],
    "T3": [-70.2, -21.3, -10.7],
    "T5": [-61.5, -65.3, 1.1],
    "F8": [56.6, 30.8, -4.1],
    "T4": [71.9, -25.2, -8.2],
    "T6": [59.3, -67.6, 3.7],
    "Fz": [0.6, 40.9, 53.9],
    "Cz": [0.8, -14.7, 73.9],
    "Pz": [0.2, -62.1, 64.5],
}

dk_parcellation = {
    'l.bankssts': [-54.343785, -44.539029, 4.163784],
    'r.bankssts': [52.975610, -40.553816, 5.303675],
    'l.caudalanteriorcingulate': [-5.030493, 20.087970, 28.999343],
    'r.caudalanteriorcingulate': [5.012041, 22.258100, 27.639678],
    'l.caudalmiddlefrontal': [-35.521824, 10.809538, 44.190969],
    'r.caudalmiddlefrontal': [35.661664, 12.293548, 44.471424],
    'l.cuneus': [-7.126394, -79.633054, 18.510034],
    'r.cuneus': [7.165208, -80.094411, 19.161360],
    'l.entorhinal': [-22.998997, -7.877750, -35.210045],
    'r.entorhinal': [22.762408, -7.619649, -34.077827],
    'l.frontalpole': [-6.785115, 64.865577, -11.499812],
    'r.frontalpole': [8.694854, 64.422083, -11.920596],
    'l.fusiform': [-35.141396, -43.474374, -22.012104],
    'r.fusiform': [35.323007, -43.239303, -21.599235],
    'l.inferiorparietal': [-40.934987, -67.711257, 28.124042],
    'r.inferiorparietal': [44.345690, -61.781892, 28.633137],
    'l.inferiortemporal': [-49.601517, -34.703196, -25.261257],
    'r.inferiortemporal': [50.781852, -31.732182, -26.194109],
    'l.insula': [-37.137130, -3.504331, 1.690092],
    'r.insula': [38.249302, -3.015601, 1.544895],
    'l.isthmuscingulate': [-6.624240, -47.248045, 16.969356],
    'r.isthmuscingulate': [7.091746, -46.163380, 16.740261],
    'l.lateraloccipital': [-30.075425, -88.498214, -1.523382],
    'r.lateraloccipital': [31.117596, -87.943778, -0.457613],
    'l.lateralorbitofrontal': [-24.788431, 28.715777, -16.968762],
    'r.lateralorbitofrontal': [24.236422, 29.349355, -17.996568],
    'l.lingual': [-14.506283, -67.608044, -5.063883],
    'r.lingual': [14.746014, -66.769025, -4.325692],
    'l.medialorbitofrontal': [-5.406928, 36.933371, -18.001864],
    'r.medialorbitofrontal': [5.859795, 37.568028, -16.583859],
    'l.middletemporal': [-57.751067, -30.223900, -13.290262],
    'r.middletemporal': [58.170481, -27.920889, -13.563721],
    'l.paracentral': [-7.897553, -29.735224, 56.120347],
    'r.paracentral': [7.827022, -28.582297, 55.544456],
    'l.parahippocampal': [-23.907604, -33.142327, -19.249481],
    'r.parahippocampal': [25.382203, -33.021476, -18.144719],
    'l.parsopercularis': [-45.746763, 14.555066, 11.852166],
    'r.parsopercularis': [46.526088, 14.243186, 13.378662],
    'l.parsorbitalis': [-42.510831, 38.551743, -14.105952],
    'r.parsorbitalis': [44.123201, 39.162297, -11.985981],
    'l.parstriangularis': [-44.017232, 30.266616, 0.805347],
    'r.parstriangularis': [46.623897, 29.459390, 3.342318],
    'l.pericalcarine': [-11.765939, -81.490718, 5.367253],
    'r.pericalcarine': [12.515956, -80.239169, 6.009491],
    'l.postcentral': [-43.230818, -23.574863, 43.947822],
    'r.postcentral': [42.365002, -22.476359, 44.557213],
    'l.posteriorcingulate': [-5.701530, -18.390072, 38.473745],
    'r.posteriorcingulate': [5.685813, -17.196104, 38.859022],
    'l.precentral': [-38.791732, -10.412277, 42.973217],
    'r.precentral': [37.897994, -9.813155, 44.547376],
    'l.precuneus': [-9.690527, -58.233298, 36.662633],
    'r.precuneus': [9.636033, -57.310060, 37.845502],
    'l.rostralanteriorcingulate': [-4.385862, 37.523613, -0.212297],
    'r.rostralanteriorcingulate': [5.367839, 37.109173, 1.676762],
    'l.rostralmiddlefrontal': [-33.210735, 42.715400, 16.838418],
    'r.rostralmiddlefrontal': [33.965750, 42.836306, 17.683447],
    'l.superiorfrontal': [-11.379796, 24.090999, 43.374715],
    'r.superiorfrontal': [12.193157, 25.698054, 43.075713],
    'l.superiorparietal': [-23.409777, -61.788495, 47.827327],
    'r.superiorparietal': [23.192057, -60.478712, 49.691973],
    'l.superiortemporal': [-53.401526, -15.660875, -4.006122],
    'r.superiortemporal': [54.388844, -12.260566, -5.118477],
    'l.supramarginal': [-52.060730, -39.127553, 31.482408],
    'r.supramarginal': [52.104994, -33.127594, 31.196859],
    'l.temporalpole': [-29.311423, 12.899209, -38.046768],
    'r.temporalpole': [30.573320, 13.776967, -35.773082],
    'l.transversetemporal': [-44.474530, -22.679342, 7.332593],
    'r.transversetemporal': [44.650935, -20.795417, 8.168757],
}

def find_closest_dk_region(electrode_coords, dk_coords):
    """
    为每个电极找到最近的DK脑区
    
    参数:
    electrode_coords: dict, 电极名称->坐标
    dk_coords: dict, DK脑区名称->坐标
    
    返回:
    dict: 电极名称->(最近的DK脑区名称, 距离)
    """
    # 转换为numpy数组以便计算
    elec_names = list(electrode_coords.keys())
    dk_names = list(dk_coords.keys())
    
    elec_points = np.array([electrode_coords[name] for name in elec_names])
    dk_points = np.array([dk_coords[name] for name in dk_names])
    
    # 计算所有点对之间的距离矩阵
    # 形状: (电极数, DK脑区数)
    distances = cdist(elec_points, dk_points, metric='euclidean')
    
    # 找到每个电极最近的DK脑区索引
    closest_indices = np.argmin(distances, axis=1)
    
    # 构建结果字典
    result = {}
    for i, elec_name in enumerate(elec_names):
        closest_dk_idx = closest_indices[i]
        closest_dk_name = dk_names[closest_dk_idx]
        min_distance = distances[i, closest_dk_idx]
        result[elec_name] = (closest_dk_name, min_distance)
    
    return result

def transform_a_to_b(dk_names, b_electrodes, match_result):
    """
    将数组a根据匹配结果转换为数组b的形状
    
    参数:
    dk_names: list, 数组a对应的DK脑区名称，顺序与a的通道对应
    b_electrodes: list, 数组b对应的电极名称，顺序与b的通道对应
    match_result: dict, 电极到DK脑区的匹配结果
    
    返回:
    b_transformed: numpy数组, 形状为(19, N)
    """
    # 获取DK脑区名称到索引的映射
    dk_name_to_idx = {name: idx for idx, name in enumerate(dk_names)}
    
    # 对于每个电极，找到对应的DK脑区并复制数据
    result = []
    for i, elec_name in enumerate(b_electrodes):
        if elec_name in match_result:
            closest_dk_name, distance = match_result[elec_name]
            
            # 如果匹配的DK脑区在数组a中
            if closest_dk_name in dk_name_to_idx:
                dk_idx = dk_name_to_idx[closest_dk_name]
                result.append(dk_idx)
                print(f"电极 {elec_name} -> DK脑区 {closest_dk_name} (距离: {distance:.2f})")
            else:
                print(f"警告: 电极 {elec_name} 匹配的脑区 {closest_dk_name} 不在数组a中")
        else:
            print(f"警告: 电极 {elec_name} 没有匹配结果")
    
    return result

def resample(eeg_data):
    """
    将250Hz数据重采样到200Hz
    """

    # 200/250 = 4/5
    resampled_data_poly = signal.resample_poly(data, 4, 5, axis=-1)
    
    return resampled_data_poly


if __name__ == '__main__':
    # pdb.set_trace()

    hz = 200
    path = "../data/Dementia200"
    
    match_result = find_closest_dk_region(electrode_10_20, dk_parcellation)

    # # edf_path = "../data/Dementia4000/caueeg-dataset/signal/edf/00007.edf"
    # # raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    # # pdb.set_trace()
    # # data, times = raw[:, :]
    # # data = data[:19, :]

    
    # # mat = sio.loadmat("../data/Dementia400/channel.mat")
    # mat = sio.loadmat("../data/Dementia400/AD/sub4_source_localized.mat")
    # data = mat['Value']


    dk_names = list(dk_parcellation.keys())
    b_electrodes = list(electrode_10_20.keys())
    indexs = transform_a_to_b(dk_names, b_electrodes, match_result)

    # b = data[indexs, :]

    
    time_series = pearson = labels = subject_ids = tags = None
     # 配置路径
    AD_path = os.path.join(path, "AD")
    for filename in os.listdir(AD_path):
        subject_id = int(re.findall(r'\d+', filename)[0])
        mat = sio.loadmat(os.path.join(AD_path, filename))

        data = mat['Value']
        data = resample(data)
        data = data[indexs, :]

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
    
    
    Nor_path = os.path.join(path, "Normal")
    for filename in os.listdir(Nor_path):
        subject_id = int(re.findall(r'\d+', filename)[0])
        mat = sio.loadmat(os.path.join(Nor_path, filename))

        data = mat['Value']
        data = resample(data)
        data = data[indexs, :]

        if data.shape[1] % (hz * 60) != 0:
            data = data[:, :-(data.shape[1] % (hz * 60))]
        data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
        data = np.transpose(data, (1, 0, 2))

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
    np.save(os.path.join(path, f"Dementia20000.npy"), {"timeseries": time_series,
                                                 "corr": pearson,
                                                 "labels": labels,
                                                 "subject_id": subject_ids})

    print("Finish")
































