import os
from scipy.io import loadmat
import h5py

from random import shuffle, randrange

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

from .data_config import DataConfig
from .dataset import BaseDataset
from .preprocess import *


class MNREDDataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(MNREDDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                           episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        data = np.load(self.data_config.data_dir, allow_pickle=True).item()
        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = data.get("hz", 250)

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 2

        self.data_config.class_weight = [1, 2]
        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        self._create_splits(labels, self.all_data['subject_id'])
        if one_hot:
            self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(self.all_data['time_series'][idx[item]]).float()
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).to(torch.int64)

        sampling_init = (randrange(time_series.size(-1) - self.data_config.time_series_size)) \
            if self.data_config.dynamic else 0
        time_series = time_series[:, sampling_init:sampling_init + self.data_config.time_series_size]
        SFC = self.connectivity(time_series, activate=False)
        DFC = dynamic_connectivity(time_series.numpy(), 3 * self.hz, 1 * self.hz)
        DFC = torch.from_numpy(DFC).float()

        return {'time_series': time_series,
                'DFC': DFC,
                'correlation': SFC,
                'labels': labels,
                'sample_idx': idx[item]}



def eeg_preprocess_test(path):
    all_data = loadmat(os.path.join(path, "Data0324.mat"))
    time_series = all_data['data']
    pearson = np.array([np.corrcoef(t) for t in time_series])
    labels = all_data['label'][0]
    subject_id = all_data['subject'][0]
    np.save(os.path.join(path, "EEG.npy"), {"timeseries": time_series,
                                            "corr": pearson,
                                            "labels": labels,
                                            "subject_id": subject_id})


if __name__ == '__main__':
    eeg_preprocess_test("../data/EEG")
