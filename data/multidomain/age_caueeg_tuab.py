import numpy as np
import torch
from random import shuffle

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import prepare_age_labels

class AgeCAUEEGTUABDataset(BaseDataset):
    """CAUEEG + TUAB 年龄回归融合（连续年龄）。

    domain_label: 0=CAUEEG, 1=TUAB。
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(AgeCAUEEGTUABDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                   episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        d1 = dict(np.load("../data/CAUEEG/caueeg_age.npz", allow_pickle=True))
        d2 = dict(np.load("../data/TUAB/tuab_age.npz", allow_pickle=True))
        assert d1['hz'] == d2['hz'], "两个数据集的采样率不一致"
        self.hz = d1['hz']

        n1 = len(d1['timeseries'])
        time_series = np.concatenate([d1['timeseries'], d2['timeseries']], axis=0)
        labels1, raw1, self.label_min, self.label_max = prepare_age_labels(
            d1, source="../data/CAUEEG/caueeg_age.npz")
        labels2, raw2, _, _ = prepare_age_labels(
            d2, source="../data/TUAB/tuab_age.npz")
        labels = np.concatenate([labels1, labels2], axis=0)
        labels_raw = np.concatenate([raw1, raw2], axis=0)
        subject_id = np.concatenate([d1['subject_id'], d2['subject_id'] + 100000], axis=0)
        domain_label = np.concatenate([
            np.zeros(n1, dtype=np.int64),
            np.ones(len(d2['timeseries']), dtype=np.int64),
        ])

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.output_dim = 1
        self.data_config.task_type = DataConfig.TASK_REGRESSION

        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['labels_raw'] = labels_raw
        self.all_data['subject_id'] = subject_id
        self.all_data['domain_label'] = domain_label

        # 合并两个数据集各自预计算的划分（第二份加偏移）
        self.train_index = np.concatenate([d1['split_train_index'], d2['split_train_index'] + n1])
        self.val_index = np.concatenate([d1['split_val_index'], d2['split_val_index'] + n1])
        self.test_index = np.concatenate([d1['split_test_index'], d2['split_test_index'] + n1])

        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(self.all_data['time_series'][idx[item]]).float()
        labels = torch.tensor(self.all_data['labels'][idx[item]], dtype=torch.float32)

        # ── EEGNet fallback：仅时域信号（在线 FC/PSD 死代码已删）──
        return {
                'time_series': time_series,
                'labels': labels,
                'domain_label': torch.tensor(
                    self.all_data['domain_label'][idx[item]], dtype=torch.long),
        }
