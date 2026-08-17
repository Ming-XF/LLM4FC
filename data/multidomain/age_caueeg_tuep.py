import numpy as np
import torch
from random import shuffle

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import AGE_LABEL_MIN, AGE_LABEL_MAX


class AgeCAUEEGTUEPDataset(BaseDataset):
    """CAUEEG + TUEP 年龄回归融合（连续年龄）。

    domain_label: 0=CAUEEG, 1=TUEP。
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(AgeCAUEEGTUEPDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                   episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        d1 = dict(np.load("../data/CAUEEG/caueeg_age.npz", allow_pickle=True))
        d2 = dict(np.load("../data/TUEP/tuep_age.npz", allow_pickle=True))
        assert d1['hz'] == d2['hz'], "两个数据集的采样率不一致"
        self.hz = d1['hz']

        n1 = len(d1['timeseries'])
        time_series = np.concatenate([d1['timeseries'], d2['timeseries']], axis=0)
        # 联合归一化：固定先验边界，统一跨域量纲（无泄漏）
        raw1 = d1['labels_raw'] if 'labels_raw' in d1 else d1['labels']
        raw2 = d2['labels_raw'] if 'labels_raw' in d2 else d2['labels']
        labels_raw = np.concatenate([raw1, raw2], axis=0).astype(np.float32)
        label_min = float(AGE_LABEL_MIN)
        label_max = float(AGE_LABEL_MAX)
        span = label_max - label_min
        labels = ((labels_raw - label_min) / span).astype(np.float32)
        self.label_min = label_min
        self.label_max = label_max
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

        SFC = self.connectivity(time_series)
        SFC = self.sparsify_fc(SFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)
        window_size = 12 * self.hz
        step_size = (60 * self.hz - window_size) // 9
        DFC = self.dynamic_connectivity(time_series, window_size, step_size)
        DFC = self.sparsify_fc(DFC, self.data_config.fc_threshold, self.data_config.fc_keep_ratio)

        return {
                'DFC': DFC,
                'correlation': SFC,
                'labels': labels,
                'domain_label': torch.tensor(
                    self.all_data['domain_label'][idx[item]], dtype=torch.long),
        }
