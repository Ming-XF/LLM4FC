from abc import abstractmethod

import torch
from nilearn import connectome
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import Dataset
import numpy as np

from .data_config import DataConfig


class BaseDataset(Dataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(BaseDataset, self).__init__()
        self.data_config = data_config
        self._mode = 'train' if train else 'test'
        if data_config.n_splits > 1:
            self.k_fold = GroupKFold(n_splits=data_config.n_splits)
        else:
            self.k_fold = None                  # train/val/test 模式
        self.k = k
        self.episode_seed = episode_seed
        self.all_data = {}
        self.train_index = None
        self.val_index = None                    # 新增 — 仅 train/val/test 模式有效
        self.test_index = None
        self.train_data = None
        self.test_data = None
        self.load_data(one_hot=one_hot)

    # ── mode 属性 — 取代旧的 train boolean ────────────────────────────
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ('train', 'val', 'test'):
            raise ValueError(
                f"mode must be 'train', 'val', or 'test', got '{value}'")
        self._mode = value

    @property
    def train(self):
        """向后兼容 — 旧代码 ``dataset.train = False`` 仍然有效。"""
        return self._mode == 'train'

    @train.setter
    def train(self, value):
        self._mode = 'train' if value else 'test'

    @property
    def _active_index(self):
        """根据当前 mode 返回对应的样本索引数组。"""
        if self._mode == 'train':
            return self.train_index
        elif self._mode == 'val':
            return self.val_index if self.val_index is not None and len(self.val_index) > 0 else self.test_index
        else:
            return self.test_index

    # ── 数据划分 — 自动选择 K-Fold 或 train/val/test 模式 ──────────────
    def _create_splits(self, labels, groups):
        """创建数据划分。

        当 ``num_repeat >= 2`` 时沿用原有 GroupKFold 逻辑；
        当 ``num_repeat == 1`` 时自动切换为 60/20/20 按被试分组划分。
        """
        if self.k_fold is not None:
            # ── 原有 K-Fold 模式（num_repeat >= 2）──
            self.train_index, self.test_index = list(
                self.k_fold.split(np.zeros(len(labels)), labels, groups=groups)
            )[self.k]

            # Create a val split from within train (10% of train, by subject)
            train_groups = groups[self.train_index]
            train_lbls = labels[self.train_index]
            unique_train_subjs = np.unique(train_groups)
            if len(unique_train_subjs) >= 3:
                subj_lbls = np.array([
                    np.bincount(train_lbls[train_groups == s].astype(int)).argmax()
                    for s in unique_train_subjs
                ])
                val_ratio = self.data_config.val_set / self.data_config.train_set
                train_subjs, val_subjs = train_test_split(
                    unique_train_subjs, test_size=min(val_ratio, 0.3),
                    stratify=subj_lbls, random_state=42)
                val_mask = np.isin(train_groups, val_subjs)
                train_mask = np.isin(train_groups, train_subjs)
                self.val_index = self.train_index[val_mask]
                self.train_index = self.train_index[train_mask]
            else:
                self.val_index = np.array([], dtype=int)
        else:
            # ── train/val/test 模式 — 按被试分组划分，防止数据泄露 ──
            train_ratio = self.data_config.train_set   # default 0.6
            val_ratio = self.data_config.val_set        # default 0.2
            unique_subjs = np.unique(groups)

            # 每个被试的主标签（用于分层抽样）
            subj_labels = np.array([
                np.bincount(labels[groups == s].astype(int)).argmax()
                for s in unique_subjs
            ])

            # Step 1: train vs (val + test)
            train_subjs, rest_subjs = train_test_split(
                unique_subjs,
                test_size=1.0 - train_ratio,
                stratify=subj_labels,
                random_state=42,
            )

            # Step 2: val vs test from rest
            rest_mask = np.isin(unique_subjs, rest_subjs)
            rest_labels = subj_labels[rest_mask]
            val_frac = val_ratio / (1.0 - train_ratio)    # 0.1 / 0.2 = 0.5
            val_subjs, test_subjs = train_test_split(
                rest_subjs,
                test_size=1.0 - val_frac,
                stratify=rest_labels,
                random_state=42,
            )

            self.train_index = np.where(np.isin(groups, train_subjs))[0]
            self.val_index = np.where(np.isin(groups, val_subjs))[0]
            self.test_index = np.where(np.isin(groups, test_subjs))[0]

        # ── few-shot：由外部 episode_seed 控制采样时机 ──
        if self.episode_seed is not None and self.data_config.few_shot > 0:
            self._apply_few_shot_sampling(labels, groups, self.episode_seed)

    # ── Few-shot 被试采样（独立方法，外部按 episode 调用）──────────────
    def _apply_few_shot_sampling(self, labels, groups, episode_seed):
        """从训练集中每类采样 N 个被试，保留其全部窗口样本。

        Parameters
        ----------
        episode_seed : int
            该 episode 的随机种子；不同 episode 产生不同采样，但 train/val/test
            划分不受影响（split 使用固定种子 42）。
        """
        n_subj_per_class = self.data_config.few_shot
        if n_subj_per_class <= 0 or len(self.train_index) == 0:
            return

        train_groups = groups[self.train_index]
        train_lbls = labels[self.train_index]

        # 每个受试者的主标签
        unique_subjs = np.unique(train_groups)
        subj_lbls = np.array([
            np.bincount(train_lbls[train_groups == s].astype(int)).argmax()
            for s in unique_subjs
        ])

        # 正/负类受试者分别随机采样
        pos_subjs = unique_subjs[subj_lbls == 1]
        neg_subjs = unique_subjs[subj_lbls == 0]

        rng = np.random.RandomState(episode_seed)
        n = min(n_subj_per_class, len(pos_subjs), len(neg_subjs))

        keep_pos = rng.choice(pos_subjs, size=n, replace=False)
        keep_neg = rng.choice(neg_subjs, size=n, replace=False)
        keep_subjs = np.concatenate([keep_pos, keep_neg])

        keep_mask = np.isin(train_groups, keep_subjs)
        self.train_index = self.train_index[keep_mask]

    @abstractmethod
    def load_data(self, one_hot=True):
        pass

    @staticmethod
    def connectivity(time_series, activate=True):
        conn_measure = connectome.ConnectivityMeasure(kind='correlation')
        # conn_measure = connectome.ConnectivityMeasure(kind='correlation', cov_estimator=OAS(store_precision=False))
        connectivity = conn_measure.fit_transform(time_series.T.unsqueeze(0).numpy())[0]
        connectivity = torch.from_numpy(connectivity)
        if activate:
            connectivity = torch.arctanh(connectivity)
            connectivity = torch.clamp(connectivity, -1.0, 1.0)
            diag = torch.diag_embed(torch.diag(connectivity))
            connectivity = connectivity - diag
        return connectivity

    @staticmethod
    def correlation(time_series, activate=True):
        feature = torch.einsum('nt, mt ->nm', time_series, time_series) / (time_series.size(1)-1)
        feature = torch.clamp(feature, -1.0, 1.0)
        if activate:
            feature = torch.arctanh(feature)
            feature = torch.clamp(feature, -1.0, 1.0)
            diag = torch.diag_embed(torch.diag(feature))
            feature = feature - diag
        return feature

    @staticmethod
    def norm(time_series):
        time_series -= torch.mean(time_series, dim=1, keepdim=True)
        std = torch.std(time_series, dim=1, keepdim=True)
        std[std < torch.finfo(torch.float64).eps] = 1.
        time_series /= std
        return time_series

    def __len__(self):
        return len(self._active_index)

    @abstractmethod
    def __getitem__(self, item):
        pass
