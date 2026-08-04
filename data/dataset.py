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
    # ── 辅助：被试标签聚合 ────────────────────────────────────────────
    @staticmethod
    def _make_subject_labels(labels, groups, unique_subjs):
        """为每个被试聚合样本级别的标签为被试级别。

        分类任务：取多数类（bincount argmax）；
        回归任务：取均值。
        """
        subj_lbls = np.zeros(len(unique_subjs))
        for i, s in enumerate(unique_subjs):
            mask = groups == s
            lbls = labels[mask]
            if np.issubdtype(lbls.dtype, np.integer):
                subj_lbls[i] = np.bincount(lbls.astype(int)).argmax()
            else:
                subj_lbls[i] = float(lbls.mean())
        return subj_lbls

    @staticmethod
    def _bin_continuous_labels(continuous_labels, n_bins=5):
        """将连续标签按分位数分箱为离散类别（用于回归任务的分层划分）。

        若唯一值不足 n_bins，则直接返回原值作为类别标识。
        """
        uniq = np.unique(continuous_labels)
        if len(uniq) < n_bins:
            # 唯一值太少，直接用原值（如婴幼儿年龄集中在 0-2）
            n_bins = len(uniq)
        if n_bins <= 1:
            return np.zeros_like(continuous_labels, dtype=int)
        bins = np.percentile(continuous_labels, np.linspace(0, 100, n_bins + 1))
        bins[-1] += 1e-6  # 确保最大值不被漏掉
        return np.digitize(continuous_labels, bins[:-1]) - 1

    # ── 数据划分 — 自动选择 K-Fold 或 train/val/test 模式 ──────────────
    def _create_splits(self, labels, groups):
        """创建数据划分。

        当 ``num_repeat >= 2`` 时沿用原有 GroupKFold 逻辑；
        当 ``num_repeat == 1`` 时自动切换为 60/20/20 按被试分组划分。

        分层策略按 ``task_type`` 自适应：
        - 分类：按每被试多数类分层
        - 回归：按等价分箱后的类别分层
        - 多值回归：不做分层，随机划分
        """
        task_type = self.data_config.task_type
        from .data_config import DataConfig as DC

        # ── 计算被试级别标签（用于分层）──
        unique_subjs = np.unique(groups)
        subj_labels_raw = self._make_subject_labels(labels, groups, unique_subjs)

        # ── 决定是否分层 + 分层用标签 ──
        if task_type == DC.TASK_MULTI_OUTPUT_REGRESSION:
            do_stratify = False
            stratify_labels = None
        elif task_type == DC.TASK_REGRESSION:
            do_stratify = True
            stratify_labels = self._bin_continuous_labels(subj_labels_raw)
        else:
            # classification（含二分类与多分类）
            do_stratify = True
            stratify_labels = subj_labels_raw.astype(int)

        if self.k_fold is not None:
            # ── 原有 K-Fold 模式（num_repeat >= 2）──
            stratify_for_kfold = stratify_labels[np.searchsorted(
                unique_subjs, groups)] if do_stratify else None
            self.train_index, self.test_index = list(
                self.k_fold.split(np.zeros(len(labels)),
                                  y=stratify_for_kfold,
                                  groups=groups)
            )[self.k]

            # Create a val split from within train (by subject)
            train_groups = groups[self.train_index]
            unique_train_subjs = np.unique(train_groups)
            if len(unique_train_subjs) >= 3:
                # per-subject labels for train subset
                train_subj_lbls = self._make_subject_labels(
                    labels[self.train_index], train_groups, unique_train_subjs)
                if task_type == DC.TASK_REGRESSION:
                    train_strat = self._bin_continuous_labels(train_subj_lbls)
                elif task_type == DC.TASK_MULTI_OUTPUT_REGRESSION:
                    train_strat = None
                else:
                    train_strat = train_subj_lbls.astype(int)
                val_ratio = self.data_config.val_set / self.data_config.train_set
                train_subjs, val_subjs = train_test_split(
                    unique_train_subjs, test_size=min(val_ratio, 0.3),
                    stratify=train_strat,
                    random_state=42)
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

            stratify_arg = stratify_labels if do_stratify else None

            # Step 1: train vs (val + test)
            train_subjs, rest_subjs = train_test_split(
                unique_subjs,
                test_size=1.0 - train_ratio,
                stratify=stratify_arg,
                random_state=42,
            )

            # Step 2: val vs test from rest
            rest_mask = np.isin(unique_subjs, rest_subjs)
            if do_stratify:
                rest_stratify = stratify_labels[rest_mask]
            else:
                rest_stratify = None
            val_frac = val_ratio / (1.0 - train_ratio)    # 0.1 / 0.2 = 0.5
            val_subjs, test_subjs = train_test_split(
                rest_subjs,
                test_size=1.0 - val_frac,
                stratify=rest_stratify,
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
    def connectivity(time_series):
        """Compute static functional connectivity (Pearson correlation matrix).

        Uses nilearn ConnectivityMeasure under the hood.
        time_series: (N, T) tensor → returns (N, N) correlation matrix.
        """
        conn_measure = connectome.ConnectivityMeasure(kind='correlation')
        conn = conn_measure.fit_transform(time_series.T.unsqueeze(0).numpy())[0]
        return torch.from_numpy(conn)

    @staticmethod
    def sparsify_fc(fc, threshold=0.0, keep_ratio=1.0):
        """对 FC 矩阵进行阈值 + Top-K 稀疏化（保留自环，保持对称）。

        支持 2D (N, N) 和 3D (W, N, N) 输入。
        当 threshold <= 0 且 keep_ratio >= 1.0 时直接返回原矩阵。

        Parameters
        ----------
        fc : Tensor, shape (N, N) or (W, N, N)
        threshold : float  绝对值低于此值的边置零（0 = 不过滤）
        keep_ratio : float  Top-K 保留比例（1.0 = 全保留），仅作用于非对角线

        Returns
        -------
        fc_sparse : Tensor, same shape as fc
        """
        if threshold <= 0.0 and keep_ratio >= 1.0:
            return fc

        single = (fc.dim() == 2)
        if single:
            fc = fc.unsqueeze(0)  # → (1, N, N)

        B, N, _ = fc.shape
        device = fc.device

        fc_sparse = fc.clone()
        fc_abs = fc.abs().clone()

        # 临时清零对角线，避免自环参与 Top-K 竞争
        idx = torch.arange(N, device=device)
        fc_sparse[:, idx, idx] = 0.0
        fc_abs[:, idx, idx] = 0.0

        # ── Top-K 稀疏化：每张图独立选择绝对值最大的 keep_ratio 条边 ──
        if keep_ratio < 1.0:
            triu_idx = torch.triu_indices(N, N, offset=1)
            for b in range(B):
                vals = fc_abs[b, triu_idx[0], triu_idx[1]]
                k = max(1, int(round(vals.shape[0] * keep_ratio)))
                th = vals.topk(k).values[-1]
                mask = (fc_abs[b] >= th)
                fc_abs[b] = fc_abs[b] * mask
                fc_sparse[b] = fc_sparse[b] * mask

        # ── 阈值稀疏化 ──
        if threshold > 0.0:
            fc_sparse[fc_abs < threshold] = 0.0

        # ── 恢复对角线为 1 ──
        fc_sparse[:, idx, idx] = 1.0

        if single:
            fc_sparse = fc_sparse.squeeze(0)
        return fc_sparse

    @staticmethod
    def dynamic_connectivity(time_series, window_size, step_size):
        """Compute dynamic FC via sliding windows.

        Each window calls connectivity() (nilearn Pearson correlation).

        Args:
            time_series: (N, T) tensor.
            window_size: int, samples per window.
            step_size:   int, stride between window starts.

        Returns:
            (num_windows, N, N) tensor.
        """
        N, T = time_series.shape
        num_windows = (T - window_size) // step_size + 1

        dfc = []
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            window_data = time_series[:, start:end]
            dfc.append(BaseDataset.connectivity(window_data))

        return torch.stack(dfc, dim=0)

    def __len__(self):
        return len(self._active_index)

    @abstractmethod
    def __getitem__(self, item):
        pass
