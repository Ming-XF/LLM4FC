from abc import abstractmethod
from pathlib import Path

import torch
from nilearn import connectome
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import Dataset
import numpy as np

from .data_config import DataConfig
from .knowledge_preprocess import FEATURE_VERSION, knowledge_feature_path


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

        # ── few-shot 采样：load_data 完成后独立调用，与是否使用预计算划分无关 ──
        if self.episode_seed is not None and self.data_config.few_shot > 0:
            labels = self.all_data['labels']
            if labels.ndim > 1 and labels.shape[1] > 1:
                labels = labels.argmax(axis=1)
            self._apply_few_shot_sampling(
                labels, self.all_data['subject_id'], self.episode_seed)

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

    @property
    def uses_timellmknowledge_features(self):
        """Whether this dataset must load the offline knowledge sidecar."""
        return self.data_config.model == 'TimeLLMKnowledge'

    def _load_timellmknowledge_features(self, raw_npz_path, raw_data):
        """Load and validate the sidecar that is aligned to the raw NPZ."""
        if not self.uses_timellmknowledge_features:
            return

        feature_path = knowledge_feature_path(raw_npz_path)
        if not feature_path.exists():
            raise FileNotFoundError(
                f"TimeLLMKnowledge feature file is missing: {feature_path}. "
                f"Generate it with: python -m data.knowledge_preprocess "
                f"--input {raw_npz_path}")

        features = np.load(feature_path, allow_pickle=False)
        try:
            required = {
                'feature_version', 'source_size', 'source_mtime_ns',
                'source_shape', 'hz', 'num_local_segments', 'sample_index',
                'subject_id', 'global_fc', 'local_fc', 'global_psd',
                'local_psd', 'local_mask',
            }
            missing = required.difference(features.files)
            if missing:
                raise ValueError(
                    f"Feature file {feature_path} is missing fields: {sorted(missing)}")

            version = str(np.asarray(features['feature_version']).item())
            if version != FEATURE_VERSION:
                raise ValueError(
                    f"Unsupported feature version {version!r}; expected {FEATURE_VERSION!r}")

            raw_path = Path(raw_npz_path).expanduser().resolve()
            stat = raw_path.stat()
            if int(np.asarray(features['source_size']).item()) != stat.st_size:
                raise ValueError(
                    f"Feature file {feature_path} does not match raw file size")
            if int(np.asarray(features['source_mtime_ns']).item()) != stat.st_mtime_ns:
                raise ValueError(
                    f"Feature file {feature_path} is stale: raw file modification time changed")

            time_series = np.asarray(raw_data['timeseries'])
            source_shape = tuple(np.asarray(features['source_shape']).tolist())
            if source_shape != time_series.shape:
                raise ValueError(
                    f"Feature source shape {source_shape} does not match raw shape "
                    f"{time_series.shape}")
            hz = int(np.asarray(features['hz']).item())
            if hz != int(np.asarray(raw_data['hz']).item()):
                raise ValueError("Feature sampling rate does not match raw NPZ")

            num_samples, num_channels, _ = time_series.shape
            sample_index = np.asarray(features['sample_index'])
            if not np.array_equal(sample_index, np.arange(num_samples)):
                raise ValueError("Feature sample_index is not aligned to raw sample order")
            raw_subjects = np.asarray(raw_data['subject_id']).astype(str)
            if not np.array_equal(
                    np.asarray(features['subject_id']).astype(str), raw_subjects):
                raise ValueError("Feature subject_id order does not match raw NPZ")

            num_segments = int(np.asarray(features['num_local_segments']).item())
            expected = {
                'global_fc': (num_samples, num_channels, num_channels),
                'local_fc': (num_samples, num_segments, num_channels, num_channels),
                'global_psd': (num_samples, num_channels, None),
                'local_psd': (num_samples, num_segments, num_channels, None),
                'local_mask': (num_samples, num_segments),
            }
            arrays = {}
            for name, shape in expected.items():
                array = np.asarray(features[name])
                if array.ndim != len(shape) or any(
                        expected_dim is not None and actual_dim != expected_dim
                        for actual_dim, expected_dim in zip(array.shape, shape)):
                    raise ValueError(
                        f"Feature {name} has shape {array.shape}, expected {shape}")
                arrays[name] = array
            if arrays['global_psd'].shape[-1] != arrays['local_psd'].shape[-1]:
                raise ValueError("Global and local PSD frequency dimensions differ")

            self.all_data.update(arrays)
            self.data_config.hz = hz
            self.data_config.num_local_segments = num_segments
        finally:
            features.close()

    def _timellmknowledge_item(self, sample_idx, time_series, labels):
        """Build one model input without recomputing FC or PSD online."""
        if not self.uses_timellmknowledge_features:
            return None
        return {
            'time_series': time_series,
            'global_fc': torch.from_numpy(
                self.all_data['global_fc'][sample_idx]).float(),
            'global_psd': torch.from_numpy(
                self.all_data['global_psd'][sample_idx]).float(),
            'local_fc': torch.from_numpy(
                self.all_data['local_fc'][sample_idx]).float(),
            'local_psd': torch.from_numpy(
                self.all_data['local_psd'][sample_idx]).float(),
            'local_mask': torch.from_numpy(
                self.all_data['local_mask'][sample_idx]).bool(),
            'labels': labels,
        }

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
        if task_type == DC.TASK_REGRESSION:
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

    # ── Few-shot 被试采样（独立方法，由 __init__ 调用）──────────────
    def _apply_few_shot_sampling(self, labels, groups, episode_seed):
        """从训练集中每类采样 N 个被试，保留其全部窗口样本。

        按任务类型自适应：
        - 分类（二类/多类）：每类采样 n_subj_per_class 个被试
        - 回归：随机采样 n_subj_per_class 个被试
        - 多值回归：随机采样 n_subj_per_class 个被试

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
        unique_subjs = np.unique(train_groups)
        rng = np.random.RandomState(episode_seed)

        task_type = self.data_config.task_type

        if task_type == DataConfig.TASK_REGRESSION:
            # 回归 / 多值回归：随机采样
            n = min(n_subj_per_class, len(unique_subjs))
            keep_subjs = rng.choice(unique_subjs, size=n, replace=False)

        else:
            # 分类（二类/多类）：每类采样
            subj_lbls = np.array([
                np.bincount(train_lbls[train_groups == s].astype(int)).argmax()
                for s in unique_subjs
            ])
            keep_subjs = []
            for c in np.unique(subj_lbls):
                pool = unique_subjs[subj_lbls == c]
                n = min(n_subj_per_class, len(pool))
                keep_subjs.append(rng.choice(pool, size=n, replace=False))
            keep_subjs = np.concatenate(keep_subjs)

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
    def spectral_stats(time_series, hz):
        """CAER-v2: 确定性频谱证据（9 维，无监督规则）。

        整分钟版（TimeLLMKnowledge）：15 个非重叠 4s 段做 Welch PSD
        （hanning 窗），每段先算比值、再对 15 段求均值，得到
        delta/theta/alpha/beta 相对功率，以及 theta/beta、
        alpha/(d+t)、慢波指数 (d+t)/(a+b)、额/枕 alpha 比、颞区 theta
        占比。不再使用 10 个 12s 滑窗。

        Returns:
            torch.Tensor shape (9,)
        """
        nperseg = 4 * hz
        nseg = (60 * hz) // nperseg                          # 15
        freqs = np.fft.rfftfreq(nperseg, 1 / hz)
        bands = {'d': (0.5, 4.0), 't': (4.0, 8.0),
                 'a': (8.0, 13.0), 'b': (13.0, 30.0)}
        frontal = [0, 1, 5, 6, 10, 13]
        posterior = [3, 4, 8, 9, 12, 15]
        temporal = [11, 12, 14, 15]
        win = np.hanning(nperseg)
        ts = time_series.detach().numpy().astype(np.float64)
        ts = ts[:, :nseg * nperseg]
        rows = []
        for k in range(nseg):
            w = ts[:, k * nperseg:(k + 1) * nperseg]
            w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-8)
            pw = {bn: [] for bn in bands}
            for ch in range(w.shape[0]):
                ps = np.abs(np.fft.rfft(w[ch] * win)) ** 2
                for bn, (lo, hi) in bands.items():
                    pw[bn].append(ps[(freqs >= lo) & (freqs < hi)].sum())
            # +1e-12：全平段（如坏导/零填充，约 0.3% 样本）PSD 全零，
            # 否则 rel = 0/0 = nan 会沿 stats 传播进模型
            total = sum(np.mean(v) for v in pw.values()) + 1e-12
            rel = {bn: np.mean(v) / total for bn, v in pw.items()}
            fa = sum(pw['a'][ch] for ch in frontal)
            pa = sum(pw['a'][ch] for ch in posterior)
            tt = sum(pw['t'][ch] for ch in temporal)
            rows.append([
                rel['d'], rel['t'], rel['a'], rel['b'],
                rel['t'] / max(rel['b'], 1e-9),
                rel['a'] / max(rel['d'] + rel['t'], 1e-9),
                (rel['d'] + rel['t']) / max(rel['a'] + rel['b'], 1e-9),
                fa / max(pa, 1e-9),
                tt / total,
            ])
        return torch.tensor(np.array(rows).mean(0), dtype=torch.float32)

    @staticmethod
    def morph_stats(time_series, hz):
        """CAER-v3: 确定性形态学/周期-非周期证据（14 维，无监督规则）。

        时域形态（发作间期放电相关）:
          ll_mean/ll_max      线长（|Δx| 均值，逐通道均值/最大）
          sharp_mean/max/focal 锐度 |ΔΔx|（均值/最大/跨通道灶性 std）
          kurt_mean/max       幅度峰度（瞬态尖波的厚尾特征）
          zcr_mean            零交叉率
          gamma_pow           30-45Hz 相对功率
          spike_mean/max      |ΔΔx|>6×MAD 的棘波速率（均值/最大通道）
        周期-非周期（年龄相关）:
          apf / apf_std       alpha 峰值频率（7-13Hz，均值/15 段 std）
          slope               2-40Hz log-log PSD 非周期斜率

        整分钟版（TimeLLMKnowledge）：时域特征在整分钟上计算（不变），
        PSD 特征改用 15 个非重叠 4s 段（替换 10 个 12s 滑窗）。

        Returns:
            torch.Tensor shape (14,)
        """
        ts = time_series.detach().numpy().astype(np.float64)
        ts = (ts - ts.mean(1, keepdims=True)) / (ts.std(1, keepdims=True) + 1e-8)
        C, N = ts.shape

        d1 = np.abs(np.diff(ts, axis=1))
        d2 = np.abs(np.diff(ts, n=2, axis=1))
        ll = d1.sum(1) / (N - 1)
        sharp = d2.mean(1)
        sharp_max_ch = d2.max(1)
        kurt = ((ts - ts.mean(1, keepdims=True)) ** 4).mean(1) / (
            (ts ** 2).mean(1) ** 2 + 1e-12) - 3.0
        zcr = (np.diff(ts > 0, axis=1) != 0).sum(1) / (N - 1)
        mad = np.median(d2, axis=1) / 0.6745
        spike = ((d2 > 6 * mad[:, None]).sum(1)) / (N - 2)

        # PSD: 整分钟 15 个非重叠 4s 分段 → 0.25Hz 分辨率
        nperseg = 4 * hz
        nseg = (60 * hz) // nperseg
        freqs = np.fft.rfftfreq(nperseg, 1 / hz)
        win = np.hanning(nperseg)
        gmask = (freqs >= 30) & (freqs < 45)
        amask = (freqs >= 7) & (freqs < 13)
        smask = (freqs >= 2) & (freqs < 40)
        lf = np.log(freqs[smask])
        ts_seg = ts[:, :nseg * nperseg].reshape(C, nseg, nperseg)
        gamma_vals, apf_vals, slope_vals = [], [], []
        for k in range(nseg):
            w = ts_seg[:, k]
            w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-8)
            ps = np.abs(np.fft.rfft(w * win, axis=1)) ** 2
            tot = ps.sum(1) + 1e-12
            gamma_vals.append((ps[:, gmask].sum(1) / tot).mean())
            apf_vals.append(freqs[amask][ps[:, amask].argmax(1)].mean())
            slope_vals.append(np.polyfit(
                lf, np.log(ps[:, smask].mean(0) + 1e-12), 1)[0])

        return torch.tensor([
            ll.mean(), ll.max(),
            sharp.mean(), sharp_max_ch.max(), sharp_max_ch.std(),
            kurt.mean(), kurt.max(),
            zcr.mean(),
            np.mean(gamma_vals),
            spike.mean(), spike.max(),
            np.mean(apf_vals), np.std(apf_vals),
            np.mean(slope_vals),
        ], dtype=torch.float32)

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
