import os
from random import shuffle

import mne
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def _init_worker():
    """Set BLAS threads to 1 so each worker is single-threaded."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

from sklearn.model_selection import train_test_split

from ..data_config import DataConfig
from ..dataset import BaseDataset
from ..preprocess import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Standard 19-channel 10-20 EEG order (CAUEEG-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

_TUEP_CHANNEL_ORDER = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',          # left hemisphere
    'Fp2', 'F4', 'C4', 'P4', 'O2',          # right hemisphere
    'F7', 'T3', 'T5',                         # left temporal
    'F8', 'T4', 'T6',                         # right temporal
    'Fz', 'Cz', 'Pz',                         # midline
]

# Mapping from EDF channel-name variants to standard 10-20 short names.
_CHANNEL_NORM_MAP = {
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
}

def _normalize_channel(name):
    """Normalize an EDF channel name to the 10-20 short form.

    ``EEG FP1-REF`` → ``Fp1``, ``EEG C3-LE`` → ``C3``.
    """
    for prefix in ['EEG ']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ['-REF', '-LE']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return _CHANNEL_NORM_MAP.get(name, name)

def _resolve_param(val, pos):
    """Resolve a per-class parameter that may be an int (both classes) or
    tuple ``(pos_val, neg_val)``.

    Parameters
    ----------
    val : int, tuple, or None
    pos : bool
        ``True`` for positive class, ``False`` for negative.

    Returns
    -------
    int or None
    """
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return val
    return val[0] if pos else val[1]

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class DiseaseTUEPDataset(BaseDataset):
    """TUH EEG Epilepsy Corpus — epilepsy vs no-epilepsy binary classification.

    Task: given a 1‑minute window of 19‑channel EEG time series,
    classify the subject as having epilepsy (1) or not (0).

    The dataset is preprocessed by ``disease_tuep_preprocess()`` and stored as a
    single ``.npz`` file.
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(DiseaseTUEPDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                                 episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        _suffix = "_10s" if os.environ.get("LLM4FC_DATA_10S") == "1" else ""
        raw_path = f"../data/TUEP/tuep_disease{_suffix}.npz"
        raw = np.load(raw_path, allow_pickle=True)
        data = dict(raw) if hasattr(raw, 'files') else raw.item()
        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = int(np.asarray(data["hz"]).item())
        self.data_config.hz = self.hz

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.output_dim = 2
        self.data_config.task_type = DataConfig.TASK_CLASSIFICATION

        self.data_config.class_weight = [1, 1]
        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id
        self._load_timellmknowledge_features(raw_path, data)

        if 'split_train_index' in data:
            self.train_index = data['split_train_index']
            self.val_index = data['split_val_index']
            self.test_index = data['split_test_index']
        else:
            print("  [WARN] 未找到预计算划分，回退到 _create_splits")
            self._create_splits(labels, self.all_data['subject_id'])
        self.all_data['labels'] = F.one_hot(
            torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        sample_idx = idx[item]
        time_series = torch.from_numpy(
            self.all_data['time_series'][sample_idx]).float()
        labels = torch.from_numpy(
            self.all_data['labels'][sample_idx]).to(torch.int64)
        knowledge_item = self._timellmknowledge_item(
            sample_idx, time_series, labels)
        if knowledge_item is not None:
            return knowledge_item

        # ── EEGNet fallback：仅时域信号（在线 FC/PSD 死代码已删）──
        return {
                'time_series': time_series,
                'labels': labels,
        }
