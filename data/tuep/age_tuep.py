import os
import re
from random import shuffle

import mne
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def _init_worker():
    """Set BLAS threads to 1 so each worker is single-threaded."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

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
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7', 'T3', 'T5',
    'F8', 'T4', 'T6',
    'Fz', 'Cz', 'Pz',
]

_CHANNEL_NORM_MAP = {
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
}

def _normalize_channel(name):
    """Normalize an EDF channel name to the 10-20 short form."""
    for prefix in ['EEG ']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ['-REF', '-LE']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return _CHANNEL_NORM_MAP.get(name, name)

def _parse_age_from_edf_header(edf_path):
    """Extract age from EDF file header (Patient ID field).

    The TUH EDF Patient ID field has format:
    ``{subject_id} {M|F} 01-JAN-0000 {subject_id} Age:{NN}``

    Returns age as float, or None if not parseable.
    """
    try:
        with open(edf_path, 'rb') as f:
            header = f.read(256)
            patient_id = header[8:88].decode('ascii', errors='replace').strip()
        match = re.search(r'Age:(\d+)', patient_id)
        if match:
            age = float(match.group(1))
            return age
        return None
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AgeTUEPDataset(BaseDataset):
    """TUEP age regression dataset — predict chronological age from EEG.

    Task: given a 1‑minute window of 19‑channel EEG time series,
    predict the subject's age (continuous value in years).

    Uses all subjects from the TUH EEG Epilepsy Corpus.  Age is extracted
    from the EDF file header (``Age:NN`` field in Patient ID).
    """

    def __init__(self, data_config: DataConfig, k=0, train=True, one_hot=True,
                 episode_seed=None):
        super(AgeTUEPDataset, self).__init__(data_config, k, train, one_hot=one_hot,
                                             episode_seed=episode_seed)

    def load_data(self, one_hot=True):
        raw_path = "../data/TUEP/tuep_age.npz"
        raw = np.load(raw_path, allow_pickle=True)
        data = dict(raw) if hasattr(raw, 'files') else raw.item()
        time_series = data["timeseries"]
        labels, labels_raw, self.label_min, self.label_max = prepare_age_labels(
            data, source=raw_path)
        subject_id = data["subject_id"]
        self.hz = int(np.asarray(data["hz"]).item())
        self.data_config.hz = self.hz

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.output_dim = 1  # regression
        self.data_config.task_type = DataConfig.TASK_REGRESSION

        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['labels_raw'] = labels_raw
        self.all_data['subject_id'] = subject_id
        self._load_timellmknowledge_features(raw_path, data)

        # ── 使用预处理阶段预计算的划分（随机，无分层）──
        if 'split_train_index' in data:
            self.train_index = data['split_train_index']
            self.val_index = data['split_val_index']
            self.test_index = data['split_test_index']
        else:
            # 向后兼容：旧 .npz 无预计算划分时回退到分层划分
            print("  [WARN] 未找到预计算划分，回退到 _create_splits 分层划分")
            self._create_splits(labels, self.all_data['subject_id'])
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        sample_idx = idx[item]
        time_series = torch.from_numpy(
            self.all_data['time_series'][sample_idx]).float()
        labels = torch.tensor(
            self.all_data['labels'][sample_idx], dtype=torch.float32)
        knowledge_item = self._timellmknowledge_item(
            sample_idx, time_series, labels)
        if knowledge_item is not None:
            return knowledge_item

        # ── EEGNet fallback：仅时域信号（在线 FC/PSD 死代码已删）──
        return {
                'time_series': time_series,
                'labels': labels,
        }
