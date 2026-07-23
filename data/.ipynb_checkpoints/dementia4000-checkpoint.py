import os
import random
import time
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
from multiprocessing import Pool, cpu_count

from .dementia_mms import dynamic_connectivity

import pdb


class Dementia4000Dataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(Dementia4000Dataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

    def load_data(self, one_hot=True):
        raw = np.load(self.data_config.data_dir, allow_pickle=True)
        data = dict(raw) if hasattr(raw, 'files') else raw.item()  # 兼容 .npz / .npy
        time_series = data["timeseries"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        self.hz = data["hz"]

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 4

        self.data_config.class_weight = [1, 1, 1, 1]
        self.all_data['time_series'] = time_series
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id
        if 'dfc' in data:
            self.all_data['dfc'] = data['dfc']

        # ── 加载推理文本 (tech2.md §七) ──
        # reasoning_texts.json 格式: {"0": [text_t03, text_t06, text_t09], "1": [...], ...}
        # 如果文件不存在或加载失败，设为 None（训练时 fallback 到短诊断结论）
        self.reasoning_texts = None
        reasoning_path = os.path.join(os.path.dirname(self.data_config.data_dir),
                                       'reasoning_texts.json')
        if os.path.exists(reasoning_path):
            try:
                with open(reasoning_path, 'r', encoding='utf-8') as f:
                    self.reasoning_texts = json.load(f)
                n_loaded = len(self.reasoning_texts)
                print(f"[Dementia4000] Loaded reasoning_texts.json: {n_loaded} samples")
            except Exception as e:
                print(f"[Dementia4000] Failed to load reasoning_texts.json: {e}")
        else:
            print(f"[Dementia4000] reasoning_texts.json not found at {reasoning_path} — "
                  f"will use short conclusion only")

        if self.subject_id:
            self.select_subject()
        self._create_splits(labels, self.all_data['subject_id'])
        self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()
        shuffle(self.train_index)

    def __getitem__(self, item):
        idx = self._active_index
        time_series = torch.from_numpy(self.all_data['time_series'][idx[item]]).float()
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).to(torch.int64)

        sampling_init = (randrange(time_series.size(-1) - self.data_config.time_series_size)) \
            if self.data_config.dynamic else 0
        time_series = time_series[:, sampling_init:sampling_init + self.data_config.time_series_size]
        SFC = self.connectivity(time_series, activate=False)
        if not self.data_config.dynamic and 'dfc' in self.all_data:
            DFC = torch.from_numpy(self.all_data['dfc'][idx[item]]).float()
        else:
            DFC = dynamic_connectivity(time_series.numpy(), 3 * self.hz, 1 * self.hz)
            DFC = torch.from_numpy(DFC).float()

        # ── 推理文本 (tech2.md §九) ──
        # 每个 batch 从 N 条推理文本中随机选一条，防止过拟合。
        # 如果 reasoning_texts 不可用，返回空字符串（训练时 fallback 到短诊断结论）。
        reasoning_text = ""
        if self.reasoning_texts is not None:
            idx_str = str(idx[item])
            texts_list = self.reasoning_texts.get(idx_str)
            if texts_list and len(texts_list) > 0:
                valid_texts = [t for t in texts_list if t and t.strip()]
                if valid_texts:
                    reasoning_text = random.choice(valid_texts)

        return {'time_series': time_series,
                'DFC': DFC,
                'correlation': SFC,  # SFC, for backward compat
                'labels': labels,
                'reasoning_text': reasoning_text}

    def select_subject(self):
        self.selected = [self.subject_id]
        index = np.sum(self.all_data["subject_id"] == i for i in self.selected) == 1
        self.all_data['time_series'] = self.all_data['time_series'][index]
        self.all_data['labels'] = self.all_data['labels'][index]
        self.all_data['subject_id'] = self.all_data['subject_id'][index]
        if 'dfc' in self.all_data:
            self.all_data['dfc'] = self.all_data['dfc'][index]


def _compute_dfc(args):
    """计算单个 segment 的 DFC，供 multiprocessing worker 调用"""
    segment, hz = args
    return dynamic_connectivity(segment, 3 * hz, 1 * hz)


def _compute_dfc_description_worker(args):
    """计算单个样本的 DFC 时间统计量 + 文本描述，供 multiprocessing worker 调用

    Args:
        args: tuple — (index, ts, hz, class_name)

    Returns:
        tuple — (index, fc_mean, fc_std, num_windows, fc_description)
    """
    idx, ts, hz, class_name = args
    fc_mean, fc_std, nw = compute_dfc_stats(ts, hz)
    fc_desc = fc_to_text_description(fc_mean, class_name, fc_std=fc_std, num_windows=nw)
    return idx, fc_mean.astype(np.float32), fc_std.astype(np.float32), nw, fc_desc


def _call_api_worker(args):
    """单次 DeepSeek API 调用，供 ThreadPoolExecutor worker 使用（模块级函数以支持 pickle）

    Args:
        args: tuple — (sample_idx, temp_idx, temperature, fc_desc, class_name,
                       system_prompt, api_key, base_url, model)

    Returns:
        tuple — (sample_idx, temp_idx, temperature, result_dict)
    """
    sample_idx, temp_idx, temperature, fc_desc, class_name, system_prompt, api_key, base_url, model = args

    task_prompt = build_task_prompt_ds(fc_desc, class_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]
    result = _call_deepseek_api(messages, temperature, api_key, base_url, model)
    return sample_idx, temp_idx, temperature, result


def _process_one_sample(args):
    """处理单个 EDF 样本，供 multiprocessing worker 调用（模块级函数以支持 pickle）"""
    sample, signal_folder, hz = args
    serial = sample['serial']
    subject_id = int(re.findall(r'\d+', serial)[0])
    edf_path = os.path.join(signal_folder, f"{serial}.edf")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, times = raw[:, :]
    data = data[:19, :]

    if data.shape[1] % (hz * 60) != 0:
        data = data[:, :-(data.shape[1] % (hz * 60))]
    data = data.reshape(data.shape[0], data.shape[1] // (hz * 60), -1)
    data = np.transpose(data, (1, 0, 2))

    if 'ad' in sample['symptom']:
        label = np.full(data.shape[0], 3)
    elif 'mci_amnestic' in sample['symptom']:
        label = np.full(data.shape[0], 2)
    elif 'smi' in sample['symptom']:
        label = np.full(data.shape[0], 1)
    else:
        label = np.full(data.shape[0], 0)

    subj_ids = np.full(data.shape[0], subject_id)

    return data, label, subj_ids


def dementia_preprocess(path="../data/Dementia4000/", hz=200):
    # 配置路径
    annotation_file = os.path.join(path, "caueeg-dataset/annotation.json")
    signal_folder = os.path.join(path, os.path.join('caueeg-dataset/signal', "edf"))
    output_path = os.path.join(path, "Dementia4000.npz")

    # 读取标注文件
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)

    # 筛选目标样本
    target_samples = [s for s in annotation['data']
                      if 'ad' in s['symptom'] or 'cb_normal' in s['symptom']
                      or 'smi' in s['symptom'] or 'mci_amnestic' in s['symptom']]

    n_workers = min(cpu_count(), len(target_samples), 8)

    print(f"并行处理 {len(target_samples)} 个样本，使用 {n_workers} 个进程...")

    # 构建参数列表
    task_args = [(sample, signal_folder, hz) for sample in target_samples]

    # 多进程并行处理
    ts_list, lbl_list, subj_list = [], [], []
    with Pool(processes=n_workers) as pool:
        for data, label, subj_ids in tqdm(pool.imap_unordered(_process_one_sample, task_args),
                                          total=len(task_args), desc="加载EDF"):
            ts_list.append(data)
            lbl_list.append(label)
            subj_list.append(subj_ids)

    # 一次性拼接，避免循环中 np.append 的 O(n²) 开销
    time_series = np.concatenate(ts_list, axis=0)
    labels = np.concatenate(lbl_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)

    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)

    print(f"Precomputing DFC ({len(time_series)} segments, {n_workers} workers)...")
    dfc_task_args = [(time_series[i], hz) for i in range(len(time_series))]
    with Pool(processes=n_workers) as pool:
        dfc_list = list(tqdm(pool.imap_unordered(_compute_dfc, dfc_task_args),
                             total=len(dfc_task_args), desc="DFC"))
    dfc = np.stack(dfc_list, axis=0)

    print(time_series.shape, dfc.shape)
    np.savez(output_path, timeseries=time_series, labels=labels, subject_id=subject_ids, hz=hz, dfc=dfc)


# ═══════════════════════════════════════════════════════════════════════════════
# 推理文本离线生成 (tech2.md §三-§五)
# ═══════════════════════════════════════════════════════════════════════════════

# 通道名称与排序 — 与 EDF 数据的 signal_header 严格对齐
# ['Fp1-AVG','F3-AVG','C3-AVG','P3-AVG','O1-AVG',
#  'Fp2-AVG','F4-AVG','C4-AVG','P4-AVG','O2-AVG',
#  'F7-AVG', 'T3-AVG','T5-AVG',
#  'F8-AVG', 'T4-AVG','T6-AVG',
#  'FZ-AVG', 'CZ-AVG','PZ-AVG']
EEG_19_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',       # 0-4:  左半球
    'Fp2', 'F4', 'C4', 'P4', 'O2',       # 5-9:  右半球
    'F7',  'T3', 'T5',                    # 10-12: 左颞
    'F8',  'T4', 'T6',                    # 13-15: 右颞
    'Fz',  'Cz', 'Pz',                    # 16-18: 中线
]

EEG_CHANNEL_GROUPS = {
    '额叶(含额极)':  {'channels': ['Fp1','Fp2','F3','F4','F7','F8','Fz'], 'indices': [0,5,1,6,10,13,16]},
    '颞叶':          {'channels': ['T3', 'T4', 'T5', 'T6'],                   'indices': [11, 14, 12, 15]},
    '中央区(SMN)':   {'channels': ['C3', 'Cz', 'C4'],                         'indices': [2, 17, 7]},
    '顶叶':          {'channels': ['P3', 'Pz', 'P4'],                         'indices': [3, 18, 8]},
    '枕叶(VIS)':     {'channels': ['O1', 'O2'],                               'indices': [4, 9]},
}

# 基于 10-20 系统标准解剖距离 (cm) 的电极间空间距离近似表
# 长程: >10cm (不同脑叶远距离), 中程: 5-10cm, 短程: <5cm (同区内或邻近区)
# 注：这是基于成人头围 ~56cm 的近似估算，实际距离因人而异
_ELECTRODE_DISTANCE = {
    # 短程 (<5cm) — 同区内或邻近区
    (0,10): 4.5, (0,16): 4.0, (0,5): 5.0,
    (1,10): 3.5, (1,2): 3.0, (1,16): 3.0, (1,6): 5.0,
    (2,11): 5.0, (2,3): 3.0, (2,17): 3.0, (2,7): 5.0,
    (3,12): 4.5, (3,18): 3.0, (3,8): 5.0,
    (4,12): 5.5, (4,9): 5.0,
    (5,13): 4.5, (5,16): 4.0,
    (6,13): 3.5, (6,7): 3.0, (6,16): 3.0,
    (7,14): 5.0, (7,8): 3.0, (7,17): 3.0,
    (8,15): 4.5, (8,18): 3.0,
    (9,15): 5.5,
    (10,11): 3.5, (10,13): 5.5, (10,16): 4.0,
    (11,12): 3.0, (11,14): 5.5,
    (12,15): 5.5, (12,3): 5.5, (12,4): 5.5,
    (13,14): 3.5, (13,15): 3.5, (13,16): 4.0,
    (14,15): 3.5,
    (15,8): 5.5,
    # 中程 (5-10cm) — 邻近脑叶间
    (0,11): 6.5, (0,1): 7.5, (0,2): 8.5,
    (1,11): 5.5, (1,12): 7.0,
    (2,12): 6.5, (2,1): 3.0,
    (3,4): 7.0,
    (4,3): 7.0, (4,18): 6.5,
    (5,14): 6.5, (5,6): 7.5, (5,7): 8.5,
    (6,14): 5.5, (6,15): 7.0,
    (7,15): 6.5,
    (8,9): 7.0,
    (9,18): 6.5,
    (11,3): 7.0, (11,18): 6.0,
    (12,18): 5.5,
    (14,8): 7.0, (14,18): 6.0,
    (15,18): 5.5,
    # 长程 (>10cm) — 跨多脑叶远距离
    (0,9): 15.0, (0,4): 13.0, (0,8): 12.0, (0,3): 11.0,
    (0,12): 11.0, (0,15): 13.0, (0,14): 11.0,
    (5,4): 15.0, (5,9): 13.0, (5,8): 12.0, (5,3): 11.0,
    (5,12): 13.0, (5,11): 11.0,
    (1,8): 10.5, (1,9): 11.0, (1,4): 10.5,
    (6,3): 10.5, (6,4): 10.5, (6,9): 11.0,
    (2,4): 11.0, (2,9): 12.0,
    (7,4): 12.0, (7,9): 11.0,
    (10,8): 11.5, (10,9): 12.5, (10,4): 11.0, (10,3): 10.5,
    (13,3): 11.5, (13,4): 12.5, (13,2): 10.5,
    (11,4): 11.0, (11,9): 12.0, (11,8): 10.5,
    (14,3): 10.5, (14,4): 11.0, (14,9): 12.0,
    (12,8): 10.5, (12,9): 11.0,
    (15,3): 10.5, (15,4): 11.0,
    (16,4): 11.0, (16,9): 11.0, (16,8): 10.5,
    (17,4): 12.0, (17,9): 11.0,
    (18,4): 11.0, (18,9): 10.5,
}

def _get_electrode_distance(i, j):
    """获取两个电极之间的近似解剖距离 (cm)。

    基于 10-20 系统标准电极间距近似值。同区内相邻电极 ~3cm，
    跨脑叶远距离 >10cm。未列入的对默认为中程 (~7cm)。
    """
    key = (min(i, j), max(i, j))
    return _ELECTRODE_DISTANCE.get(key, 7.0)


# 长程连接的判定阈值 (>10cm)
_LONG_RANGE_DISTANCE_CM = 10.0

# 半球间同源连接 (左右对称通道对，不含中线 Fz/Cz/Pz)
HOMOLOGOUS_PAIRS = [
    ('Fp1', 'Fp2', 0, 5),
    ('F3',  'F4',  1, 6),
    ('C3',  'C4',  2, 7),
    ('P3',  'P4',  3, 8),
    ('O1',  'O2',  4, 9),
    ('F7',  'F8',  10, 13),
    ('T3',  'T4',  11, 14),
    ('T5',  'T6',  12, 15),
]

# 左/右半球通道索引 (不含中线 Fz/Cz/Pz)
LEFT_HEMI_INDICES  = [0, 1, 2, 3, 4, 10, 11, 12]   # Fp1, F3, C3, P3, O1, F7, T3, T5
RIGHT_HEMI_INDICES = [5, 6, 7, 8, 9, 13, 14, 15]   # Fp2, F4, C4, P4, O2, F8, T4, T6

# 中文类别名 (独立于 LDDE2th.py 的 CLASS_NAMES_ZH，供离线生成独立运行)
CLASS_NAMES_ZH_STANDALONE = {0: "正常认知", 1: "主观记忆障碍", 2: "轻度认知障碍", 3: "阿尔茨海默病"}


def _channel_idx_to_name(idx):
    """通道索引 → 通道名。"""
    return EEG_19_CHANNELS[idx]


def _get_network_for_channel(idx):
    """返回通道所属的网络名称。"""
    for net_name, info in EEG_CHANNEL_GROUPS.items():
        if idx in info['indices']:
            return net_name
    return "未知"


def _get_edge_classification(i, j):
    """判断连接边 (i, j) 的类型。

    Returns:
        dict with keys:
          - 'type': 'intra' | 'cross' | 'homologous'
          - 'net_i': 通道 i 的网络名
          - 'net_j': 通道 j 的网络名
          - 'is_homologous': bool
    """
    net_i = _get_network_for_channel(i)
    net_j = _get_network_for_channel(j)

    # 检查是否半球间同源
    is_homologous = False
    for _, _, hi, hj in HOMOLOGOUS_PAIRS:
        if (i == hi and j == hj) or (i == hj and j == hi):
            is_homologous = True
            break

    if net_i == net_j:
        edge_type = 'intra'
    else:
        edge_type = 'cross'

    return {
        'type': edge_type,
        'net_i': net_i,
        'net_j': net_j,
        'is_homologous': is_homologous,
    }


def compute_dfc_stats(ts, hz):
    """计算 DFC 时间统计量：时间均值 + 时间变异性（标准差）。

    使用与训练时一致的滑动窗口参数（3 秒窗、1 秒步长），
    保证离线生成的推理文本与模型实际接收的 DFC 信息对齐。

    Args:
        ts: (19, T) numpy array — EEG time series
        hz: int — 采样率 (Hz)

    Returns:
        dfc_mean: (19, 19) — 跨窗口时间平均 FC
        dfc_std:  (19, 19) — 跨窗口 FC 标准差（时间变异性）
        num_windows: int — 滑动窗口数
    """
    dfc = dynamic_connectivity(ts, 3 * hz, 1 * hz)  # (L, 19, 19)
    dfc_mean = dfc.mean(axis=0)
    dfc_std = dfc.std(axis=0)
    return dfc_mean, dfc_std, dfc.shape[0]


def _compute_graph_metrics(fc, density=0.25):
    """计算加权图论指标：聚类系数和特征路径长度。

    使用比例阈值（保留 |r| 最大的 top density 边）而非 r>0 二值化，
    避免因 EEG 容积传导导致的正相关偏倚使图过度密集而丧失区分能力。

    Args:
        fc:      (N, N) 功能连接矩阵
        density: float — 保留的边密度比例（默认 0.25 = top 25%）

    Returns:
        (clustering_coef, characteristic_path_length)
    """
    N = fc.shape[0]
    fc_abs = np.abs(fc.copy())
    np.fill_diagonal(fc_abs, 0)

    # ── 比例阈值：每行保留 top density 的边 ──
    k = max(1, int(np.ceil(N * density)))
    adj = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        row = fc_abs[i]
        # 注意排除自连接
        top_k_idx = np.argpartition(-row, k)[:k]
        for j in top_k_idx:
            if row[j] > 0:
                adj[i, j] = row[j]   # 加权边
    # 对称化
    adj = (adj + adj.T) / 2

    # ── 加权聚类系数 (Onnela 2005) ──
    # C_i = (sum_{j,k} (w_ij * w_jk * w_ki)^(1/3)) / (k_i * (k_i - 1))
    degrees = (adj > 0).sum(axis=1).astype(np.float64)
    C_per_node = np.zeros(N)
    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) < 2:
            C_per_node[i] = 0.0
            continue
        tri_sum = 0.0
        for a_idx, j in enumerate(neighbors):
            for k in neighbors[a_idx + 1:]:
                if adj[j, k] > 0:
                    tri_sum += np.cbrt(adj[i, j] * adj[j, k] * adj[k, i])
        denom = degrees[i] * (degrees[i] - 1)
        C_per_node[i] = tri_sum / denom if denom > 0 else 0.0

    valid = degrees >= 2
    C = C_per_node[valid].mean() if valid.any() else 0.0

    # ── 加权特征路径长度 ──
    # 距离矩阵：1 / |r|（有边）或 inf（无边）
    dist = np.full((N, N), np.inf)
    np.fill_diagonal(dist, 0)
    for i in range(N):
        for j in range(N):
            if adj[i, j] > 0:
                dist[i, j] = 1.0 / adj[i, j]  # 高 FC → 短距离

    # Floyd-Warshall
    for k in range(N):
        dk = dist[:, k:k+1] + dist[k:k+1, :]
        dist = np.minimum(dist, dk)

    finite = dist[np.isfinite(dist)]
    L = finite.mean() if len(finite) > 0 else 0.0

    return float(C), float(L)


def get_channel_pairs_top_bottom(fc, fc_std=None, k=8):
    """从 FC 矩阵中提取 Top-k 最强和 Bottom-k 最弱连接边。

    Args:
        fc:     (19, 19) 功能连接矩阵（DFC 时间均值或 SFC）
        fc_std: (19, 19) or None — DFC 时间标准差，可选
        k:      返回前后各 k 条边

    Returns:
        top:    list of dict — 最强的 k 条边，含通道名、FC 值、分类、时间变异性
        bottom: list of dict — 最弱的 k 条边
    """
    N = fc.shape[0]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            val = fc[i, j]
            classification = _get_edge_classification(i, j)
            dist_cm = _get_electrode_distance(i, j)
            is_long_range = (dist_cm > _LONG_RANGE_DISTANCE_CM
                             and classification['type'] == 'cross')
            # 时间变异性
            edge_std = float(fc_std[i, j]) if fc_std is not None else None
            cv = edge_std / abs(val) if (edge_std is not None and abs(val) > 1e-6) else None
            edges.append({
                'i': i, 'j': j,
                'ch_i': _channel_idx_to_name(i),
                'ch_j': _channel_idx_to_name(j),
                'fc': float(val),
                'fc_std': edge_std,
                'cv': cv,
                'distance_cm': dist_cm,
                'is_long_range': is_long_range,
                **classification,
            })

    edges_sorted = sorted(edges, key=lambda e: e['fc'], reverse=True)
    top = edges_sorted[:k]
    bottom = edges_sorted[-k:][::-1]  # reverse so weakest first
    return top, bottom


def fc_to_text_description(fc, class_name, fc_std=None, num_windows=None):
    """将功能连接矩阵转换为含精确数字的结构化中文描述。

    当 fc_std 和 num_windows 可用时（DFC 模式），描述基于 DFC 时间统计量；
    否则保持向后兼容的 SFC 模式。

    这是 DeepSeek 推理生成的输入，包含所有定量证据，
    使 DeepSeek 能基于具体数字进行医学推理。

    Args:
        fc:          (19, 19) numpy array — FC 矩阵（DFC 时间均值 或 SFC）
        class_name:  str — 该样本的类别名称
        fc_std:      (19, 19) or None — DFC 时间标准差（可选）
        num_windows: int or None — 滑动窗口数（可选）

    Returns:
        str — ~800-1200 tokens 的结构化中文描述
    """
    N = fc.shape[0]
    fc = fc.copy()
    np.fill_diagonal(fc, 0.0)  # 排除自连接

    has_dfc = (fc_std is not None and num_windows is not None)

    # ── 提取上三角（i < j）所有连接对 ──
    triu_idx = np.triu_indices(N, k=1)
    all_edges = fc[triu_idx]  # 171 个值

    mean_fc = float(np.mean(all_edges))
    std_fc = float(np.std(all_edges))
    pos_ratio = float(np.mean(all_edges > 0)) * 100
    strong_ratio = float(np.mean(np.abs(all_edges) > 0.5)) * 100
    C_global, L_char = _compute_graph_metrics(fc)

    # 时间变异性全局统计（仅 DFC 模式）
    if has_dfc:
        fc_std_copy = fc_std.copy()
        np.fill_diagonal(fc_std_copy, 0.0)
        avg_std = float(np.mean(fc_std_copy[triu_idx]))

    lines = []

    # ── 标题 ──
    lines.append("脑网络功能连接分析（19通道，10-20国际标准系统）：")
    lines.append("【通道分组】额叶(Fp1,Fp2,F3,F4,F7,F8,Fz)、颞叶(T3,T4,T5,T6)、")
    lines.append("   中央区(C3,Cz,C4)、顶叶(P3,Pz,P4)、枕叶(O1,O2)，共5组。")
    lines.append("")

    # ── 全局统计 ──
    lines.append("【全局统计】")
    if has_dfc:
        lines.append(f"- DFC时间平均连接强度：{mean_fc:.3f}"
                     f"（{num_windows}个滑动窗口，窗长3s步长1s，171个连接对的时间平均r值）")
        lines.append(f"- DFC时间平均标准差：{std_fc:.3f}（连接对间差异）")
        lines.append(f"- 平均时间变异性：{avg_std:.3f}（所有连接对跨窗口FC标准差的均值）")
    else:
        lines.append(f"- 平均功能连接强度：{mean_fc:.3f}（所有171个连接对的Pearson r均值）")
        lines.append(f"- 功能连接标准差：{std_fc:.3f}")
    lines.append(f"- 正连接比例：{pos_ratio:.1f}%（r>0）")
    lines.append(f"- 强连接比例：{strong_ratio:.1f}%（|r|>0.5）")
    if has_dfc:
        lines.append(f"- 加权聚类系数：{C_global:.3f}（基于DFC时间平均矩阵，比例阈值 top 25% |r|，Onnela加权）")
        lines.append(f"- 加权特征路径长度：{L_char:.2f}（基于DFC时间平均矩阵，距离=1/|r|）")
    else:
        lines.append(f"- 加权聚类系数：{C_global:.3f}（比例阈值 top 25% |r|，Onnela加权）")
        lines.append(f"- 加权特征路径长度：{L_char:.2f}（距离=1/|r|）")
    lines.append("")

    # ── 网络内部平均连接 ──
    lines.append("【网络内部平均连接】")
    group_names = ['额叶(含额极)', '颞叶', '中央区(SMN)', '顶叶', '枕叶(VIS)']
    intra_vals = {}
    for gname in group_names:
        indices = EEG_CHANNEL_GROUPS[gname]['indices']
        vals = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                vals.append(fc[indices[a], indices[b]])
        intra_val = float(np.mean(vals)) if vals else 0.0
        intra_vals[gname] = intra_val
        short = gname.split('(')[0]  # e.g. "额叶" or "中央区"
        if 'SMN' in gname:
            label = f"{short}/感觉运动区内"
        elif 'VIS' in gname:
            label = f"{short}/视觉区内"
        else:
            label = f"{short}区内"
        lines.append(f"- {label}：{intra_val:.3f}")
    lines.append("")

    # ── 跨网络平均连接 ──
    lines.append("【跨网络平均连接（网络间耦合）】")
    cross_lines = []
    for gi in range(len(group_names)):
        for gj in range(gi + 1, len(group_names)):
            gni = group_names[gi]
            gnj = group_names[gj]
            idx_i = EEG_CHANNEL_GROUPS[gni]['indices']
            idx_j = EEG_CHANNEL_GROUPS[gnj]['indices']
            vals = [fc[i, j] for i in idx_i for j in idx_j]
            cross_val = float(np.mean(vals))
            sni = gni.split('(')[0]
            snj = gnj.split('(')[0]
            cross_lines.append(f"{sni}-{snj}：{cross_val:.3f}")

    # 分行输出，每行 3 对
    for row_start in range(0, len(cross_lines), 3):
        chunk = cross_lines[row_start:row_start + 3]
        lines.append("- " + " | ".join(chunk))
    lines.append("")

    # ── 半球间同源连接 ──
    lines.append("【半球间同源连接（左右对称通道对）】")
    homo_parts = []
    for ch_i, ch_j, hi, hj in HOMOLOGOUS_PAIRS:
        val = float(fc[hi, hj])
        homo_parts.append(f"{ch_i}-{ch_j}：{val:.3f}")
    # 分行，每行 4 对
    for row_start in range(0, len(homo_parts), 4):
        chunk = homo_parts[row_start:row_start + 4]
        lines.append("- " + " | ".join(chunk))
    lines.append("")

    # ── Top-8 最强连接边 ──
    top8, bottom8 = get_channel_pairs_top_bottom(fc, fc_std=fc_std, k=8)
    lines.append("【连接强度排名 — Top-8 最强连接边】")
    for rank, e in enumerate(top8, 1):
        ci, cj = e['ch_i'], e['ch_j']
        val = e['fc']
        # 分类标签
        if e['is_homologous']:
            tag = "半球间同源"
        elif e['type'] == 'intra':
            tag = f"{e['net_i'].split('(')[0]}区内"
        else:
            tag = f"{e['net_i'].split('(')[0]}-{e['net_j'].split('(')[0]}，跨网络"
        # 时间变异性（仅 DFC 模式）
        if has_dfc and e['fc_std'] is not None:
            lines.append(f"{rank}. {ci}-{cj}（{tag}）：{val:.3f}，时间σ={e['fc_std']:.3f}")
        else:
            lines.append(f"{rank}. {ci}-{cj}（{tag}）：{val:.3f}")
    lines.append("")

    # ── Bottom-8 最弱连接边 ──
    lines.append("【连接强度排名 — Bottom-8 最弱连接边】")
    for rank, e in enumerate(bottom8, 1):
        ci, cj = e['ch_i'], e['ch_j']
        val = e['fc']
        if e['is_homologous']:
            tag = "半球间同源"
        elif e['type'] == 'intra':
            tag = f"{e['net_i'].split('(')[0]}区内"
        else:
            net_label = f"{e['net_i'].split('(')[0]}-{e['net_j'].split('(')[0]}，跨网络"
            if e['is_long_range']:
                tag = f"{net_label}，长程—正常空间衰减"
            else:
                tag = f"{net_label}，近-中程"
        # 时间变异性（仅 DFC 模式）
        if has_dfc and e['fc_std'] is not None:
            lines.append(f"{rank}. {ci}-{cj}（{tag}）：{val:.3f}，时间σ={e['fc_std']:.3f}")
        else:
            lines.append(f"{rank}. {ci}-{cj}（{tag}）：{val:.3f}")
    lines.append("注：跨脑叶长程弱连接（如Fp1-O2、T3-O1）多为正常空间衰减，")
    lines.append("其FC接近零不必然代表病理失连接，推理时需与同区内或近距弱连接区分。")
    lines.append("")

    # ── 关键临床连接：DMN后部相关 (tech2.md §三) ──
    # 这些连接对在 AD/MCI 文献中一致报道为敏感的电生理标志物
    lines.append("【关键临床连接 — DMN后部相关】")
    dmn_pairs = [
        ('左后颞-顶叶(T5-P3)',      12, 3),
        ('右后颞-顶叶(T6-P4)',      15, 8),
        ('顶叶中线-枕叶(Pz-O1)',    18, 4),
        ('左颞叶-顶叶中线(T5-Pz)',  12, 18),
        ('右颞叶-顶叶中线(T6-Pz)',  15, 18),
    ]
    for label, ai, bi in dmn_pairs:
        if has_dfc:
            val_std = fc_std[ai, bi]
            lines.append(f"- {label}：{fc[ai, bi]:.3f}，时间σ={val_std:.3f}")
        else:
            lines.append(f"- {label}：{fc[ai, bi]:.3f}")
    lines.append("")

    # ── 连接的时间稳定性（仅 DFC 模式）──
    if has_dfc:
        lines.append("【连接的时间稳定性 — 跨窗口DFC波动分析】")
        # 收集所有 |μ| > 0.05 的连接对，按 CV 排序
        stable_edges = []
        for i in range(N):
            for j in range(i + 1, N):
                mu = fc[i, j]
                sigma = fc_std[i, j]
                if abs(mu) > 0.05 and sigma > 0:
                    cv = sigma / abs(mu)
                    classification = _get_edge_classification(i, j)
                    tag = ""
                    if classification['is_homologous']:
                        tag = "半球间同源"
                    elif classification['type'] == 'intra':
                        tag = f"{classification['net_i'].split('(')[0]}区内"
                    else:
                        tag = f"{classification['net_i'].split('(')[0]}-{classification['net_j'].split('(')[0]}，跨网络"
                    stable_edges.append({
                        'i': i, 'j': j,
                        'ch_i': _channel_idx_to_name(i),
                        'ch_j': _channel_idx_to_name(j),
                        'mu': mu, 'sigma': sigma, 'cv': cv,
                        'tag': tag,
                    })

        if stable_edges:
            stable_edges.sort(key=lambda e: e['cv'])

            lines.append("全脑最稳定连接（CV最小，|μ|>0.05，Top-5）：")
            for rank, e in enumerate(stable_edges[:5], 1):
                lines.append(f"  {rank}. {e['ch_i']}-{e['ch_j']}（{e['tag']}）："
                             f"μ={e['mu']:.3f}, σ={e['sigma']:.3f}, CV={e['cv']:.3f}")

            lines.append("全脑最不稳定连接（CV最大，|μ|>0.05，Top-5）：")
            for rank, e in enumerate(stable_edges[-5:][::-1], 1):
                lines.append(f"  {rank}. {e['ch_i']}-{e['ch_j']}（{e['tag']}）："
                             f"μ={e['mu']:.3f}, σ={e['sigma']:.3f}, CV={e['cv']:.3f}")

        # 网络级时间变异性
        lines.append("网络级平均时间变异性（σ均值）：")
        net_std_parts = []
        for gname in group_names:
            indices = EEG_CHANNEL_GROUPS[gname]['indices']
            # 区内 σ
            intra_stds = []
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    intra_stds.append(fc_std[indices[a], indices[b]])
            intra_std_avg = float(np.mean(intra_stds)) if intra_stds else 0.0
            short = gname.split('(')[0]
            if 'SMN' in gname:
                label = f"{short}/SMN区内"
            elif 'VIS' in gname:
                label = f"{short}/VIS区内"
            else:
                label = f"{short}区内"
            net_std_parts.append(f"{label} σ={intra_std_avg:.3f}")

        # 跨网络 σ（关键对）
        for gni, gnj, label_cn in [
            ('颞叶', '顶叶', '颞叶-顶叶'),
            ('额叶(含额极)', '顶叶', '额叶-顶叶'),
            ('颞叶', '枕叶(VIS)', '颞叶-枕叶'),
        ]:
            idx_i = EEG_CHANNEL_GROUPS[gni]['indices']
            idx_j = EEG_CHANNEL_GROUPS[gnj]['indices']
            cross_stds = [fc_std[i, j] for i in idx_i for j in idx_j]
            cross_std_avg = float(np.mean(cross_stds)) if cross_stds else 0.0
            net_std_parts.append(f"{label_cn}跨网络 σ={cross_std_avg:.3f}")

        for row_start in range(0, len(net_std_parts), 3):
            chunk = net_std_parts[row_start:row_start + 3]
            lines.append("- " + " | ".join(chunk))
        lines.append("")

    # ── 左右半球连通性对比 ──
    left_vals = [fc[i, j] for i in LEFT_HEMI_INDICES for j in LEFT_HEMI_INDICES if i < j]
    right_vals = [fc[i, j] for i in RIGHT_HEMI_INDICES for j in RIGHT_HEMI_INDICES if i < j]
    inter_vals = [fc[i, j] for i in LEFT_HEMI_INDICES for j in RIGHT_HEMI_INDICES]
    # 左/右半球颞-顶连接 (T3-P3=11-3, T5-P3=12-3, T3-Pz=11-18; right: T4-P4=14-8, T6-P4=15-8, T4-Pz=14-18)
    left_tp = [fc[11, 3], fc[12, 3], fc[11, 18]]
    right_tp = [fc[14, 8], fc[15, 8], fc[14, 18]]

    lines.append("【左右半球连通性对比】")
    lines.append(f"- 左半球平均连接：{np.mean(left_vals):.3f} | 右半球平均连接：{np.mean(right_vals):.3f}")
    lines.append(f"- 半球间平均连接：{np.mean(inter_vals):.3f}")
    lines.append(f"- 左半球颞-顶连接：{np.mean(left_tp):.3f} | 右半球颞-顶连接：{np.mean(right_tp):.3f}")
    lines.append("")

    # ── 类别标签 ──
    lines.append(f"该数据所属类别：{class_name}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeek Prompt 模板 (tech2.md §四)
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_prompt_ds():
    """构建 DeepSeek 系统提示（神经影像专家知识）。

    Returns:
        str — 约 900 中文字符的领域知识基准
    """
    return """你是一位精通脑电图(EEG)功能连接分析的神经影像专家，拥有超过20年的临床神经生理学经验。你擅长根据19通道EEG功能连接矩阵识别痴呆亚型的特征性脑网络改变模式。

重要声明：19通道头皮EEG的空间分辨率约为5-7cm，无法直接定位海马、后扣带回等深部结构。以下fMRI文献中关于DMN/PCC的描述仅作为推理参考框架，不可声称直接观测到深部结构。你的推理应基于可以可靠测量的皮层表面连接模式：额叶、颞叶、顶叶、枕叶及感觉运动皮层之间的功能连接(FC)。

你的领域知识基准：
- 阿尔茨海默病(AD)的典型FC改变：
  * DMN后部连接显著减弱（顶叶-颞叶跨网络连接降低，是AD最一致的电生理标志物）
  * 颞叶内部连接减弱，顶叶中线-枕叶功能连接下降
  * 感觉运动网络(SMN)内部连接相对保留（区别于其他类型痴呆的关键特征）
  * 额叶内部连接可能出现代偿性增强（AD早期的额叶代偿假说）
  * 半球间同源连接相对保留，尤其在中央区和枕叶
  * 注意：长程跨脑叶弱连接（如Fp1-O2）为正常空间衰减，不应解读为病理

- 轻度认知障碍(MCI)的典型FC改变：
  * DMN后部连接轻度至中度降低（介于正常与AD之间）
  * 额顶网络连接出现代偿性激活增强
  * 部分长程连接保留，局部网络内连接可能出现代偿性增强
  * 颞叶-顶叶连接降低幅度小于典型AD
  * 半球间同源连接大多正常或轻微降低

- 主观记忆障碍(SMI)的典型FC改变：
  * 整体FC模式与正常认知接近，网络拓扑仍较完整
  * DMN后部连接可能出现轻微下降（未达MCI/AD水平）
  * 额叶执行网络可能出现早期代偿性增强
  * 颞叶-顶叶连接基本正常
  * 全脑功能连接整合度可能轻微下降但无显著性差异

- 正常认知的典型FC特征：
  * 全脑功能连接整合良好，无显著异常
  * DMN/SN/CEN三大核心网络内部连接强度适中
  * 长程与短程连接分布均衡，无特定脑区出现显著失连接
  * 各网络间耦合模式正常，无代偿性异常增强或减弱

- 动态功能连接(DFC)的时间变异性具有临床意义：
  * AD患者的DMN后部连接不仅时间平均强度降低，而且跨窗口波动性（时间σ）增大，
    反映神经活动的非平稳性增加和网络稳定性下降
  * 正常认知的连接模式在时间上更为稳定，关键网络连接的变异系数(CV)通常较低
  * 代偿性增强的连接（如MCI阶段的额顶网络）可能呈现较高的时间变异性，
    反映代偿机制本身的不稳定性——这种"高均值+高波动"组合是MCI的特征标志
  * 感觉运动网络(SMN)的连接在AD中不仅时间均值保留，时间稳定性也较高
  * 注意：短窗口(3s)的DFC估计本身有一定噪声，时间σ解读时需关注
    相对差异（哪些连接异常波动）而非绝对值"""


def build_task_prompt_ds(fc_description, class_name):
    """构建 DeepSeek 任务提示，拼接在 FC 描述之后。

    Args:
        fc_description: str — fc_to_text_description() 的输出
        class_name:     str — 类别名称

    Returns:
        str — 完整的 user prompt
    """
    task = f"""下面是从19通道EEG数据计算得到的动态功能连接(DFC)矩阵的时间统计描述（含时间均值与时间变异性）。请作为神经影像专家，基于这些DFC数据中的具体数值，逐步推理并诊断该患者的痴呆类型。

请自然展开推理，根据该样本的数据特点灵活组织段落顺序，无需遵循固定模板。推理中应涵盖以下要素（作为检查清单而非顺序约束）：

- 识别该DFC矩阵中最显著的连接模式——以该样本自身的平均FC强度为内部参照，明确指出哪些网络内或网络间的连接相对增强或减弱，哪些连接处于正常水平。注意区分"长程弱连接（正常空间衰减）"与"近距弱连接（可能具有病理意义）"。
- 逐网络分析功能状态：DMN后部（顶叶-颞叶连接，参考关键临床连接section）完整性如何、感觉运动网络是否保留、额叶是否出现代偿性增强、视觉网络完整性、半球间同源连接整体状况。
- 分析连接的时间稳定性：哪些连接的跨窗口波动异常增大或减小？时间变异性（CV、σ）的高/低模式是否符合特定疾病特征？是否存在"高均值+高波动"的代偿不稳定模式或"低均值+高波动"的失连接恶化模式？
- 将观察到的连接模式（含时间稳定性特征）与AD、MCI、SMI的典型FC特征进行逐项对比，指出最匹配的疾病模式以及与其他类别的不一致之处。
- 给出诊断结论。

要求（按重要性排序）：
- 输出严格不超过300个中文字符，超出将被自动截断导致结尾不完整
- 禁止出现任何具体数字：不能引用FC数值（如0.342）、百分比（如32.2%）、σ值、CV值或任何其他数字。全部用"较强/较弱/正常/轻微/显著"等定性词描述
- 诊断结论必须为：{class_name}
- 不要出现markdown格式标记，不要用"1."、"2."等编号前缀，写成连贯的自然段

{fc_description}"""

    return task


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeek API 调用与批量生成 (tech2.md §八-§九)
# ═══════════════════════════════════════════════════════════════════════════════

# ── DeepSeek API 配置 ──
DEEPSEEK_API_KEY = "sk-your-key-here"  # TODO: 替换为实际 API key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 多温度采样配置
TEMPERATURES = [0.3, 0.6, 0.9]  # N=3
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # 指数退避 (秒)
CHECKPOINT_INTERVAL = 50   # 每 50 个样本保存一次中间结果
RATE_LIMIT_SLEEP = 1.5     # 批次间等待 (秒)


def _call_deepseek_api(messages, temperature, api_key, base_url, model, max_tokens=500):
    """单次调用 DeepSeek API，含重试逻辑。

    Args:
        messages:    list[dict] — [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        temperature: float
        api_key:     str
        base_url:    str
        model:       str
        max_tokens:  int — 输出 token 上限 (默认 500, ~300 中文字符)

    Returns:
        dict — {"text": str, "finish_reason": str, "error": str or None}
               text 永远不为 None（失败时为空字符串）
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = response.choices[0]
            finish_reason = choice.finish_reason or "unknown"
            content = choice.message.content

            # content 为 None：安全过滤或模型拒绝
            if content is None:
                return {"text": "", "finish_reason": finish_reason,
                        "error": f"content is None (finish_reason={finish_reason})"}

            text = content.strip()

            # 空字符串：模型未生成任何内容
            if not text:
                return {"text": "", "finish_reason": finish_reason,
                        "error": f"empty response (finish_reason={finish_reason})"}

            # finish_reason="length" 警告：文本被截断
            if finish_reason == "length":
                print(f"[WARN] finish_reason=length (truncated at {max_tokens} tokens)",
                      end=' ', flush=True)

            return {"text": text, "finish_reason": finish_reason, "error": None}

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"[WARN] API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                print(f"       Retrying in {delay}s...", end=' ', flush=True)
                time.sleep(delay)
            else:
                print(f"[ERROR] API call failed after {MAX_RETRIES} attempts: {e}",
                      end=' ', flush=True)
                return {"text": "", "finish_reason": "error",
                        "error": f"exception after {MAX_RETRIES} retries: {e}"}


def generate_reasoning_texts(fc_matrices, class_names, sample_indices=None,
                              api_key=None, base_url=None, model=None,
                              temperatures=None,
                              fc_stds=None, num_windows_list=None,
                              checkpoint_dir=None):
    """批量调用 DeepSeek API 生成推理文本。

    每条样本以 N 种 temperature 生成 N 条推理文本，
    支持断点续传、速率控制和自动重试。

    Args:
        fc_matrices:      (N_total, 19, 19) numpy array — 所有样本的 FC 矩阵（DFC 时间均值）
        class_names:      list[str] — 每条样本的类别名称
        sample_indices:   list[int] — 要处理的样本索引（None = 全部）
        api_key:          str — DeepSeek API key
        base_url:         str — API base URL
        model:            str — 模型名称
        temperatures:     list[float] — temperature 列表（默认 [0.3, 0.6, 0.9]）
        fc_stds:          (N_total, 19, 19) or None — DFC 时间标准差（可选）
        num_windows_list: list[int] or None — 每个样本的滑动窗口数（可选）
        checkpoint_dir:   str — 中间结果保存目录

    Returns:
        dict — {str(idx): [text_t1, text_t2, text_t3], ...}
    """
    if api_key is None:
        api_key = os.environ.get('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY)
    if base_url is None:
        base_url = os.environ.get('DEEPSEEK_BASE_URL', DEEPSEEK_BASE_URL)
    if model is None:
        model = os.environ.get('DEEPSEEK_MODEL', DEEPSEEK_MODEL)
    if temperatures is None:
        temperatures = TEMPERATURES
    if sample_indices is None:
        sample_indices = list(range(len(fc_matrices)))

    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not set. Please set the environment variable:\n"
            "  export DEEPSEEK_API_KEY=your_key_here"
        )

    system_prompt = build_system_prompt_ds()
    results = {}
    prompts_log = {}  # 收集所有 prompt 用于调试，最后写入标准 JSON

    # ── 断点续传：尝试加载已有中间结果 ──
    checkpoint_path = None
    start_idx_offset = 0
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'reasoning_texts_checkpoint.json')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            N = len(temperatures)
            # 检测并清理不完整的条目（上次运行可能在某个 temperature 中途崩溃）
            incomplete_keys = []
            for k, texts in results.items():
                # 兼容旧格式（list of strings）和新格式（dict with "texts" key）
                text_list = texts if isinstance(texts, list) else texts.get("texts", [])
                if len(text_list) < N or any(not t or not t.strip() for t in text_list):
                    incomplete_keys.append(k)
            for k in incomplete_keys:
                del results[k]

            completed = set(int(k) for k in results.keys())
            # 跳过已完成的样本
            sample_indices = [i for i in sample_indices if i not in completed]
            start_idx_offset = len(results)           # 已完成的样本数，用于进度显示

            if incomplete_keys:
                print(f"[断点续传] 已加载 {len(results)} 条完整结果，"
                      f"清理 {len(incomplete_keys)} 条不完整记录，"
                      f"剩余 {len(sample_indices)} 条待处理")
            else:
                print(f"[断点续传] 已加载 {len(results)} 条已有结果，"
                      f"剩余 {len(sample_indices)} 条待处理")

    N = len(temperatures)
    total_requests = len(sample_indices) * N
    print(f"开始生成推理文本：{len(sample_indices)} 样本 × {N} temperatures "
          f"= {total_requests} 次 API 调用")
    print(f"模型: {model}")

    # 统计各类 finish_reason
    reason_stats = {}

    for idx_count, sample_idx in enumerate(sample_indices):
        fc = fc_matrices[sample_idx]
        cls_name = class_names[sample_idx]

        # 透传 DFC 时间变异性统计量
        fc_std = fc_stds[sample_idx] if fc_stds is not None else None
        nw = num_windows_list[sample_idx] if num_windows_list is not None else None

        fc_desc = fc_to_text_description(fc, cls_name, fc_std=fc_std, num_windows=nw)
        task_prompt = build_task_prompt_ds(fc_desc, cls_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        sample_texts = []
        sample_meta = []  # per-temperature metadata for debugging
        for t_idx, temp in enumerate(temperatures):
            global_idx = start_idx_offset + idx_count
            print(f"  [{global_idx + 1}/{start_idx_offset + len(sample_indices)}] "
                  f"样本 {sample_idx}, T={temp:.1f}...", end=' ', flush=True)

            result = _call_deepseek_api(messages, temp, api_key, base_url, model)
            text = result["text"]
            finish_reason = result["finish_reason"]

            # 统计
            reason_stats[finish_reason] = reason_stats.get(finish_reason, 0) + 1

            sample_meta.append({
                "temperature": temp,
                "finish_reason": finish_reason,
                "error": result.get("error"),
                "text_len": len(text),
            })

            if text:
                sample_texts.append(text)
                status = "OK" if not result.get("error") else f"OK ({finish_reason})"
                print(f"{status} ({len(text)} chars)")
            else:
                sample_texts.append("")
                err = result.get("error", "unknown")
                print(f"EMPTY [{finish_reason}] {err}")

            # 速率控制
            time.sleep(RATE_LIMIT_SLEEP)

        results[str(sample_idx)] = sample_texts

        # ── 收集 prompt 用于调试 ──
        prompts_log[str(sample_idx)] = {
            "class_name": cls_name,
            "system_prompt": messages[0]["content"],
            "user_prompt": messages[1]["content"],
            "results": sample_meta,
        }

        # ── 定期保存 checkpoint ──
        if checkpoint_path and (idx_count + 1) % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [Checkpoint] {len(results)} 条结果已保存至 {checkpoint_path}")

    # ── 最终统计 ──
    if reason_stats:
        total = sum(reason_stats.values())
        print(f"\nfinish_reason 分布 ({total} 次调用):")
        for reason, count in sorted(reason_stats.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

    # ── 最终保存 ──
    if checkpoint_path:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[完成] 全部 {len(results)} 条结果已保存至 {checkpoint_path}")
        # 保存 prompts 调试文件（标准 JSON 格式，可在编辑器中直接打开）
        prompts_path = os.path.join(checkpoint_dir, 'reasoning_prompts.json')
        with open(prompts_path, 'w', encoding='utf-8') as f:
            json.dump(prompts_log, f, ensure_ascii=False, indent=2)
        print(f"[完成] Prompts 调试文件已保存至 {prompts_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 三阶段推理文本生成 — 阶段一：DFC 预计算 (tech2.md §十)
# ═══════════════════════════════════════════════════════════════════════════════

def dementia_dfc_precompute(data_path="../data/Dementia4000/Dementia4000.npz",
                            output_path="../data/Dementia4000/dfc_descriptions.npz",
                            max_samples=None):
    """阶段一：DFC 时间统计量 + 文本描述一次性预计算。

    纯本地计算（无 API 调用）。使用多进程并行，结果保存为 .npz 文件，
    供阶段二的 dementia_reasoning_generate() 读取。

    运行一次即可。

    Args:
        data_path:   str — Dementia4000.npz 路径
        output_path: str — 输出 .npz 路径
        max_samples: int or None — 限制处理的样本数（调试用）
    """
    print("=" * 60)
    print("  Stage 1: DFC Precompute + Text Description")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1/3] Loading: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    time_series_all = data['timeseries']   # (N, 19, T)
    labels_all = data['labels']            # (N,) int
    hz = int(data['hz'])

    N_total = len(time_series_all)
    print(f"  Total samples: {N_total}")

    if max_samples is not None and max_samples < N_total:
        time_series = time_series_all[:max_samples]
        labels = labels_all[:max_samples]
        N = max_samples
        print(f"  → Processing first {N} (max_samples={max_samples})")
    else:
        time_series = time_series_all
        labels = labels_all
        N = N_total

    # 2. 类别索引 → 中文名
    class_names = [CLASS_NAMES_ZH_STANDALONE.get(int(l), f"Class{int(l)}")
                   for l in labels]

    # 3. 多进程并行：DFC 计算 + 文本描述生成
    print(f"\n[2/3] Computing DFC + text descriptions for {N} samples...")
    n_workers = min(cpu_count(), 8)
    print(f"  Workers: {n_workers}")

    task_args = [(i, time_series[i], hz, class_names[i]) for i in range(N)]

    fc_means = np.zeros((N, 19, 19), dtype=np.float32)
    fc_stds = np.zeros((N, 19, 19), dtype=np.float32)
    num_windows_list = np.zeros(N, dtype=np.int32)
    fc_descriptions = np.empty(N, dtype=object)

    with Pool(processes=n_workers) as pool:
        for idx, fc_mean, fc_std, nw, fc_desc in tqdm(
            pool.imap_unordered(_compute_dfc_description_worker, task_args),
            total=N, desc="DFC+Desc"
        ):
            fc_means[idx] = fc_mean
            fc_stds[idx] = fc_std
            num_windows_list[idx] = nw
            fc_descriptions[idx] = fc_desc

    # 4. 保存
    print(f"\n[3/3] Saving to: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    np.savez(output_path,
             fc_means=fc_means,
             fc_stds=fc_stds,
             num_windows=num_windows_list,
             fc_descriptions=fc_descriptions,
             class_names=np.array(class_names, dtype=object))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Samples: {N}")
    avg_desc_len = sum(len(str(d)) for d in fc_descriptions) / max(N, 1)
    print(f"  Avg description length: {avg_desc_len:.0f} chars")
    print(f"\n  ✓ Stage 1 complete!")
    print(f"  Run Stage 2 for each part:")
    for p in range(10):
        print(f"    python -c \"from data.dementia4000 import dementia_reasoning_generate; "
              f"dementia_reasoning_generate(part={p})\"")

    return fc_means, fc_stds, num_windows_list, fc_descriptions, class_names


# ═══════════════════════════════════════════════════════════════════════════════
# 三阶段推理文本生成 — 阶段二：API 调用 (tech2.md §十)
# ═══════════════════════════════════════════════════════════════════════════════

def dementia_reasoning_generate(
    part,
    num_parts=10,
    max_workers=1500,
    descriptions_path="../data/Dementia4000/dfc_descriptions.npz",
    output_dir="../data/Dementia4000",
    api_key=None, base_url=None, model=None,
):
    """阶段二：调用 DeepSeek API 生成推理文本 — 处理指定分片。

    读取阶段一预计算的 dfc_descriptions.npz，处理分配给当前 part 的样本切片。
    使用 ThreadPoolExecutor 高并发调用 API（分批，每批最多 max_workers 个并发），
    支持断点续传（checkpoint）。

    使用方法：手动运行 10 次，每次指定不同的 part 值 (0~9)。
    10 次可以在不同终端并行运行，也可以逐个运行。

    Args:
        part:              int — 处理第几片（0-based），必填
        num_parts:         int — 总分片数（默认 10）
        max_workers:       int — 每批最大并发线程数（默认 1500，< Flash 2500 上限）
        descriptions_path: str — 阶段一输出的 dfc_descriptions.npz 路径
        output_dir:        str — 输出目录（存放 part JSON 和 checkpoint）
        api_key:           str — DeepSeek API key
        base_url:          str — API base URL
        model:             str — 模型名称
    """
    if api_key is None:
        api_key = os.environ.get('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY)
    if base_url is None:
        base_url = os.environ.get('DEEPSEEK_BASE_URL', DEEPSEEK_BASE_URL)
    if model is None:
        model = os.environ.get('DEEPSEEK_MODEL', DEEPSEEK_MODEL)

    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not set. Please set the environment variable:\n"
            "  export DEEPSEEK_API_KEY=your_key_here"
        )

    print("=" * 60)
    print(f"  Stage 2: API Reasoning Text Generation — Part {part}/{num_parts}")
    print("=" * 60)
    print(f"  max_workers={max_workers}, model={model}")

    # ── 1. 加载预计算的 DFC 描述 ──
    print(f"\n[1/4] Loading precomputed descriptions: {descriptions_path}")
    data = np.load(descriptions_path, allow_pickle=True)
    fc_descriptions_all = data['fc_descriptions']  # (N,) object array of str
    class_names_all = list(data['class_names'])     # list of str

    N_total = len(fc_descriptions_all)
    temperatures = TEMPERATURES  # [0.3, 0.6, 0.9]
    print(f"  Total samples: {N_total}")

    # ── 2. 确定本片负责的样本范围 ──
    part_size = int(np.ceil(N_total / num_parts))
    start = part * part_size
    end = min(start + part_size, N_total)
    sample_indices = list(range(start, end))

    if not sample_indices:
        print(f"  Part {part}: no samples (start={start} >= N_total={N_total})")
        return {}

    n_calls = len(sample_indices) * len(temperatures)
    print(f"  Part {part}: samples [{start}, {end}), "
          f"{len(sample_indices)} samples × {len(temperatures)} temps = {n_calls} API calls")

    # ── 3. 构建任务列表 ──
    system_prompt = build_system_prompt_ds()
    tasks = []  # list of (sample_idx, temp_idx, temperature, fc_desc, class_name)
    for idx in sample_indices:
        fc_desc = str(fc_descriptions_all[idx])
        cls_name = class_names_all[idx]
        for t_idx, temp in enumerate(temperatures):
            tasks.append((int(idx), t_idx, temp, fc_desc, cls_name))

    # ── 4. 断点续传：加载已有进度 ──
    checkpoint_path = os.path.join(output_dir,
                                   f'reasoning_texts_part_{part}_checkpoint.json')
    output_path = os.path.join(output_dir, f'reasoning_texts_part_{part}.json')
    prompts_path = os.path.join(output_dir, f'reasoning_prompts_part_{part}.json')
    os.makedirs(output_dir, exist_ok=True)

    # 每个样本的 fc_description 和 class_name（按原始样本索引查找）
    sample_info = {int(idx): (str(fc_descriptions_all[idx]), class_names_all[idx])
                   for idx in sample_indices}

    # prompts_log: 保存所有 prompt 信息用于调试和复现
    prompts_log = {
        "system_prompt": system_prompt,
        "temperatures": list(temperatures),
        "samples": {},
    }

    results = {}        # {str(sample_idx): [text_t0, text_t1, text_t2]}
    completed_set = set()  # set of (sample_idx, temp_idx)

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        for sidx_str, texts in results.items():
            for t_idx, text in enumerate(texts):
                if text and text.strip():
                    completed_set.add((int(sidx_str), t_idx))
        n_comp = sum(1 for v in results.values() for t in v if t and t.strip())
        n_exp = len(results) * len(temperatures) if results else 0
        print(f"  [Checkpoint] Loaded: {len(results)} samples, "
              f"{n_comp}/{n_exp} texts completed")

        # 恢复 prompts_log（如果存在）
        if os.path.exists(prompts_path):
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_log = json.load(f)
            print(f"  [Checkpoint] Loaded prompts_log: "
                  f"{len(prompts_log.get('samples', {}))} samples")

    # 过滤出待处理的任务
    pending_tasks = [(sidx, tidx, temp, fc_desc, cls_name)
                     for sidx, tidx, temp, fc_desc, cls_name in tasks
                     if (sidx, tidx) not in completed_set]

    if not pending_tasks:
        print("  ✓ All tasks already completed!")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Output: {output_path}")
        return results

    print(f"  Pending: {len(pending_tasks)} API calls")

    # ── 5. 分批并发调用 API ──
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_batches = int(np.ceil(len(pending_tasks) / max_workers))
    print(f"\n[2/4] Calling API ({n_batches} batches, up to {max_workers} concurrent each)...")

    total_completed = 0

    for batch_i in range(n_batches):
        batch_start = batch_i * max_workers
        batch_end = min(batch_start + max_workers, len(pending_tasks))
        batch_tasks = pending_tasks[batch_start:batch_end]

        print(f"\n  Batch {batch_i + 1}/{n_batches}: {len(batch_tasks)} calls...",
              flush=True)

        batch_success = 0
        batch_empty = 0
        batch_error = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_tasks))) as executor:
            futures = {}
            for task in batch_tasks:
                sidx, tidx, temp, fc_desc, cls_name = task
                future = executor.submit(
                    _call_api_worker,
                    (sidx, tidx, temp, fc_desc, cls_name,
                     system_prompt, api_key, base_url, model)
                )
                futures[future] = (sidx, tidx, temp)

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  Batch {batch_i+1}"):
                try:
                    sidx, tidx, temp, result = future.result()
                    text = result.get("text", "")
                    finish_reason = result.get("finish_reason", "unknown")

                    # 存入结果
                    sidx_str = str(sidx)
                    if sidx_str not in results:
                        results[sidx_str] = ["", "", ""]
                    results[sidx_str][tidx] = text

                    # 存入 prompts_log
                    if sidx_str not in prompts_log['samples']:
                        fc_desc, cls_name = sample_info.get(sidx, ("", ""))
                        prompts_log['samples'][sidx_str] = {
                            "class_name": cls_name,
                            "fc_description": fc_desc,
                            "results": [{}, {}, {}],
                        }
                    prompts_log['samples'][sidx_str]['results'][tidx] = {
                        "temperature": temp,
                        "finish_reason": finish_reason,
                        "text_len": len(text),
                        "text": text,
                    }

                    if text:
                        batch_success += 1
                    else:
                        batch_empty += 1
                        err = result.get("error", "unknown")
                        # 只对非空响应的问题做 verbose 打印
                except Exception as e:
                    batch_error += 1
                    # 找到对应的 sample_idx
                    sidx, tidx, temp = futures[future]
                    print(f"\n    [ERROR] sample {sidx}, T={temp:.1f}: {e}")

        total_completed += len(batch_tasks)
        print(f"    Done: {batch_success} OK, {batch_empty} empty, {batch_error} errors")

        # 每批完成后保存 checkpoint 和 prompts
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(prompts_path, 'w', encoding='utf-8') as f:
            json.dump(prompts_log, f, ensure_ascii=False, indent=2)
        print(f"    Checkpoint: {checkpoint_path}")

    # ── 6. 保存最终输出 ──
    print(f"\n[3/4] Saving final output...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Output: {output_path}")
    with open(prompts_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_log, f, ensure_ascii=False, indent=2)
    print(f"  Prompts: {prompts_path}")

    # ── 7. 统计 ──
    print(f"\n[4/4] Summary for part {part}:")
    n_samples = len(results)
    n_texts = sum(1 for v in results.values() for t in v if t and t.strip())
    n_expected = len(sample_indices) * len(temperatures)
    texts_list = [t for v in results.values() for t in v if t and t.strip()]
    avg_len = sum(len(t) for t in texts_list) / max(len(texts_list), 1)
    print(f"  Samples: {n_samples}")
    print(f"  Texts: {n_texts}/{n_expected}")
    print(f"  Avg length: {avg_len:.0f} chars")
    if n_texts < n_expected:
        print(f"  ⚠ {n_expected - n_texts} texts are empty — "
              f"may need to re-run this part")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 三阶段推理文本生成 — 阶段三：合并 (tech2.md §十)
# ═══════════════════════════════════════════════════════════════════════════════

def dementia_reasoning_merge(num_parts=10,
                             input_dir="../data/Dementia4000",
                             output_path=None):
    """阶段三：合并各分片的推理文本 JSON 为最终文件。

    运行一次即可。将所有 reasoning_texts_part_*.json 合并为 reasoning_texts.json。

    Args:
        num_parts:   int — 总分片数（默认 10）
        input_dir:   str — 包含 part JSON 文件的目录
        output_path: str — 输出路径（默认 {input_dir}/reasoning_texts.json）

    Returns:
        dict — 合并后的完整结果
    """
    if output_path is None:
        output_path = os.path.join(input_dir, 'reasoning_texts.json')

    print("=" * 60)
    print("  Stage 3: Merge Reasoning Texts")
    print("=" * 60)

    merged = {}
    total_texts = 0
    missing_parts = []
    empty_texts = 0

    for part in range(num_parts):
        part_path = os.path.join(input_dir, f'reasoning_texts_part_{part}.json')
        if os.path.exists(part_path):
            with open(part_path, 'r', encoding='utf-8') as f:
                part_data = json.load(f)
            merged.update(part_data)
            n = sum(1 for v in part_data.values() for t in v if t and t.strip())
            n_empty = sum(1 for v in part_data.values() for t in v if not t or not t.strip())
            total_texts += n
            empty_texts += n_empty
            empty_info = f", {n_empty} empty" if n_empty > 0 else ""
            print(f"  Part {part}: {len(part_data)} samples, {n} texts{empty_info}")
        else:
            missing_parts.append(part)
            print(f"  Part {part}: MISSING")

    # 去重检查：如果同一个 key 出现在多个 part（不应发生），做一次验证
    # Key 格式为 str(sample_idx)，正常情况下每个 part 覆盖不同的 idx 范围

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    n_expected = len(merged) * len(TEMPERATURES)
    print(f"\n{'=' * 60}")
    print(f"  Merged: {len(merged)} samples, {total_texts} texts "
          f"(expected ~{n_expected})")
    if empty_texts > 0:
        print(f"  ⚠ {empty_texts} empty texts — "
              f"consider re-running affected parts")
    if missing_parts:
        print(f"  ⚠ Missing parts: {missing_parts} — run these parts first")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 三阶段推理文本生成 — 测试模式（快速验证全流程）
# ═══════════════════════════════════════════════════════════════════════════════

def dementia_reasoning_test(
    max_samples=20,
    data_path="../data/Dementia4000/Dementia4000.npz",
    work_dir="../data/Dementia4000",
    api_key=None, base_url=None, model=None,
):
    """端到端测试：用少量样本快速验证三个阶段的完整流程。

    提取前 max_samples 条样本，在 work_dir 下完成：
      - DFC 预计算 + 文本描述生成（多进程）
      - API 调用（低并发，单 part，全量处理）
      - 合并
    全部在一条命令中完成。

    Args:
        max_samples: int — 测试样本数（默认 20）
        data_path:   str — Dementia4000.npz 路径
        work_dir:    str — 输出目录（默认 data/Dementia4000）
        api_key:     str — DeepSeek API key
        base_url:    str — API base URL
        model:       str — 模型名称

    Returns:
        dict — 合并后的推理文本结果
    """
    if api_key is None:
        api_key = os.environ.get('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY)
    if base_url is None:
        base_url = os.environ.get('DEEPSEEK_BASE_URL', DEEPSEEK_BASE_URL)
    if model is None:
        model = os.environ.get('DEEPSEEK_MODEL', DEEPSEEK_MODEL)

    os.makedirs(work_dir, exist_ok=True)

    full_data = np.load(data_path, allow_pickle=True)
    N_total = len(full_data['labels'])
    n = min(max_samples, N_total)

    print("=" * 60)
    print(f"  Dementia Reasoning TEST ({n} samples)")
    print(f"  Work dir: {work_dir}")
    print("=" * 60)

    # ── 阶段一：DFC 预计算 ──
    desc_path = os.path.join(work_dir, 'dfc_descriptions.npz')
    dementia_dfc_precompute(
        data_path=data_path,
        output_path=desc_path,
        max_samples=n,
    )

    # ── 阶段二：API 调用（单 part，低并发）──
    print("\n")

    # 调试模式：确认生成的描述是否合理
    desc_data = np.load(desc_path, allow_pickle=True)
    print(f"[Test] Descriptions preview (sample 0, first 200 chars):")
    print(f"  {str(desc_data['fc_descriptions'][0])[:200]}...")
    print()

    dementia_reasoning_generate(
        part=0,
        num_parts=1,
        max_workers=5,
        descriptions_path=desc_path,
        output_dir=work_dir,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    # ── 阶段三：合并 ──
    print("\n")
    final_path = os.path.join(work_dir, 'reasoning_texts.json')
    merged = dementia_reasoning_merge(
        num_parts=1,
        input_dir=work_dir,
        output_path=final_path,
    )

    # ── 快速质量检查 ──
    print(f"\n{'=' * 60}")
    print(f"  Quality Check (spot-check first 3 samples)")
    print(f"{'=' * 60}")
    for idx_str, texts in list(merged.items())[:3]:
        print(f"\n  Sample {idx_str}:")
        for t_idx, text in enumerate(texts):
            temp = TEMPERATURES[t_idx]
            if text and text.strip():
                preview = text[:120].replace('\n', ' ')
                print(f"    T={temp:.1f} [{len(text)} chars]: {preview}...")
            else:
                print(f"    T={temp:.1f}: [EMPTY]")
    print()

    return merged


if __name__ == '__main__':
    # ══════════════════════════════════════════════════════════════════════
    #  推理文本生成 — 入口汇总 (tech2.md)
    # ══════════════════════════════════════════════════════════════════════

    API_KEY = 'sk-dd1d7002a04f42f082e6c3adeb6d4b76'
    BASE_URL = 'https://api.deepseek.com'
    MODEL = 'deepseek-v4-flash'

    # ── 测试：小样本快速验证全流程（默认启用，20 条，~1-2 分钟）──
    # dementia_reasoning_test(
    #     max_samples=20,
    #     api_key=API_KEY,
    #     base_url=BASE_URL,
    #     model=MODEL,
    # )

    # ── 正式运行（三阶段，按需取消注释，分步执行）──
    #
    # 阶段一：DFC 预计算 + 文本描述生成（运行一次，~10-20 分钟）
    # dementia_dfc_precompute()
    #
    # 阶段二：API 调用（手动运行 10 次，修改 part=0~9，可并行，每片 ~5-10 分钟）
    # for i in range(7,10):
    #     dementia_reasoning_generate(
    #         part=i,
    #         num_parts=10,
    #         max_workers=1500,
    #         api_key=API_KEY,
    #         base_url=BASE_URL,
    #         model=MODEL,
    #     )
    #
    # 阶段三：合并（所有 part 完成后运行一次，~10 秒）
    dementia_reasoning_merge(num_parts=10)
