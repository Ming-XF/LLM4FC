"""
脑图谱知识模块 — 标准脑区顺序、功能网络映射、MNI 坐标、空间先验、System Prompt。

基于 Desikan-Killiany 图谱 (68 脑区) + Yeo 7-Network 划分。
"""

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 脑区顺序 — 与 FC 矩阵 (68×68) 的行列严格对齐
# ═══════════════════════════════════════════════════════════════════════════════

REGION_ORDER = [
    'bankssts_L', 'bankssts_R',
    'caudalanteriorcingulate_L', 'caudalanteriorcingulate_R',
    'caudalmiddlefrontal_L', 'caudalmiddlefrontal_R',
    'cuneus_L', 'cuneus_R',
    'entorhinal_L', 'entorhinal_R',
    'frontalpole_L', 'frontalpole_R',
    'fusiform_L', 'fusiform_R',
    'inferiorparietal_L', 'inferiorparietal_R',
    'inferiortemporal_L', 'inferiortemporal_R',
    'insula_L', 'insula_R',
    'isthmuscingulate_L', 'isthmuscingulate_R',
    'lateraloccipital_L', 'lateraloccipital_R',
    'lateralorbitofrontal_L', 'lateralorbitofrontal_R',
    'lingual_L', 'lingual_R',
    'medialorbitofrontal_L', 'medialorbitofrontal_R',
    'middletemporal_L', 'middletemporal_R',
    'paracentral_L', 'paracentral_R',
    'parahippocampal_L', 'parahippocampal_R',
    'parsopercularis_L', 'parsopercularis_R',
    'parsorbitalis_L', 'parsorbitalis_R',
    'parstriangularis_L', 'parstriangularis_R',
    'pericalcarine_L', 'pericalcarine_R',
    'postcentral_L', 'postcentral_R',
    'posteriorcingulate_L', 'posteriorcingulate_R',
    'precentral_L', 'precentral_R',
    'precuneus_L', 'precuneus_R',
    'rostralanteriorcingulate_L', 'rostralanteriorcingulate_R',
    'rostralmiddlefrontal_L', 'rostralmiddlefrontal_R',
    'superiorfrontal_L', 'superiorfrontal_R',
    'superiorparietal_L', 'superiorparietal_R',
    'superiortemporal_L', 'superiortemporal_R',
    'supramarginal_L', 'supramarginal_R',
    'temporalpole_L', 'temporalpole_R',
    'transversetemporal_L', 'transversetemporal_R',
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 脑区 → 功能网络映射（基于 Desikan-Killiany + Yeo 7-Network, 扩展至 9 类）
# ═══════════════════════════════════════════════════════════════════════════════

# 网络 ID 常量
NET_DMN = 0   # Default Mode Network
NET_SN = 1    # Salience Network
NET_CEN = 2   # Central Executive Network / Frontoparietal
NET_VIS = 3   # Visual
NET_SMN = 4   # Sensorimotor
NET_AUD = 5   # Auditory / Language
NET_LIM = 6   # Limbic
NET_OFC = 7   # Orbitofrontal
NET_PFC = 8   # Prefrontal

NETWORK_NAMES_ZH = {
    NET_DMN: '默认模式网络(DMN)',
    NET_SN: '突显网络(SN)',
    NET_CEN: '中央执行/额顶网络(CEN)',
    NET_VIS: '视觉网络(VIS)',
    NET_SMN: '感觉运动网络(SMN)',
    NET_AUD: '听觉/语言网络(AUD)',
    NET_LIM: '边缘系统(LIM)',
    NET_OFC: '眶额皮质(OFC)',
    NET_PFC: '前额叶皮质(PFC)',
}

REGION_TO_NETWORK = {
    # DMN — 14 regions
    'precuneus_L': NET_DMN, 'precuneus_R': NET_DMN,
    'posteriorcingulate_L': NET_DMN, 'posteriorcingulate_R': NET_DMN,
    'inferiorparietal_L': NET_DMN, 'inferiorparietal_R': NET_DMN,
    'isthmuscingulate_L': NET_DMN, 'isthmuscingulate_R': NET_DMN,
    'rostralanteriorcingulate_L': NET_DMN, 'rostralanteriorcingulate_R': NET_DMN,
    'medialorbitofrontal_L': NET_DMN, 'medialorbitofrontal_R': NET_DMN,
    'parahippocampal_L': NET_DMN, 'parahippocampal_R': NET_DMN,

    # SN — 6 regions
    'insula_L': NET_SN, 'insula_R': NET_SN,
    'caudalanteriorcingulate_L': NET_SN, 'caudalanteriorcingulate_R': NET_SN,
    'supramarginal_L': NET_SN, 'supramarginal_R': NET_SN,

    # CEN — 6 regions
    'caudalmiddlefrontal_L': NET_CEN, 'caudalmiddlefrontal_R': NET_CEN,
    'rostralmiddlefrontal_L': NET_CEN, 'rostralmiddlefrontal_R': NET_CEN,
    'superiorparietal_L': NET_CEN, 'superiorparietal_R': NET_CEN,

    # VIS — 8 regions
    'cuneus_L': NET_VIS, 'cuneus_R': NET_VIS,
    'lateraloccipital_L': NET_VIS, 'lateraloccipital_R': NET_VIS,
    'lingual_L': NET_VIS, 'lingual_R': NET_VIS,
    'pericalcarine_L': NET_VIS, 'pericalcarine_R': NET_VIS,

    # SMN — 6 regions
    'precentral_L': NET_SMN, 'precentral_R': NET_SMN,
    'postcentral_L': NET_SMN, 'postcentral_R': NET_SMN,
    'paracentral_L': NET_SMN, 'paracentral_R': NET_SMN,

    # AUD — 8 regions
    'superiortemporal_L': NET_AUD, 'superiortemporal_R': NET_AUD,
    'middletemporal_L': NET_AUD, 'middletemporal_R': NET_AUD,
    'transversetemporal_L': NET_AUD, 'transversetemporal_R': NET_AUD,
    'bankssts_L': NET_AUD, 'bankssts_R': NET_AUD,

    # LIM — 8 regions
    'entorhinal_L': NET_LIM, 'entorhinal_R': NET_LIM,
    'temporalpole_L': NET_LIM, 'temporalpole_R': NET_LIM,
    'fusiform_L': NET_LIM, 'fusiform_R': NET_LIM,
    'inferiortemporal_L': NET_LIM, 'inferiortemporal_R': NET_LIM,

    # OFC — 4 regions
    'lateralorbitofrontal_L': NET_OFC, 'lateralorbitofrontal_R': NET_OFC,
    'parsorbitalis_L': NET_OFC, 'parsorbitalis_R': NET_OFC,

    # PFC — 8 regions
    'frontalpole_L': NET_PFC, 'frontalpole_R': NET_PFC,
    'superiorfrontal_L': NET_PFC, 'superiorfrontal_R': NET_PFC,
    'parsopercularis_L': NET_PFC, 'parsopercularis_R': NET_PFC,
    'parstriangularis_L': NET_PFC, 'parstriangularis_R': NET_PFC,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MNI 坐标 (Desikan-Killiany atlas, 标准 MNI152 空间)
# ═══════════════════════════════════════════════════════════════════════════════

COORDINATES = {
    'bankssts_L': [-54.34, -44.54, 4.16],
    'bankssts_R': [52.98, -40.55, 5.30],
    'caudalanteriorcingulate_L': [-5.03, 20.09, 29.00],
    'caudalanteriorcingulate_R': [5.01, 22.26, 27.64],
    'caudalmiddlefrontal_L': [-35.52, 10.81, 44.19],
    'caudalmiddlefrontal_R': [35.66, 12.29, 44.47],
    'cuneus_L': [-7.13, -79.63, 18.51],
    'cuneus_R': [7.17, -80.09, 19.16],
    'entorhinal_L': [-23.00, -7.88, -35.21],
    'entorhinal_R': [22.76, -7.62, -34.08],
    'frontalpole_L': [-6.79, 64.87, -11.50],
    'frontalpole_R': [8.69, 64.42, -11.92],
    'fusiform_L': [-35.14, -43.47, -22.01],
    'fusiform_R': [35.32, -43.24, -21.60],
    'inferiorparietal_L': [-40.93, -67.71, 28.12],
    'inferiorparietal_R': [44.35, -61.78, 28.63],
    'inferiortemporal_L': [-49.60, -34.70, -25.26],
    'inferiortemporal_R': [50.78, -31.73, -26.19],
    'insula_L': [-37.14, -3.50, 1.69],
    'insula_R': [38.25, -3.02, 1.54],
    'isthmuscingulate_L': [-6.62, -47.25, 16.97],
    'isthmuscingulate_R': [7.09, -46.16, 16.74],
    'lateraloccipital_L': [-30.08, -88.50, -1.52],
    'lateraloccipital_R': [31.12, -87.94, -0.46],
    'lateralorbitofrontal_L': [-24.79, 28.72, -16.97],
    'lateralorbitofrontal_R': [24.24, 29.35, -18.00],
    'lingual_L': [-14.51, -67.61, -5.06],
    'lingual_R': [14.75, -66.77, -4.33],
    'medialorbitofrontal_L': [-5.41, 36.93, -18.00],
    'medialorbitofrontal_R': [5.86, 37.57, -16.58],
    'middletemporal_L': [-57.75, -30.22, -13.29],
    'middletemporal_R': [58.17, -27.92, -13.56],
    'paracentral_L': [-7.90, -29.74, 56.12],
    'paracentral_R': [7.83, -28.58, 55.54],
    'parahippocampal_L': [-23.91, -33.14, -19.25],
    'parahippocampal_R': [25.38, -33.02, -18.14],
    'parsopercularis_L': [-45.75, 14.56, 11.85],
    'parsopercularis_R': [46.53, 14.24, 13.38],
    'parsorbitalis_L': [-42.51, 38.55, -14.11],
    'parsorbitalis_R': [44.12, 39.16, -11.99],
    'parstriangularis_L': [-44.02, 30.27, 0.81],
    'parstriangularis_R': [46.62, 29.46, 3.34],
    'pericalcarine_L': [-11.77, -81.49, 5.37],
    'pericalcarine_R': [12.52, -80.24, 6.01],
    'postcentral_L': [-43.23, -23.57, 43.95],
    'postcentral_R': [42.37, -22.48, 44.56],
    'posteriorcingulate_L': [-5.70, -18.39, 38.47],
    'posteriorcingulate_R': [5.69, -17.20, 38.86],
    'precentral_L': [-38.79, -10.41, 42.97],
    'precentral_R': [37.90, -9.81, 44.55],
    'precuneus_L': [-9.69, -58.23, 36.66],
    'precuneus_R': [9.64, -57.31, 37.85],
    'rostralanteriorcingulate_L': [-4.39, 37.52, -0.21],
    'rostralanteriorcingulate_R': [5.37, 37.11, 1.68],
    'rostralmiddlefrontal_L': [-33.21, 42.72, 16.84],
    'rostralmiddlefrontal_R': [33.97, 42.84, 17.68],
    'superiorfrontal_L': [-11.38, 24.09, 43.37],
    'superiorfrontal_R': [12.19, 25.70, 43.08],
    'superiorparietal_L': [-23.41, -61.79, 47.83],
    'superiorparietal_R': [23.19, -60.48, 49.69],
    'superiortemporal_L': [-53.40, -15.66, -4.01],
    'superiortemporal_R': [54.39, -12.26, -5.12],
    'supramarginal_L': [-52.06, -39.13, 31.48],
    'supramarginal_R': [52.10, -33.13, 31.20],
    'temporalpole_L': [-29.31, 12.90, -38.05],
    'temporalpole_R': [30.57, 13.78, -35.77],
    'transversetemporal_L': [-44.47, -22.68, 7.33],
    'transversetemporal_R': [44.65, -20.80, 8.17],
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 派生索引表
# ═══════════════════════════════════════════════════════════════════════════════

# 名称 → 整数索引
_name_to_idx = {name: i for i, name in enumerate(REGION_ORDER)}

# 半球 ID: 0 = 左, 1 = 右
HEMI_IDS = {name: (0 if name.endswith('_L') else 1) for name in REGION_ORDER}

# 网络 ID 列表（按 REGION_ORDER 顺序）
NETWORK_IDS_LIST = [REGION_TO_NETWORK[name] for name in REGION_ORDER]

# 半球 ID 列表
HEMI_IDS_LIST = [HEMI_IDS[name] for name in REGION_ORDER]

# 坐标 numpy 数组 (68, 3)
COORDS_ARRAY = np.array([COORDINATES[name] for name in REGION_ORDER], dtype=np.float32)

# 每个脑区的短名称（去掉 _L/_R 后缀）
def _short_name(name):
    base = name.replace('_L', '').replace('_R', '')
    return base

SHORT_NAMES = [_short_name(name) for name in REGION_ORDER]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def get_region_index(name: str):
    """根据脑区名称获取 FC 矩阵中的行列索引。"""
    return _name_to_idx.get(name, None)


def normalize_region_name(name: str):
    """
    将脑区名称标准化为 REGION_ORDER 中的格式。
    支持：完整名 bankssts_L、空格名 bankssts L、短名 bankssts。
    返回标准化名称，失败返回 None。
    """
    # 精确匹配
    if name in _name_to_idx:
        return name

    # 空格 → 下划线
    name_us = name.replace(' ', '_')
    if name_us in _name_to_idx:
        return name_us

    # 尝试加 _L / _R 后缀
    for suffix in ['_L', '_R']:
        candidate = name_us + suffix
        if candidate in _name_to_idx:
            return candidate

    # 模糊匹配（短名包含在标准名中）
    name_lower = name_us.lower()
    for std_name in REGION_ORDER:
        if name_lower in std_name.lower():
            return std_name

    return None


def get_network_for_region(name: str):
    """获取脑区所属功能网络 ID。"""
    return REGION_TO_NETWORK.get(name, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 空间先验矩阵
# ═══════════════════════════════════════════════════════════════════════════════

def build_spatial_prior(network_ids=None, coords=None, hemi_ids=None,
                        alpha=0.1) -> torch.Tensor:
    """
    构建基于解剖先验的空间偏置矩阵。

    规则:
      - 同功能网络 → +0.15
      - 同半球 → +0.05
      - 空间近邻 (Gaussian, sigma=50mm) → 最多 +0.10

    Args:
        network_ids: list[int], length C  网络 ID 列表
        coords:       np.ndarray, (C, 3)  MNI 坐标
        hemi_ids:     list[int], length C  半球 ID (0=左, 1=右)
        alpha:        缩放系数

    Returns:
        prior: torch.Tensor (C, C)  空间先验矩阵，对角线为 0
    """
    if network_ids is None:
        network_ids = NETWORK_IDS_LIST
    if coords is None:
        coords = COORDS_ARRAY
    if hemi_ids is None:
        hemi_ids = HEMI_IDS_LIST

    C = len(network_ids)
    prior = torch.zeros(C, C)

    for i in range(C):
        for j in range(i + 1, C):
            bias = 0.0
            # 同功能网络
            if network_ids[i] == network_ids[j]:
                bias += 0.15
            # 同半球
            if hemi_ids[i] == hemi_ids[j]:
                bias += 0.05
            # 空间近邻
            dist = np.linalg.norm(coords[i] - coords[j])
            bias += 0.10 * np.exp(-dist ** 2 / (2 * 50 ** 2))

            prior[i, j] = prior[j, i] = bias

    return alpha * prior


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EEG 19 通道图谱 — 用于非 MRI 的 EEG 数据集（如 Dementia_MMS）
# ═══════════════════════════════════════════════════════════════════════════════

# 标准 10-20 系统，对齐 Dementia_MMS 数据集的 signal_header 顺序
EEG19_CHANNEL_ORDER = [
    'Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'O1-AVG',
    'Fp2-AVG', 'F4-AVG', 'C4-AVG', 'P4-AVG', 'O2-AVG',
    'F7-AVG', 'T3-AVG', 'T5-AVG',
    'F8-AVG', 'T4-AVG', 'T6-AVG',
    'FZ-AVG', 'CZ-AVG', 'PZ-AVG',
]

# 显示名 — 去掉 -AVG 后缀并补全标准电极名
EEG19_CHANNEL_LABELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7', 'T3', 'T5',
    'F8', 'T4', 'T6',
    'Fz', 'Cz', 'Pz',
]

# 中文功能描述 — 每个电极的解剖/功能意义
EEG19_CHANNEL_NAMES_ZH = [
    '左前额极(Fp1)', '左额区(F3)', '左中央区(C3)', '左顶区(P3)', '左枕区(O1)',
    '右前额极(Fp2)', '右额区(F4)', '右中央区(C4)', '右顶区(P4)', '右枕区(O2)',
    '左前颞(F7)',   '左中颞(T3)', '左后颞(T5)',
    '右前颞(F8)',   '右中颞(T4)', '右后颞(T6)',
    '中线额区(Fz)', '中线中央区(Cz)', '中线顶区(Pz)',
]

# 功能区聚合 — 供 per-group 汇总（当不需要逐个通道列出时）
EEG19_CHANNEL_GROUPS = {
    '额区':   [0, 1, 5, 6, 10, 13, 16],   # Fp1,F3,Fp2,F4,F7,F8,Fz
    '中央区': [2, 7, 17],                   # C3,C4,Cz
    '颞区':   [11, 12, 14, 15],             # T3,T5,T4,T6
    '顶区':   [3, 8, 18],                   # P3,P4,Pz
    '枕区':   [4, 9],                        # O1,O2
}

# name → index dicts for quick lookup
EEG19_CHANNEL_IDX = {label: i for i, label in enumerate(EEG19_CHANNEL_LABELS)}

# ═══════════════════════════════════════════════════════════════════════════════
# 8. 通用通道图谱查询
# ═══════════════════════════════════════════════════════════════════════════════

def get_channel_atlas(node_size: int):
    """根据 FC 矩阵的通道数返回对应的电极/脑区图谱。

    返回 dict 或 None:
      - 68  → Desikan-Killiany 脑图谱 (MRI/fMRI DFC)
      - 19  → 10-20 EEG 电极 (Dementia_MMS 等)
      - 其他 → None（调用方应避免生成无法理解的文本）

    返回 dict 结构:
      names_zh:       list[str]  C 个通道的中文名称
      group_indices:  dict[str → list[int]]  功能区→通道索引列表
      channel_labels: list[str]  C 个通道的简短标签
    """
    if node_size == 68:
        return {
            'names_zh': [f'{REGION_TO_NETWORK.get(name, -1)}网-{name}'
                         for name in REGION_ORDER],
            'group_indices': None,  # 68 脑区走 per-network 路径，不在此处处理
            'channel_labels': REGION_ORDER,
        }
    elif node_size == 19:
        return {
            'names_zh': EEG19_CHANNEL_NAMES_ZH,
            'group_indices': EEG19_CHANNEL_GROUPS,
            'channel_labels': EEG19_CHANNEL_LABELS,
        }
    elif node_size == 62:
        # 62 通道 EEG（扩展 10-20）：暂不映射，退回 top-k 通道名列表
        return None
    else:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# 9. System Prompt 构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(region_to_network=None):
    """
    构建 LLM system prompt，说明输入结构和任务目标。

    Returns:
        system_prompt: str
    """
    return (
        "你是一位神经影像专家。接下来你会依次收到两组 Soft Prompt，"
        "它们由脑功能网络的专业知识文本嵌入经患者特异性激活加权生成："
        "(1) 跨时间窗的整体脑功能网络激活画像，"
        "(2) 各个时间窗口的动态脑网络激活快照。"
        "请基于这些信息，诊断该患者属于以下四类之一："
        "阿尔茨海默病、轻度认知障碍、主观认知下降或正常认知。"
    )
