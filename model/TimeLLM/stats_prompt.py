"""Stats prompt: format CAER-v3 evidence stats as text fed to the frozen LLM.

每个样本的 18/32 维 display 统计量（已按固定尺度缩放、裁剪 [-5,5]，
见 graph_evidence.compute_graph_evidence）被格式化成一段固定结构的
英文文本，作为统计 prompt 插在任务 prompt 之后、DFC patches 之前。
"""

import math

_STATS_HEADER = (
    "EEG statistical prompt.\n"
    "These continuous statistics summarize one EEG sample over 10 dynamic "
    "windows.\n"
    "Each displayed value is clip(raw_value / fixed_scale, -5, +5).\n"
    "Relative powers are fractions. Alpha peak frequency is measured in Hz "
    "before scaling."
)
_STATS_END = "End of EEG statistical prompt."

_FC_GROUP = "Connectivity statistics:"
_SPEC_GROUP = "Spectral statistics:"
_MORPH_GROUP = "Morphology and periodicity statistics:"

_FC_FIELDS = [
    ("strength", "overall connectivity strength"),
    ("inter_ratio", "inter-hemispheric coupling ratio"),
    ("front_ratio", "frontal-to-posterior connectivity ratio"),
    ("asym", "hemispheric asymmetry"),
    ("cent_mw", "central connectivity"),
    ("temp_mw", "temporal connectivity"),
    ("pari_mw", "parietal connectivity"),
    ("occ_mw", "occipital connectivity"),
    ("mid_mw", "midline connectivity"),
]

_SPEC_FIELDS = [
    ("rel_d", "delta relative power"),
    ("rel_t", "theta relative power"),
    ("rel_a", "alpha relative power"),
    ("rel_b", "beta relative power"),
    ("tb_ratio", "theta-to-beta power ratio"),
    ("a_dt", "alpha-to-delta-plus-theta power ratio"),
    ("slowing", "diffuse slowing index"),
    ("fa_pa", "frontal-to-posterior alpha power ratio"),
    ("tt_total", "temporal theta power fraction"),
]

_MORPH_FIELDS = [
    ("ll_mean", "line length mean"),
    ("ll_max", "line length maximum"),
    ("sharp_mean", "sharpness mean"),
    ("sharp_max", "sharpness maximum"),
    ("sharp_focal", "sharpness focality"),
    ("kurt_mean", "amplitude kurtosis mean"),
    ("kurt_max", "amplitude kurtosis maximum"),
    ("zcr_mean", "zero-crossing rate"),
    ("gamma_pow", "gamma power fraction"),
    ("spike_mean", "spike rate mean"),
    ("spike_max", "spike rate maximum"),
    ("apf", "alpha peak frequency"),
    ("apf_std", "alpha peak frequency variability"),
    ("slope", "aperiodic spectral slope"),
]


def get_stats_schema(stats_dim):
    """按统计量宽度返回有序 (分组头, 字段表) 列表。

    stats_dim: 9（仅 FC）/ 18（FC+频谱）/ 32（全三族）。
    """
    if stats_dim == 32:
        return [
            (_FC_GROUP, _FC_FIELDS),
            (_SPEC_GROUP, _SPEC_FIELDS),
            (_MORPH_GROUP, _MORPH_FIELDS),
        ]
    if stats_dim == 18:
        return [(_FC_GROUP, _FC_FIELDS), (_SPEC_GROUP, _SPEC_FIELDS)]
    if stats_dim == 9:
        return [(_FC_GROUP, _FC_FIELDS)]
    raise ValueError(f"Unexpected stats dim {stats_dim}")


def _format_value(v):
    """固定 4 位小数、强制正负号、无科学计数法；-0.0000 归正。"""
    v = float(v)
    if not math.isfinite(v):
        v = 0.0
    v = max(-5.0, min(5.0, v))
    s = f"{v:+.4f}"
    return "+0.0000" if s == "-0.0000" else s


def build_stats_text(stats):
    """stats: (B, D) display 值 → 每样本一段统计 prompt 文本。"""
    groups = get_stats_schema(stats.shape[-1])
    texts = []
    for row in stats:
        parts = [_STATS_HEADER]
        offset = 0
        for group_name, fields in groups:
            parts.append(group_name)
            for i, (key, name) in enumerate(fields):
                parts.append(f"{name} [{key}] = {_format_value(row[offset + i])}")
            offset += len(fields)
        parts.append(_STATS_END)
        texts.append("\n".join(parts))
    return texts
