"""Per-dataset prompt configuration for TimeLLM."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ── 通用 EEG 19 通道 10-20 配置（CAUEEG / DS 共用）──────

_EEG_19_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7',  'T3', 'T5',
    'F8',  'T4', 'T6',
    'Fz',  'Cz', 'Pz',
]

_EEG_CHANNEL_GROUPS = {
    'frontal':  [0, 5, 1, 6, 10, 13, 16],
    'temporal': [11, 14, 12, 15],
    'central':  [2, 17, 7],
    'parietal': [3, 18, 8],
    'occipital': [4, 9],
}

_EEG_HOMOLOGOUS_PAIRS = [
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    (10, 13), (11, 14), (12, 15),
]

# ── Prompt 模板 ──────────────────────────────────────

_PROMPT_DATASET_DS = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "with subjects being AD patients and healthy controls. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order). "
    "Channel order (10-20 system): Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, "
    "F7, T3, T5, F8, T4, T6, Fz, Cz, Pz."
)

_PROMPT_DATASET_CAUEEG = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "with subjects being AD, MCI, SCD patients and normal controls. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order). "
    "Channel order (10-20 system): Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, "
    "F7, T3, T5, F8, T4, T6, Fz, Cz, Pz."
)

_PROMPT_DATASET_TUAB = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "data from the TUH Abnormal EEG Corpus, labeled as normal or abnormal. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order). "
    "Channel order (10-20 system): Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, "
    "F7, T3, T5, F8, T4, T6, Fz, Cz, Pz."
)

_PROMPT_DATASET_TUEP = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "data from the TUH Epilepsy EEG Corpus, including epilepsy patients "
    "and non-epilepsy controls. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order). "
    "Channel order (10-20 system): Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, "
    "F7, T3, T5, F8, T4, T6, Fz, Cz, Pz."
)

_PROMPT_DATASET_BEIRUT = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "data from 5 epilepsy patients, for seizure prediction. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order). "
    "Channel order (10-20 system): Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, "
    "F7, T3, T5, F8, T4, T6, Fz, Cz, Pz."
)

_PROMPT_TASK_DISEASE = (
    "Given 19-channel dynamic functional connectivity matrices "
    "(10 time windows), classify whether the subject has the target condition."
)

_PROMPT_TASK_AGE = (
    "Given 19-channel dynamic functional connectivity matrices "
    "(10 time windows), predict the subject's age."
)

_PROMPT_TASK_GENDER = (
    "Given 19-channel dynamic functional connectivity matrices "
    "(10 time windows), classify the subject's gender as male or female."
)

_PROMPT_TASK_FUTUREFC = (
    "Given historical dynamic functional connectivity matrices, "
    "predict the next dynamic functional connectivity matrix."
)

_PROMPT_STATS_EEG = (
    "Maximum connection: {max_pair} (r={max_val:.3f}). "
    "Minimum positive connection: {min_pos_pair} (r={min_pos_val:.3f}). "
    "Strongest negative connection: {max_neg_pair} (r={max_neg_val:.3f}). "
    "Mean intra-frontal FC: {fc_frontal:.3f}. "
    "Mean inter-hemispheric homologous FC: {fc_homologous:.3f}. "
    "Global mean FC: {mean_fc:.3f} (std: {std_fc:.3f})."
)


@dataclass
class PromptConfig:
    channel_names: List[str]
    channel_groups: Dict[str, List[int]]
    homologous_pairs: List[Tuple[int, int]]
    prompt_dataset: str
    prompt_task: str
    prompt_stats_template: Optional[str]

    @property
    def system_prompt(self) -> str:
        return "\n".join([self.prompt_dataset, self.prompt_task])


# ── 数据集配置字典 ──────────────────────────────────

DATASET_PROMPTS: Dict[str, dict] = {
    # ── DS ──
    'DS': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_DS,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'DiseaseDS': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_DS,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'GenderDS': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_DS,
        'prompt_task': _PROMPT_TASK_GENDER,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'AgeDS': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_DS,
        'prompt_task': _PROMPT_TASK_AGE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    # ── CAUEEG ──
    'CAUEEG': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'DiseaseCAUEEG': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'AgeCAUEEG': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_AGE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    # ── TUAB ──
    'TUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUAB,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'DiseaseTUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUAB,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'GenderTUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUAB,
        'prompt_task': _PROMPT_TASK_GENDER,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'AgeTUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUAB,
        'prompt_task': _PROMPT_TASK_AGE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    # ── TUEP ──
    'TUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUEP,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'DiseaseTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUEP,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'GenderTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUEP,
        'prompt_task': _PROMPT_TASK_GENDER,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'AgeTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUEP,
        'prompt_task': _PROMPT_TASK_AGE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    # ── Beirut ──
    'Beirut': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_BEIRUT,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
    'DiseaseBeirut': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_BEIRUT,
        'prompt_task': _PROMPT_TASK_DISEASE,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
}


def get_prompt_config(dataset_name: str) -> PromptConfig:
    """Return the prompt configuration for *dataset_name*.

    Falls back to ``'CAUEEG'`` if the dataset is not in the registry.
    """
    cfg = DATASET_PROMPTS.get(dataset_name, DATASET_PROMPTS['CAUEEG'])
    return PromptConfig(**cfg)
