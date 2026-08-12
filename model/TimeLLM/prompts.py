"""Per-dataset prompt configuration for TimeLLM."""

from dataclasses import dataclass
from typing import List, Dict, Tuple


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
    "via cross-attention (10 windows x 19 channels, time-first order)."
)

_PROMPT_DATASET_CAUEEG = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "with subjects being AD, MCI, SCD patients and normal controls. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order)."
)

_PROMPT_DATASET_TUAB = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "data from the TUH Abnormal EEG Corpus, labeled as normal or abnormal. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order)."
)

_PROMPT_DATASET_TUEP = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "data from the TUH Epilepsy EEG Corpus, including epilepsy patients "
    "and non-epilepsy controls. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order)."
)

_PROMPT_DATASET_BEIRUT = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "data from 5 epilepsy patients, for seizure prediction. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order)."
)

_PROMPT_DATASET_MULTIDOMAIN_DISEASE = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "with data pooled from multiple datasets and recording sites, "
    "labeled as control/normal versus pathological (disease). "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order)."
)

_PROMPT_DATASET_MULTIDOMAIN_AGE = (
    "Dynamic functional connectivity matrices computed from EEG, "
    "with data pooled from multiple datasets and recording sites, "
    "for continuous age prediction. "
    "Matrices are encoded by GCN then mapped to 190 LLM tokens "
    "via cross-attention (10 windows x 19 channels, time-first order)."
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

@dataclass
class PromptConfig:
    channel_names: List[str]
    channel_groups: Dict[str, List[int]]
    homologous_pairs: List[Tuple[int, int]]
    prompt_dataset: str
    prompt_task: str

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
    },
    'DiseaseDS': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_DS,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'GenderDS': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_DS,
        'prompt_task': _PROMPT_TASK_GENDER,
    },
    'AgeDS': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_DS,
        'prompt_task': _PROMPT_TASK_AGE,
    },
    # ── CAUEEG ──
    'CAUEEG': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'DiseaseCAUEEG': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'AgeCAUEEG': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_AGE,
    },
    # ── TUAB ──
    'TUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUAB,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'DiseaseTUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUAB,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'GenderTUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUAB,
        'prompt_task': _PROMPT_TASK_GENDER,
    },
    'AgeTUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUAB,
        'prompt_task': _PROMPT_TASK_AGE,
    },
    # ── TUEP ──
    'TUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUEP,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'DiseaseTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUEP,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'GenderTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUEP,
        'prompt_task': _PROMPT_TASK_GENDER,
    },
    'AgeTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_TUEP,
        'prompt_task': _PROMPT_TASK_AGE,
    },
    # ── Beirut ──
    'Beirut': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_BEIRUT,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'DiseaseBeirut': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_BEIRUT,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    # ── 多域融合（跨数据集）──
    'DisCAUEEGTUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_MULTIDOMAIN_DISEASE,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'DisCAUEEGTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_MULTIDOMAIN_DISEASE,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'DisTUABTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_MULTIDOMAIN_DISEASE,
        'prompt_task': _PROMPT_TASK_DISEASE,
    },
    'AgeCAUEEGTUAB': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_MULTIDOMAIN_AGE,
        'prompt_task': _PROMPT_TASK_AGE,
    },
    'AgeCAUEEGTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_MULTIDOMAIN_AGE,
        'prompt_task': _PROMPT_TASK_AGE,
    },
    'AgeTUABTUEP': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_MULTIDOMAIN_AGE,
        'prompt_task': _PROMPT_TASK_AGE,
    },
}


def get_prompt_config(dataset_name: str) -> PromptConfig:
    """Return the prompt configuration for *dataset_name*.

    Falls back to ``'CAUEEG'`` if the dataset is not in the registry.
    """
    cfg = DATASET_PROMPTS.get(dataset_name, DATASET_PROMPTS['CAUEEG'])
    return PromptConfig(**cfg)
