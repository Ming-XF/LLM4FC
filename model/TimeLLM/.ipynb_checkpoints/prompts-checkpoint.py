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

_PROMPT_DATASET_CAUEEG = (
    "This dataset is used for AD dementia diagnosis, "
    "based on resting-state EEG data. Each "
    "subject's 60-second recording is divided into 10 consecutive time "
    "windows, and Pearson correlation "
    "is computed between each of the 19 EEG channels within each window to "
    "produce DFC matrices. The following "
    "data are presented as a sequence of 190 tokens: for each of the 19 channels, "
    "10 time-window representations (T0 to T9) are provided in order (channel-first "
    "ordering). The 19 channels (international 10-20 system) are: Fp1, F3, C3, P3, "
    "O1, Fp2, F4, C4, P4, O2, F7, T3, T5, F8, T4, T6, FZ, CZ, PZ."
)

_PROMPT_TASK_BINARY = (
    "Given 19-channel brain functional connectivity, classify the subject "
    "as AD or NC."
)

_PROMPT_DATASET_BEIRUT = (
    "This dataset is used for epileptic seizure prediction, based on "
    "long-term intracranial EEG recordings from 5 patients. Each 60-second "
    "sliding window (30-second stride) of 19-channel EEG is converted into "
    "dynamic functional connectivity (DFC) matrices via Pearson correlation "
    "within 10 consecutive 3-second sub-windows. The following data are "
    "presented as a sequence of 190 tokens: for each of the 19 channels, "
    "10 time-window representations (T0 to T9) are provided in order "
    "(channel-first ordering). The 19 channels (international 10-20 system) "
    "are: Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, F7, T3, T5, F8, T4, "
    "T6, FZ, CZ, PZ."
)

_PROMPT_TASK_BEIRUT = (
    "Given 19-channel brain functional connectivity, predict whether an "
    "epileptic seizure will occur within the next 10 minutes (pre-ictal) "
    "or not (inter-ictal)."
)

_PROMPT_TASK_4CLASS = (
    "Given 19-channel brain functional connectivity, classify the subject "
    "as AD, MCI, "
    "SCD (subjective cognitive decline), or NC."
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
    'CAUEEG2': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_BINARY,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },

    'CAUEEG4': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_4CLASS,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },

    'DS': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_CAUEEG,
        'prompt_task': _PROMPT_TASK_BINARY,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },

    'Beirut': {
        'channel_names': _EEG_19_CHANNELS,
        'channel_groups': _EEG_CHANNEL_GROUPS,
        'homologous_pairs': _EEG_HOMOLOGOUS_PAIRS,
        'prompt_dataset': _PROMPT_DATASET_BEIRUT,
        'prompt_task': _PROMPT_TASK_BEIRUT,
        'prompt_stats_template': _PROMPT_STATS_EEG,
    },
}


def get_prompt_config(dataset_name: str) -> PromptConfig:
    """Return the prompt configuration for *dataset_name*.

    Falls back to ``'CAUEEG2'`` if the dataset is not in the registry.
    """
    cfg = DATASET_PROMPTS.get(dataset_name, DATASET_PROMPTS['CAUEEG2'])
    return PromptConfig(**cfg)
