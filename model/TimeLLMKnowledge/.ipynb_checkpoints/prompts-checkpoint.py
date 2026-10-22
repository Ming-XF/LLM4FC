"""Task prompts used by TimeLLMKnowledge (domain-knowledge compiler)."""

from dataclasses import dataclass

# 约定：所有 prompt 必须以 ":" 结尾——末位 token 即固定读出位置。

_PROMPT_TASK_CAUEEG = (
    "Analyze the EEG feature description above and determine "
    "whether the subject has Alzheimer's disease:"
)

_PROMPT_TASK_TUAB = (
    "Analyze the EEG feature description above and determine "
    "whether the EEG is abnormal:"
)

_PROMPT_TASK_TUEP = (
    "Analyze the EEG feature description above and determine "
    "whether the subject has epilepsy:"
)

_PROMPT_TASK_DISEASE = (
    "Analyze the EEG feature description above and determine "
    "whether the subject has the target condition:"
)

_PROMPT_TASK_AGE = (
    "Analyze the EEG feature description above and estimate "
    "the subject's age:"
)

_PROMPT_TASK_GENDER = (
    "Analyze the EEG feature description above and determine "
    "the subject's gender:"
)

_PROMPT_TASK_FUTURE_FC = (
    "Analyze the historical dynamic functional connectivity matrices "
    "above and predict the next matrix:"
)


@dataclass(frozen=True)
class PromptConfig:
    prompt_task: str


def get_prompt_config(dataset_name: str) -> PromptConfig:
    """Return a task prompt inferred from the dataset name (must end with ':')."""
    if dataset_name.startswith('DiseaseCAUEEG'):
        prompt = _PROMPT_TASK_CAUEEG
    elif dataset_name.startswith('DiseaseTUAB'):
        prompt = _PROMPT_TASK_TUAB
    elif dataset_name.startswith('DiseaseTUEP'):
        prompt = _PROMPT_TASK_TUEP
    elif dataset_name.startswith('Age'):
        prompt = _PROMPT_TASK_AGE
    elif dataset_name.startswith('Gender'):
        prompt = _PROMPT_TASK_GENDER
    elif dataset_name.startswith('FutureFC'):
        prompt = _PROMPT_TASK_FUTURE_FC
    else:
        prompt = _PROMPT_TASK_DISEASE
    return PromptConfig(prompt_task=prompt)
