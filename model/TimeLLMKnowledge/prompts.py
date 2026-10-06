"""Task prompts used by TimeLLMKnowledge (domain-knowledge compiler)."""

from dataclasses import dataclass


_PROMPT_TASK_DISEASE = (
    "Given a 19-channel EEG time series, classify whether the subject "
    "has the target condition."
)

_PROMPT_TASK_AGE = (
    "Given 19-channel EEG time series, predict the subject's age."
)

_PROMPT_TASK_GENDER = (
    "Given a 19-channel EEG time series, classify the subject's gender "
    "as male or female."
)

_PROMPT_TASK_FUTURE_FC = (
    "Given historical dynamic functional connectivity matrices, "
    "predict the next dynamic functional connectivity matrix."
)


@dataclass(frozen=True)
class PromptConfig:
    prompt_task: str


def get_prompt_config(dataset_name: str) -> PromptConfig:
    """Return a task-only prompt inferred from the dataset name."""
    if dataset_name.startswith('Age'):
        prompt = _PROMPT_TASK_AGE
    elif dataset_name.startswith('Gender'):
        prompt = _PROMPT_TASK_GENDER
    elif dataset_name.startswith('FutureFC'):
        prompt = _PROMPT_TASK_FUTURE_FC
    else:
        prompt = _PROMPT_TASK_DISEASE
    return PromptConfig(prompt_task=prompt)
