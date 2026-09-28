"""Task prompts and semantic prototype bank used by TimeLLM."""

from dataclasses import dataclass


# E2: 神经语义原型库 —— 用冻结 LLM 编码的可解释神经连接短语替代
# 源域监督学出的词表混合原型，使 GCN soft token 的 cross-attention
# 键值锚定在语言空间，跨数据集稳定。
NEURAL_SEMANTIC_PROTOTYPES = [
    "strong inter-hemispheric coupling",
    "weak inter-hemispheric coupling",
    "strong left frontal connectivity",
    "strong right frontal connectivity",
    "strong fronto-temporal connectivity",
    "strong temporo-parietal connectivity",
    "strong occipital connectivity",
    "high global network segregation",
    "low global network segregation",
    "high network integration",
    "low network integration",
    "stable connectivity over time",
    "increasing temporal variability",
    "decreasing temporal variability",
    "strong hemispheric asymmetry",
    "reduced connectivity associated with aging",
    "increased slow-wave connectivity",
    "diffuse cortical slowing",
    "focal connectivity deficit",
    "widespread hyperconnectivity",
    "widespread hypoconnectivity",
    "elevated frontal midline connectivity",
    "reduced posterior connectivity",
    "left hemisphere dominant connectivity",
    "right hemisphere dominant connectivity",
    "modular small-world organization",
    "high global efficiency",
    "low global efficiency",
    "strong hub connectivity",
    "weak hub connectivity",
    "coupled frontoparietal networks",
    "disrupted default mode connectivity",
    "enhanced sensorimotor connectivity",
    "reduced sensorimotor connectivity",
    "high clustering coefficient",
    "low clustering coefficient",
    "balanced excitatory-inhibitory connectivity",
    "age-related network reconfiguration",
    "high-frequency oscillatory coupling",
    "low-frequency oscillatory coupling",
    "anterior-posterior gradient of connectivity",
    "short-range dense connectivity",
    "long-range sparse connectivity",
    "task-positive network activation",
    "task-negative network deactivation",
]


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
