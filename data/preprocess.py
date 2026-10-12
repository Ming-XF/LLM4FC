import os

import numpy as np
from scipy.linalg import sqrtm


# ── 年龄回归的固定先验归一化边界 ──
# 不从数据拟合 min/max（避免 test 泄漏），且所有域统一量纲，
# 便于 few-shot / 跨域迁移时 head 输出量纲一致。
AGE_LABEL_MIN = 0.0
AGE_LABEL_MAX = 100.0

EEG_PREPROCESSING_VERSION = "eeg_v2_common19_avgref_200hz"
AGE_EEG_PREPROCESSING_VERSION = "age_eeg_v2_common19_avgref_200hz_fixedage"
EEG_19_CHANNEL_ORDER = (
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F7', 'T3', 'T5',
    'F8', 'T4', 'T6',
    'Fz', 'Cz', 'Pz',
)

_EEG_CHANNEL_ALIASES = {
    'T7': 'T3',
    'P7': 'T5',
    'T8': 'T4',
    'P8': 'T6',
}
_EEG_CANONICAL_NAMES = {
    channel.upper(): channel for channel in EEG_19_CHANNEL_ORDER
}


def normalize_eeg_channel_name(name):
    """Normalize common EDF channel labels to the canonical 10-20 names."""
    normalized = str(name).strip().upper()
    if normalized.startswith('EEG '):
        normalized = normalized[4:].strip()
    for suffix in ('-REF', '-LE', '-AVG', '-AV'):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
            break
    normalized = _EEG_CHANNEL_ALIASES.get(normalized, normalized)
    return _EEG_CANONICAL_NAMES.get(normalized, normalized)


def prepare_common_19_channel_raw(raw, hz):
    """Select, order, average-reference, and resample the common EEG montage."""
    source_names = list(raw.ch_names)
    channel_map = {}
    for channel_name in source_names:
        normalized = normalize_eeg_channel_name(channel_name)
        if normalized not in EEG_19_CHANNEL_ORDER:
            continue
        if normalized in channel_map:
            raise ValueError(
                f"Duplicate channel {normalized!r}: "
                f"{channel_map[normalized]!r} and {channel_name!r}")
        channel_map[normalized] = channel_name

    missing = [
        channel for channel in EEG_19_CHANNEL_ORDER
        if channel not in channel_map
    ]
    if missing:
        raise ValueError(f"Missing required EEG channels: {missing}")

    raw.pick(
        [channel_map[channel] for channel in EEG_19_CHANNEL_ORDER],
        verbose=False,
    )
    raw.set_eeg_reference('average', projection=False, verbose=False)
    if not np.isclose(float(raw.info['sfreq']), float(hz)):
        raw.resample(sfreq=hz, verbose=False)

    data = raw.get_data()
    if data.shape[0] != len(EEG_19_CHANNEL_ORDER):
        raise ValueError(
            f"Expected {len(EEG_19_CHANNEL_ORDER)} EEG channels, "
            f"got {data.shape[0]}")
    return data


def save_npz_atomic(output_path, **arrays):
    """Write an NPZ completely before replacing the destination."""
    output_path = os.fspath(output_path)
    temporary_path = f"{output_path}.tmp-{os.getpid()}.npz"
    try:
        np.savez(temporary_path, **arrays)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def prepare_age_labels(data, source="age dataset"):
    """Return age labels in the repository-wide fixed ``[0, 100]`` scale.

    Some existing NPZ files were generated before fixed age normalization was
    introduced.  Their ``labels`` arrays use dataset-specific min/max bounds,
    even though current training and transfer code assumes that every age head
    predicts ``age / 100``.  ``labels_raw`` is therefore the source of truth.

    Legacy files without ``labels_raw`` can still be read when their original
    bounds are present, because the age in years can be reconstructed exactly.
    In all cases the returned metric bounds are the fixed prior bounds.
    """
    if "labels" not in data:
        raise KeyError(f"{source} is missing the required 'labels' field")

    stored_labels = np.asarray(data["labels"], dtype=np.float32)
    source_min = (float(np.asarray(data["label_min"]).item())
                  if "label_min" in data else None)
    source_max = (float(np.asarray(data["label_max"]).item())
                  if "label_max" in data else None)

    if "labels_raw" in data:
        labels_raw = np.asarray(data["labels_raw"], dtype=np.float32)
    else:
        if (source_min is None or source_max is None
                or not np.isfinite(source_min) or not np.isfinite(source_max)
                or source_max <= source_min):
            raise ValueError(
                f"{source} has no labels_raw and no valid label_min/label_max; "
                "the age coordinate cannot be recovered safely")
        labels_raw = (
            stored_labels * (source_max - source_min) + source_min
        ).astype(np.float32)

    if labels_raw.shape != stored_labels.shape:
        raise ValueError(
            f"{source} labels_raw shape {labels_raw.shape} does not match "
            f"labels shape {stored_labels.shape}")
    if not np.all(np.isfinite(labels_raw)):
        raise ValueError(f"{source} contains non-finite raw age labels")
    if np.any(labels_raw < AGE_LABEL_MIN) or np.any(labels_raw > AGE_LABEL_MAX):
        raise ValueError(
            f"{source} contains ages outside the supported "
            f"[{AGE_LABEL_MIN:g}, {AGE_LABEL_MAX:g}] year range: "
            f"min={labels_raw.min():g}, max={labels_raw.max():g}")

    # If the file records its old coordinate, make sure labels_raw really is
    # consistent with it before overriding that coordinate for current runs.
    if source_min is not None or source_max is not None:
        if (source_min is None or source_max is None
                or not np.isfinite(source_min) or not np.isfinite(source_max)
                or source_max <= source_min):
            raise ValueError(
                f"{source} has invalid label bounds: "
                f"label_min={source_min}, label_max={source_max}")
        expected_stored = (labels_raw - source_min) / (source_max - source_min)
        if not np.allclose(
                stored_labels, expected_stored, rtol=1e-5, atol=1e-6):
            max_error = float(np.max(np.abs(stored_labels - expected_stored)))
            raise ValueError(
                f"{source} labels_raw is inconsistent with labels and the "
                f"recorded bounds (max normalized error={max_error:g})")

    span = AGE_LABEL_MAX - AGE_LABEL_MIN
    labels = ((labels_raw - AGE_LABEL_MIN) / span).astype(np.float32)
    return labels, labels_raw, float(AGE_LABEL_MIN), float(AGE_LABEL_MAX)


def data_norm(data):
    data_copy = np.copy(data)
    for i in range(len(data)):
        data_copy[i] = data_copy[i] / np.maximum(np.max(abs(data[i])), 1e-8)

    return data_copy


def preprocess_ea(data):
    R_bar = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(data)):
        R_bar += np.dot(data[i], data[i].T)
    R_bar_mean = R_bar / len(data)
    R_bar_mean += 1e-6 * np.eye(R_bar_mean.shape[0])

    for i in range(len(data)):
        data[i] = np.dot(np.linalg.inv(sqrtm(R_bar_mean)), data[i])
    return data
