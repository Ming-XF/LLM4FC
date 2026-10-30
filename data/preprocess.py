import os

import numpy as np
from scipy.linalg import sqrtm


EEG_PREPROCESSING_VERSION = "eeg_v2_common19_avgref_200hz"
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
