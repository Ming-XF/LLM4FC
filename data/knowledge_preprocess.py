"""Offline feature generation for TimeLLMKnowledge.

The raw task NPZ remains the source of labels, subject identifiers, split
indices, and EEG waveforms.  This module writes a second NPZ with deterministic
FC and relative-PSD features in exactly the order of the raw samples.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


FEATURE_VERSION = "timellmknowledge_v2"
DEFAULT_NUM_WINDOWS = 7
DEFAULT_WINDOW_SECONDS = 30
DEFAULT_STRIDE_SECONDS = 5
DEFAULT_PSD_WINDOW_SECONDS = 4
DEFAULT_PSD_OVERLAP = 0.5
DEFAULT_FMIN = 0.5
DEFAULT_FMAX = 45.0

SUPPORTED_RAW_FILES = (
    "../data/CAUEEG/caueeg_age.npz",
    "../data/CAUEEG/caueeg_disease.npz",
    "../data/TUAB/tuab_age.npz",
    "../data/TUAB/tuab_disease.npz",
    "../data/TUAB/tuab_gender.npz",
    "../data/TUEP/tuep_age.npz",
    "../data/TUEP/tuep_disease.npz",
    "../data/TUEP/tuep_gender.npz",
)


def knowledge_feature_path(raw_npz_path):
    """Return the default sidecar path for a raw task NPZ."""
    raw_path = Path(raw_npz_path)
    return raw_path.with_name(raw_path.stem + "_knowledge_v1.npz")


def _zscore(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


def _correlation_matrix(x):
    """Pearson-style correlation used by the current model implementation."""
    x_z = _zscore(x)
    return torch.bmm(x_z, x_z.transpose(1, 2)) / (x.shape[-1] - 1)


def _relative_psd(x, hz, window_seconds, overlap, fmin, fmax):
    """Welch-like relative PSD matching TimeLLMKnowledge patch_embeds."""
    nperseg = int(window_seconds * hz)
    if nperseg <= 1 or x.shape[-1] < nperseg:
        raise ValueError(
            f"PSD window has {nperseg} samples but input has {x.shape[-1]} samples")
    step = int(round(nperseg * (1.0 - overlap)))
    if step <= 0:
        raise ValueError("PSD overlap must produce a positive step size")

    freqs = torch.fft.rfftfreq(
        nperseg, d=1.0 / hz, device=x.device, dtype=x.dtype)
    mask = (freqs >= fmin) & (freqs < fmax)
    if not bool(mask.any()):
        raise ValueError(
            f"No PSD bins remain for hz={hz}, fmin={fmin}, fmax={fmax}")

    window = torch.hann_window(
        nperseg, periodic=False, device=x.device, dtype=x.dtype)
    rows = []
    for start in range(0, x.shape[-1] - nperseg + 1, step):
        segment = x[..., start:start + nperseg]
        segment = (segment - segment.mean(dim=-1, keepdim=True)) / (
            segment.std(dim=-1, keepdim=True) + 1e-8)
        power = torch.abs(torch.fft.rfft(segment * window, dim=-1)) ** 2
        relative = power / (power.sum(dim=-1, keepdim=True) + 1e-12)
        rows.append(relative[..., mask])
    return torch.stack(rows, dim=0).mean(dim=0), freqs[mask]


def preprocess_timellmknowledge_features(
        input_npz,
        output_npz=None,
        num_windows=DEFAULT_NUM_WINDOWS,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        stride_seconds=DEFAULT_STRIDE_SECONDS,
        psd_window_seconds=DEFAULT_PSD_WINDOW_SECONDS,
        psd_overlap=DEFAULT_PSD_OVERLAP,
        fmin=DEFAULT_FMIN,
        fmax=DEFAULT_FMAX,
        batch_size=16,
        device="cpu",
        overwrite=False):
    """Generate the TimeLLMKnowledge feature sidecar for one raw task NPZ."""
    input_path = Path(input_npz).expanduser().resolve()
    output_path = (Path(output_npz).expanduser().resolve()
                   if output_npz is not None
                   else knowledge_feature_path(input_path))
    if not input_path.exists():
        raise FileNotFoundError(f"Raw NPZ does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Feature NPZ already exists: {output_path}. Use --overwrite to replace it.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    raw = np.load(input_path, allow_pickle=True)
    try:
        required = {"timeseries", "hz", "subject_id"}
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f"Raw NPZ is missing required fields: {sorted(missing)}")
        time_series = raw["timeseries"]
        subject_id = np.asarray(raw["subject_id"])
        hz = int(np.asarray(raw["hz"]).item())
        if time_series.ndim != 3:
            raise ValueError(
                f"timeseries must have shape (samples, channels, time), got {time_series.shape}")
        num_samples, num_channels, num_timepoints = time_series.shape
        if num_samples == 0:
            raise ValueError("timeseries contains no samples")
        if subject_id.shape[0] != num_samples:
            raise ValueError("subject_id length does not match timeseries")

        window_length = int(window_seconds * hz)
        stride = int(stride_seconds * hz)
        required_length = (num_windows - 1) * stride + window_length
        if num_timepoints < required_length:
            raise ValueError(
                f"Need at least {required_length} samples for {num_windows} "
                f"{window_seconds}s windows (stride {stride_seconds}s), "
                f"got {num_timepoints}")

        probe = torch.as_tensor(
            np.asarray(time_series[0:1], dtype=np.float32), device=device)
        probe_psd, retained_freqs = _relative_psd(
            probe, hz, psd_window_seconds, psd_overlap, fmin, fmax)
        num_freqs = probe_psd.shape[-1]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix="timellmknowledge_", dir=output_path.parent) as temp_dir:
            temp_dir = Path(temp_dir)
            window_fc = np.memmap(
                temp_dir / "window_fc.dat", mode="w+", dtype=np.float32,
                shape=(num_samples, num_windows, num_channels, num_channels))
            window_psd = np.memmap(
                temp_dir / "window_psd.dat", mode="w+", dtype=np.float32,
                shape=(num_samples, num_windows, num_channels, num_freqs))

            with torch.no_grad():
                for start in range(0, num_samples, batch_size):
                    stop = min(start + batch_size, num_samples)
                    x = torch.as_tensor(
                        np.asarray(time_series[start:stop], dtype=np.float32),
                        device=device)
                    xw = x[..., :required_length]
                    for k in range(num_windows):
                        s = k * stride
                        win = xw[..., s:s + window_length]
                        w_fc = _correlation_matrix(win)
                        w_psd, _ = _relative_psd(
                            win, hz, psd_window_seconds, psd_overlap,
                            fmin, fmax)
                        window_fc[start:stop, k] = w_fc.cpu().numpy()
                        window_psd[start:stop, k] = w_psd.cpu().numpy()
                    print(f"  processed {stop}/{num_samples}", flush=True)

            for array in (window_fc, window_psd):
                array.flush()

            stat = input_path.stat()
            temporary_output = output_path.with_name(
                output_path.name + f".tmp-{os.getpid()}.npz")
            try:
                source_metadata = {}
                for field in (
                        'preprocessing_version', 'channel_names',
                        'reference', 'unit', 'label_min', 'label_max'):
                    if field in raw.files:
                        source_metadata[f'source_{field}'] = np.asarray(raw[field])
                np.savez_compressed(
                    temporary_output,
                    feature_version=np.asarray(FEATURE_VERSION),
                    source_name=np.asarray(input_path.name),
                    source_size=np.asarray(stat.st_size, dtype=np.int64),
                    source_mtime_ns=np.asarray(stat.st_mtime_ns, dtype=np.int64),
                    source_shape=np.asarray(time_series.shape, dtype=np.int64),
                    hz=np.asarray(hz, dtype=np.int64),
                    num_windows=np.asarray(num_windows, dtype=np.int64),
                    window_seconds=np.asarray(window_seconds, dtype=np.int64),
                    stride_seconds=np.asarray(stride_seconds, dtype=np.int64),
                    psd_window_seconds=np.asarray(
                        psd_window_seconds, dtype=np.int64),
                    psd_overlap=np.asarray(psd_overlap, dtype=np.float32),
                    fmin=np.asarray(fmin, dtype=np.float32),
                    fmax=np.asarray(fmax, dtype=np.float32),
                    frequencies=retained_freqs.cpu().numpy().astype(np.float32),
                    sample_index=np.arange(num_samples, dtype=np.int64),
                    subject_id=subject_id.astype(str),
                    window_fc=window_fc,
                    window_psd=window_psd,
                    window_mask=np.ones(
                        (num_samples, num_windows), dtype=np.bool_),
                    **source_metadata,
                )
                os.replace(temporary_output, output_path)
            finally:
                if temporary_output.exists():
                    temporary_output.unlink()
    finally:
        raw.close()

    print(f"Saved TimeLLMKnowledge features to {output_path}")
    return output_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute FC and relative PSD for TimeLLMKnowledge")
    parser.add_argument("--input", help="one raw task NPZ")
    parser.add_argument("--output", help="optional sidecar output path")
    parser.add_argument(
        "--all-supported", action="store_true",
        help="process all CAUEEG, TUAB, and TUEP task NPZ files that exist")
    parser.add_argument("--num-windows", type=int, default=DEFAULT_NUM_WINDOWS)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=int, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.all_supported:
        if args.input or args.output:
            raise ValueError("--all-supported cannot be combined with --input or --output")
        for raw_path in SUPPORTED_RAW_FILES:
            if not Path(raw_path).exists():
                print(f"Skip missing raw NPZ: {raw_path}")
                continue
            preprocess_timellmknowledge_features(
                raw_path, num_windows=args.num_windows,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                batch_size=args.batch_size, device=args.device,
                overwrite=args.overwrite)
        return
    if not args.input:
        raise ValueError("Specify --input or --all-supported")
    preprocess_timellmknowledge_features(
        args.input, output_npz=args.output, num_windows=args.num_windows,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        batch_size=args.batch_size, device=args.device,
        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
