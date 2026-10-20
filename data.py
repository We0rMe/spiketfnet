"""Benchmark 读取、独立 trial 滤波、裁窗及严格按被试划分。"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from scipy.io import loadmat
from torch.utils.data import Dataset

from config import CONFIG, Config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channel_indices(cfg):
    rows = (cfg.data_root / "64-channels.loc").read_text().splitlines()
    mapping = {row.split()[-1].upper(): int(row.split()[0]) - 1 for row in rows if row.strip()}
    indices = [mapping[name.upper()] for name in cfg.channels]
    if len(set(indices)) != len(indices) or not all(0 <= i < cfg.raw_channels for i in indices):
        raise ValueError("Invalid channel mapping")
    return indices


def preprocessing_settings(cfg):
    names = ("subject_ids", "sampling_rate", "raw_channels", "raw_samples", "num_classes", "num_blocks",
             "channels", "cue_seconds", "latency_seconds", "filter_order", "band_hz", "zero_phase",
             "extra_notch", "notch_hz", "notch_q")
    return {**{name: getattr(cfg, name) for name in names}, "max_window_ms": max(cfg.windows_ms)}


def preprocess_subject(raw, cfg, indices):
    expected = (cfg.raw_channels, cfg.raw_samples, cfg.num_classes, cfg.num_blocks)
    if raw.shape != expected or not np.isfinite(raw).all():
        raise ValueError(f"Malformed or nonfinite EEG: expected {expected}, got {raw.shape}")
    # Block-major、target-minor；标签与原 MAT target 轴完全一致。
    trials = raw[indices].transpose(3, 2, 0, 1).reshape(-1, len(indices), cfg.raw_samples)
    if cfg.extra_notch:
        b, a = signal.iirnotch(cfg.notch_hz, cfg.notch_q, fs=cfg.sampling_rate)
        apply = signal.filtfilt if cfg.zero_phase else signal.lfilter
        trials = apply(b, a, trials, axis=-1)
    sos = signal.butter(cfg.filter_order, cfg.band_hz, btype="bandpass", fs=cfg.sampling_rate, output="sos")
    apply = signal.sosfiltfilt if cfg.zero_phase else signal.sosfilt
    trials = apply(sos, trials, axis=-1)
    end = cfg.start_sample + cfg.samples(max(cfg.windows_ms))
    return np.ascontiguousarray(trials[..., cfg.start_sample:end], dtype=np.float32)


def complex_spectrum(time_data, cfg):
    spectrum = np.fft.rfft(time_data, n=cfg.fft_length, axis=-1)
    if cfg.fft_scale == "two_over_length":
        spectrum *= 2.0 / time_data.shape[-1]
    lo, hi = (round(f / cfg.fft_resolution) for f in cfg.spectrum_hz)
    spectrum = spectrum[..., lo:hi + int(cfg.include_upper_bin)]
    return np.ascontiguousarray(np.concatenate((spectrum.real, spectrum.imag), axis=-1), dtype=np.float32)


@dataclass
class PreparedData:
    time: np.ndarray
    ids: np.ndarray  # columns: subject (1-based), block (1-based), target (0-based)
    frequencies: np.ndarray
    phases: np.ndarray
    metadata: dict

    def validate(self, cfg):
        expected = (len(self.ids), len(cfg.channels), cfg.samples(max(cfg.windows_ms)))
        if self.time.shape != expected or self.ids.shape != (len(self.time), 3):
            raise ValueError("Prepared data dimensions do not match the configuration")
        if not np.isfinite(self.time).all() or not np.isfinite(self.frequencies).all() or not np.isfinite(self.phases).all():
            raise ValueError("Prepared data contains nonfinite values")
        if len(np.unique(self.ids, axis=0)) != len(self.ids):
            raise ValueError("Duplicate trial identities")
        if set(self.ids[:, 0]) != set(cfg.subject_ids):
            raise ValueError("Prepared subject pool differs from configuration")
        if not np.all((self.ids[:, 1] >= 1) & (self.ids[:, 1] <= cfg.num_blocks)):
            raise ValueError("Invalid block IDs")
        if not np.all((self.ids[:, 2] >= 0) & (self.ids[:, 2] < cfg.num_classes)):
            raise ValueError("Invalid class IDs")
        if self.frequencies.shape != (cfg.num_classes,) or self.phases.shape != (cfg.num_classes,):
            raise ValueError("Invalid frequency/phase mapping")
        return self

    def subset(self, rows):
        return PreparedData(self.time[rows], self.ids[rows], self.frequencies, self.phases, self.metadata)


def prepare_data(cfg=CONFIG):
    cfg.validate()
    files = [cfg.data_root / f"S{s}.mat" for s in cfg.subject_ids]
    files += [cfg.data_root / "Freq_Phase.mat", cfg.data_root / "64-channels.loc"]
    source_hashes = {p.name: file_sha256(p) for p in files}
    identity = {"sources": source_hashes, "preprocessing": preprocessing_settings(cfg), "code": file_sha256(__file__)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_file = cfg.cache_root / "benchmark_prepared.npz"
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["fingerprint"] == fingerprint:
                print("Using verified preprocessing cache.", flush=True)
                return PreparedData(archive["time"], archive["ids"], archive["frequencies"], archive["phases"], meta).validate(cfg)

    indices = channel_indices(cfg)
    mapping = loadmat(cfg.data_root / "Freq_Phase.mat")
    all_time, all_ids = [], []
    for subject in cfg.subject_ids:
        path = cfg.data_root / f"S{subject}.mat"
        raw = loadmat(path, variable_names=["data"])["data"]
        all_time.append(preprocess_subject(raw, cfg, indices))
        del raw
        all_ids.append(np.array([(subject, block, target) for block in range(1, cfg.num_blocks + 1)
                                 for target in range(cfg.num_classes)], dtype=np.int16))
        print(f"Prepared S{subject:02d}: {len(all_ids[-1])} trials", flush=True)
    meta = {"fingerprint": fingerprint, **identity, "channel_indices_zero_based": indices,
            "normalization": "none; BatchNorm fitted only on source training batches"}
    prepared = PreparedData(np.concatenate(all_time), np.concatenate(all_ids), mapping["freqs"].reshape(-1),
                            mapping["phases"].reshape(-1), meta).validate(cfg)
    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, time=prepared.time, ids=prepared.ids, frequencies=prepared.frequencies,
                            phases=prepared.phases, metadata=json.dumps(meta))
    temporary.replace(cache_file)
    return prepared


def loso_indices(ids, held_out, available_subjects):
    if held_out not in available_subjects or set(ids[:, 0]) != set(available_subjects):
        raise ValueError("LOSO subject pool mismatch")
    train = np.flatnonzero(ids[:, 0] != held_out)
    test = np.flatnonzero(ids[:, 0] == held_out)
    if not len(train) or not len(test) or set(ids[train, 0]) & set(ids[test, 0]):
        raise ValueError("Empty or leaking LOSO split")
    return train, test


class WindowDataset(Dataset):
    def __init__(self, prepared, cfg, window_ms, rows=None):
        if window_ms not in cfg.windows_ms:
            raise ValueError("Window is not part of this experiment")
        selected = prepared if rows is None else prepared.subset(rows)
        self.time = torch.from_numpy(np.ascontiguousarray(selected.time[..., :cfg.samples(window_ms)]))
        self.spectrum = torch.from_numpy(complex_spectrum(self.time.numpy(), cfg))
        self.labels = torch.tensor(selected.ids[:, 2], dtype=torch.long)
        self.ids = selected.ids.copy()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.time[index], self.spectrum[index], self.labels[index]


if __name__ == "__main__":
    prepared_data = prepare_data()
    print(f"Prepared {len(prepared_data.ids)} trials; cache: {CONFIG.cache_root}")
