"""最终模型评估与多种子汇总；不使用目标被试选择 checkpoint。"""

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import neuron
from torch.utils.data import DataLoader

from config import CONFIG, Config
from data import WindowDataset, file_sha256, prepare_data
from model import SpikeTFNet


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def itr_bits_per_minute(accuracy, num_classes, selection_seconds):
    if not 0 <= accuracy <= 1 or num_classes < 2 or selection_seconds <= 0:
        raise ValueError("Invalid ITR inputs")
    if accuracy <= 1.0 / num_classes:
        return 0.0
    bits = math.log2(num_classes)
    if accuracy < 1:
        bits += accuracy * math.log2(accuracy)
        bits += (1 - accuracy) * math.log2((1 - accuracy) / (num_classes - 1))
    return max(0.0, 60.0 * bits / selection_seconds)


class SpikeRecorder:
    """只记录计数供后续 SOP 分析；不在此计算效率指标或测量时延。"""

    def __init__(self, model):
        self.names, self.counts, self.elements, self.handles = [], [], [], []
        for name, layer in model.named_modules():
            if isinstance(layer, neuron.LIFNode):
                index = len(self.names)
                self.names.append(name)
                self.counts.append(0)
                self.elements.append(0)
                self.handles.append(layer.register_forward_hook(self._hook(index)))

    def _hook(self, index):
        def record(module, inputs, output):
            self.counts[index] += int(output.detach().sum().item())
            self.elements[index] += output.numel()
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()


def evaluate_model(model, loader, cfg, window_ms, device):
    model.eval()
    collected = {name: [] for name in ("logits", "spectral_logits", "temporal_logits", "gate", "embedding", "labels")}
    recorder = SpikeRecorder(model)
    probes_time, probes_spectrum = [], []
    probe_count = 0
    try:
        with torch.inference_mode():
            for time, spectrum, labels in loader:
                remaining = cfg.probe_trials - probe_count
                if remaining > 0:
                    probes_time.append(time[:remaining].numpy().copy())
                    probes_spectrum.append(spectrum[:remaining].numpy().copy())
                    probe_count += len(probes_time[-1])
                outputs = model(time.to(device), spectrum.to(device))
                for name in collected:
                    value = labels if name == "labels" else outputs[name]
                    collected[name].append(value.detach().cpu().numpy())
    finally:
        recorder.close()
    if not collected["labels"]:
        raise ValueError("Empty evaluation loader")
    arrays = {name: np.concatenate(values) for name, values in collected.items()}
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise FloatingPointError("Nonfinite model evaluation output")
    accuracy = float(np.mean(arrays["logits"].argmax(axis=1) == arrays["labels"]))
    metrics = {"acc_percent": 100 * accuracy,
               "itr_bits_per_minute": itr_bits_per_minute(accuracy, cfg.num_classes,
                                                           window_ms / 1000 + cfg.gaze_shift_seconds)}
    arrays.update(spike_layer_names=np.asarray(recorder.names), spike_counts=np.asarray(recorder.counts, dtype=np.int64),
                  spike_elements=np.asarray(recorder.elements, dtype=np.int64),
                  probe_time=np.concatenate(probes_time), probe_spectrum=np.concatenate(probes_spectrum))
    return metrics, arrays


def save_evaluation(run_dir, metrics, arrays, training_ids, test_ids):
    run_dir = Path(run_dir)
    if not np.array_equal(arrays["labels"], test_ids[:, 2]):
        raise ValueError("Prediction order does not match trial IDs")
    destination = run_dir / "details" / "evaluation.npz"
    with destination.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, **arrays, training_ids=training_ids, test_ids=test_ids)
    destination.with_suffix(".tmp").replace(destination)
    write_json(run_dir / "metrics.json", metrics)


def run_path(root, window_ms, subject, seed):
    return Path(root) / f"window_{window_ms:04d}ms" / f"test_S{subject:02d}" / f"seed_{seed}"


def aggregate_matrix(matrix, ddof):
    """matrix[subject, seed]；主统计是各 seed 的 LOSO 均值之间的方差。"""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Invalid subject-by-seed metric matrix")

    def stats(values):
        return {"count": len(values), "mean": float(np.mean(values)),
                "variance": float(np.var(values, ddof=ddof)) if len(values) > ddof else None}

    return {"across_repeats": stats(matrix.mean(axis=0)),
            "across_subjects": stats(matrix.mean(axis=1)),
            "subjects": [stats(row) for row in matrix],
            "repeats": [stats(column) for column in matrix.T]}


def summarize_experiment(root, cfg):
    rows = []
    for window in cfg.windows_ms:
        values = {metric: np.zeros((cfg.N, cfg.M)) for metric in ("acc_percent", "itr_bits_per_minute")}
        for i, subject in enumerate(cfg.test_subjects):
            for j, seed in enumerate(cfg.seeds):
                folder = run_path(root, window, subject, seed)
                provenance = json.loads((folder / "details" / "provenance.json").read_text(encoding="utf-8"))
                if provenance["status"] != "complete" or provenance["test_subject"] != subject or provenance["seed"] != seed:
                    raise ValueError(f"Incomplete or mismatched run: {folder}")
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                for metric in values:
                    values[metric][i, j] = metrics[metric]
        for metric, matrix in values.items():
            stats = aggregate_matrix(matrix, cfg.variance_ddof)
            groups = [("across_repeats", "all", stats["across_repeats"]),
                      ("across_subjects", "all", stats["across_subjects"])]
            groups += [("within_subject", f"S{s:02d}", st) for s, st in zip(cfg.test_subjects, stats["subjects"])]
            groups += [("within_repeat", str(seed), st) for seed, st in zip(cfg.seeds, stats["repeats"])]
            for scope, member, item in groups:
                variance = item["variance"]
                display = f"{item['mean']:.6f} ± {variance:.6f}" if variance is not None else f"{item['mean']:.6f} ± NA"
                rows.append({"window_ms": window, "metric": metric, "scope": scope, "member": member,
                             **item, "ddof": cfg.variance_ddof, "mean_plus_variance": display})
            main = rows[-len(groups)]["mean_plus_variance"]
            print(f"{window} ms | {metric} | across repeats: {main}", flush=True)
    destination = Path(root) / "summary.csv"
    with destination.with_suffix(".tmp").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    destination.with_suffix(".tmp").replace(destination)
    return rows


def reevaluate_experiment(root, prepared=None, data_root=None, device_name=None):
    root = Path(root)
    manifest = json.loads((root / "details" / "experiment.json").read_text(encoding="utf-8"))
    cfg = Config.from_dict(manifest["config"])
    if data_root is not None:
        cfg = replace(cfg, data_root=Path(data_root))
    if device_name is not None:
        cfg = replace(cfg, device=device_name)
    if file_sha256(Path(__file__).with_name("model.py")) != manifest["code_sha256"]["model.py"]:
        raise ValueError("Model source changed; use the archived source to replay this experiment")
    prepared = prepare_data(cfg) if prepared is None else prepared.validate(cfg)
    if prepared.metadata["fingerprint"] != manifest["dataset_fingerprint"]:
        raise ValueError("Dataset/preprocessing differs from the training experiment")
    lookup = {tuple(identity): i for i, identity in enumerate(prepared.ids)}
    device = resolve_device(cfg.device)
    torch.set_num_threads(cfg.cpu_threads)
    for window in cfg.windows_ms:
        for subject in cfg.test_subjects:
            for seed in cfg.seeds:
                folder = run_path(root, window, subject, seed)
                with np.load(folder / "details" / "evaluation.npz", allow_pickle=False) as archive:
                    training_ids, test_ids = archive["training_ids"], archive["test_ids"]
                if set(training_ids[:, 0]) & set(test_ids[:, 0]) or set(test_ids[:, 0]) != {subject}:
                    raise ValueError("Persisted split leaks the held-out subject")
                rows = np.array([lookup[tuple(identity)] for identity in test_ids])
                dataset = WindowDataset(prepared, cfg, window, rows)
                loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
                model = SpikeTFNet(cfg).to(device)
                model.load_state_dict(torch.load(folder / "details" / "model.pt", map_location=device, weights_only=True))
                metrics, arrays = evaluate_model(model, loader, cfg, window, device)
                save_evaluation(folder, metrics, arrays, training_ids, test_ids)
                del model
    return summarize_experiment(root, cfg)


if __name__ == "__main__":
    reevaluate_experiment(CONFIG.output_root / CONFIG.experiment_name, data_root=CONFIG.data_root, device_name=CONFIG.device)
