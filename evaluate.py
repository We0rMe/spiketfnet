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


def write_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_history(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{key: (int(value) if key == "epoch" else float(value)) if value else None
                 for key, value in row.items()} for row in csv.DictReader(handle)]


def plot_training(history, destination, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    monitored = [row for row in history if row.get("monitor_loss") is not None]
    for ax, training_key, monitor_key, label in zip(
            axes, ("loss", "train_acc_percent"), ("monitor_loss", "monitor_acc_percent"), ("Joint loss", "ACC (%)")):
        ax.plot([row["epoch"] for row in history], [row[training_key] for row in history],
                label="Training batches (dropout ON)", marker="." if len(history) < 10 else None)
        if monitored:
            ax.plot([row["epoch"] for row in monitored], [row[monitor_key] for row in monitored], ".-",
                    label="Training subset (dropout OFF; NOT validation)")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[1].set_ylim(0, 100)
    fig.suptitle(title)
    temporary = Path(destination).with_suffix(".tmp")
    try:
        fig.savefig(temporary, format="png", dpi=140)
        temporary.replace(destination)
    finally:
        plt.close(fig)


def plot_confusion(folder, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(Path(folder) / "details/evaluation.npz", allow_pickle=False) as arrays:
        labels, predictions = arrays["labels"], arrays["logits"].argmax(1)
    counts = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    np.add.at(counts, (labels, predictions), 1)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(counts, cmap="Blues", vmin=0)
    ticks = np.arange(0, cfg.num_classes, 5)
    ax.set(xticks=ticks, xticklabels=ticks + 1, yticks=ticks, yticklabels=ticks + 1,
           xlabel="Predicted target (MAT order)", ylabel="True target (MAT order)", title="Final test confusion")
    fig.colorbar(im, ax=ax, label="Trials")
    temporary = Path(folder) / "confusion.tmp"
    try:
        fig.savefig(temporary, format="png", dpi=140)
        temporary.replace(Path(folder) / "confusion.png")
    finally:
        plt.close(fig)


def write_run_report(folder, cfg, history, provenance):
    def number(value):
        return "未记录" if value is None else f"{value:.4f}"
    monitored = [row for row in history if row.get("monitor_loss") is not None]
    last = history[-1] if history else {}
    lines = [f"# SpikeTFNet · S{provenance['test_subject']:02d} · seed {provenance['seed']}",
             "", f"状态：{provenance['status']}；历史记录进度：{last.get('epoch', 0)}/{cfg.epochs} epoch。",
             f"窗口：{provenance['window_ms']} ms；设备：{provenance.get('device', '未记录')}。",
             f"训练被试：{provenance['training_subjects']}；训练 trial：{provenance['training_trials']}；"
             f"测试 trial：{provenance['test_trials']}。",
             "", "| 训练观察 | 最新记录 |", "| --- | ---: |",
             f"| 训练联合 loss（Dropout 开） | {number(last.get('loss'))} |",
             f"| 训练 ACC %（Dropout 开） | {number(last.get('train_acc_percent'))} |",
             f"| 已记录训练用时，不含监测/保存/绘图（分钟） | "
             f"{sum(row.get('train_seconds') or 0 for row in history) / 60:.2f} |"]
    if monitored:
        lines += [f"| 训练子集监测 epoch | {monitored[-1]['epoch']} |",
                  f"| 训练子集监测联合 loss（Dropout 关） | {number(monitored[-1]['monitor_loss'])} |",
                  f"| 训练子集监测 ACC %（Dropout 关） | {number(monitored[-1]['monitor_acc_percent'])} |"]
    if provenance["status"] == "complete":
        metrics = json.loads((Path(folder) / "metrics.json").read_text(encoding="utf-8"))
        lines += ["", "固定末轮模型的测试结果：", "", "| 指标 | 数值 |", "| --- | ---: |",
                  f"| ACC % | {metrics['acc_percent']:.4f} |",
                  f"| ITR bits/min | {metrics['itr_bits_per_minute']:.4f} |",
                  "", "![测试混淆矩阵](confusion.png)"]
    lines += ["", "训练集监测使用源训练被试的第一个 block，**不是独立验证集**，不能用于判断跨被试泛化。",
              "不按监测值早停或选模型。留出被试只在固定最后一轮之后测试。",
              f"监测间隔：{cfg.monitor_every}（0 为关闭）；历史中有监测值的 epoch：{len(monitored)}。",
              "升级前缺少的监测值不回填；旧控制台输出无法恢复，日志中的 RECOVERED 行仅由已保存历史重建。",
              "", "![训练过程](training.png)", "",
              "[训练日志](training.log) · [原始训练历史](details/history.csv) · [配置与源码](../../../details/experiment.json)"]
    write_text(Path(folder) / "report.md", "\n".join(lines) + "\n")


def refresh_progress(root, cfg):
    rows, completed = [], 0
    for window in cfg.windows_ms:
        for subject in cfg.test_subjects:
            for seed in cfg.seeds:
                folder = run_path(root, window, subject, seed)
                path = folder / "details/provenance.json"
                provenance = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                status = provenance.get("status", "pending")
                history = read_history(folder / "details/history.csv")
                epoch = history[-1]["epoch"] if history else 0
                acc, itr = "—", "—"
                if status == "complete":
                    metric = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    acc, itr = f"{metric['acc_percent']:.4f}", f"{metric['itr_bits_per_minute']:.4f}"
                    completed += 1
                relative = folder.relative_to(root).as_posix()
                link = f"[详情]({relative}/report.md)" if (folder / "report.md").exists() else "—"
                rows.append(f"| {window} | S{subject:02d} | {seed} | {status} | {epoch}/{cfg.epochs} | {acc} | {itr} | {link} |")
    total = len(rows)
    text = ["# SpikeTFNet 实验进度", "", f"已完成 **{completed}/{total}** 个任务。",
            "部分结果：未完成项不计为 0，也不将部分任务均值称为完整 LOSO 结果。" if completed < total
            else "全部任务已完成；正式统计见 [summary.csv](summary.csv)，按均值 ± 方差呈现。",
            "", "| 窗口 ms | 测试被试 | seed | 状态 | epoch | ACC % | ITR bits/min | 报告 |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- |", *rows, "",
            "训练曲线的监测线来自训练子集，不是独立验证集。最终测试结果不参与早停或选模。",
            "M=1 时没有独立种子重复，跨重复方差 0 不代表稳定性。",
            "", "[主进程日志](details/experiment.log) · [实验配置及恢复记录](details/experiment.json)"]
    write_text(Path(root) / "report.md", "\n".join(text) + "\n")


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
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu and cuda devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; install the CUDA requirements on the server")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"cuda:{index} requested but only {torch.cuda.device_count()} GPUs are visible")
        device = torch.device("cuda", index)
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
            self.counts[index] = self.counts[index] + output.detach().sum(dtype=torch.int64)
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
                outputs = model(time.to(device, non_blocking=True), spectrum.to(device, non_blocking=True))
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
    arrays.update(spike_layer_names=np.asarray(recorder.names),
                  spike_counts=torch.stack(recorder.counts).cpu().numpy(),
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
        cfg = replace(cfg, evaluation_device=device_name)
    if file_sha256(Path(__file__).with_name("model.py")) != manifest["code_sha256"]["model.py"]:
        raise ValueError("Model source changed; use the archived source to replay this experiment")
    prepared = prepare_data(cfg) if prepared is None else prepared.validate(cfg)
    if prepared.metadata["fingerprint"] != manifest["dataset_fingerprint"]:
        raise ValueError("Dataset/preprocessing differs from the training experiment")
    lookup = {tuple(identity): i for i, identity in enumerate(prepared.ids)}
    device = resolve_device(cfg.evaluation_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
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
                history = read_history(folder / "details/history.csv")
                provenance = json.loads((folder / "details/provenance.json").read_text(encoding="utf-8"))
                plot_confusion(folder, cfg)
                write_run_report(folder, cfg, history, provenance)
                del model
    summary = summarize_experiment(root, cfg)
    refresh_progress(root, cfg)
    return summary


if __name__ == "__main__":
    reevaluate_experiment(CONFIG.output_root / CONFIG.experiment_name, data_root=CONFIG.data_root,
                         device_name=CONFIG.evaluation_device)
