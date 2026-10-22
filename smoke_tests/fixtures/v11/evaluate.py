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

from config import CONFIG, PROJECT_ROOT, Config
from data import WindowDataset, file_sha256, prepare_data
from experiments.variants import build_model
from utils.timing import normalize_history
from utils.branch_timing import timing_markdown, timing_rows, timing_totals
from utils.provenance import CONV_EXECUTION_V9_SOURCES, verified_source_upgrade, verified_line_endings


def write_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_history(path, include_diagnostics=False):
    """常规绘图/跨折汇总只读数值历史，避免加载所有折的细粒度诊断。"""
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [{key: (int(value) if key in ("epoch", "train_trials") else float(value)) if value else None
                 for key, value in row.items()} for row in csv.DictReader(handle)]
    provenance = Path(path).parent / "provenance.json"
    count = json.loads(provenance.read_text(encoding="utf-8")).get("training_trials") if provenance.exists() else None
    detail_path = Path(path).with_name("t2_diagnostics.json")
    if include_diagnostics and detail_path.exists():
        diagnostics = json.loads(detail_path.read_text(encoding="utf-8"))
        for row in rows:
            if str(row["epoch"]) in diagnostics:
                row["_t2_diagnostics"] = diagnostics[str(row["epoch"])]
    return normalize_history(rows, count)


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
    folder = Path(folder)
    last = history[-1] if history else {}
    monitored = [row for row in history if row.get("monitor_loss") is not None]
    complete = provenance["status"] == "complete"
    lines = [f"# SpikeTFNet · S{provenance['test_subject']:02d} · 测试结果" if complete
             else f"# SpikeTFNet · S{provenance['test_subject']:02d} · 训练进度", "",
             f"{cfg.model_variant}；seed {provenance['seed']}；{provenance['window_ms']} ms；"
             f"{last.get('epoch', 0)}/{cfg.epochs} epoch；{provenance['status']}。", "",
             "| 指标 | 结果 |", "| --- | ---: |"]
    if complete:
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        count = provenance["test_trials"]
        lines += [f"| 测试 ACC | {metrics['acc_percent']:.4f}% |",
                  f"| 测试 ITR | {metrics['itr_bits_per_minute']:.4f} bits/min |",
                  f"| 正确 / 总试次 | {round(metrics['acc_percent'] * count / 100)} / {count} |"]
    if last:
        lines += [f"| 最后一轮训练 loss | {last['loss']:.4f} |",
                  f"| 最后一轮训练 ACC（Dropout 开） | {last['train_acc_percent']:.2f}% |"]
    if monitored:
        monitor = monitored[-1]
        lines += [f"| 训练子集监测 ACC（Dropout 关，epoch {monitor['epoch']}） | {monitor['monitor_acc_percent']:.2f}% |"]
    lines += ["", f"训练 {len(provenance['training_subjects'])} 名源被试，S{provenance['test_subject']:02d} 留出测试。"
              "固定末轮模型；训练子集监测不是验证集。"]
    lines += timing_markdown(timing_totals([(cfg.model_variant, history)], cfg.epochs))
    lines += ["", "![训练过程](training.png)"]
    if complete:
        lines += ["", "![测试混淆矩阵](confusion.png)"]
    lines += ["", "[训练日志](training.log) · [逐轮记录](details/history.csv) · "
              "[配置与源码](../../../details/experiment.json)"]
    write_text(folder / "report.md", "\n".join(lines) + "\n")


def refresh_progress(root, cfg):
    root = Path(root)
    rows, completed, timing_runs = [], 0, []
    results = {}
    for window in cfg.windows_ms:
        for subject in cfg.test_subjects:
            for seed in cfg.seeds:
                folder = run_path(root, window, subject, seed)
                path = folder / "details/provenance.json"
                provenance = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                status = provenance.get("status", "pending")
                history = read_history(folder / "details/history.csv")
                timing_runs.append((cfg.model_variant, history))
                epoch = history[-1]["epoch"] if history else 0
                acc, itr = "—", "—"
                if status == "complete":
                    if (provenance.get("test_subject"), provenance.get("seed"), provenance.get("window_ms")) != (subject, seed, window):
                        raise ValueError(f"Mismatched completed run: {folder}")
                    metric = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    results[window, subject, seed] = metric
                    acc, itr = f"{metric['acc_percent']:.2f}", f"{metric['itr_bits_per_minute']:.2f}"
                    completed += 1
                relative = folder.relative_to(root).as_posix()
                link = f"[详情]({relative}/report.md)" if (folder / "report.md").exists() else "—"
                rows.append(f"| {window} | S{subject:02d} | {seed} | {status} | {epoch}/{cfg.epochs} | {acc} | {itr} | {link} |")
    total = len(rows)
    text = ["# SpikeTFNet · LOSO 汇总", "", f"完成 **{completed}/{total}** 次实验；{cfg.model_variant}，N={cfg.N}，M={cfg.M}。",
            "完整计划已完成。" if completed == total else "尚未完成；下表仅使用已完成测试，未完成项不补零。",
            "", f"下表为同一种子下的跨被试 **均值 ± 方差**（ddof={cfg.variance_ddof}，不是标准差）；单个观测记 NA。",
            "", "| 窗口 ms | 种子 | 完成被试 / 计划 | 测试 ACC (%) | 测试 ITR (bits/min) |",
            "| --- | --- | ---: | ---: | ---: |"]
    for window in cfg.windows_ms:
        for seed in cfg.seeds:
            metrics = [results[window, subject, seed] for subject in cfg.test_subjects if (window, subject, seed) in results]
            displays = [format_stat(metric_stats([m[key] for m in metrics], cfg.variance_ddof)) if metrics else "—"
                        for key in ("acc_percent", "itr_bits_per_minute")]
            text.append(f"| {window} | {seed} | {len(metrics)} / {cfg.N} | {displays[0]} | {displays[1]} |")
    if cfg.M > 1:
        for window in cfg.windows_ms:
            if not all((window, subject, seed) in results for subject in cfg.test_subjects for seed in cfg.seeds):
                continue
            text += ["", f"{window} ms：先对每名被试的 {cfg.M} 个种子取均值，再汇总被试间差异："]
            for key, label in (("acc_percent", "ACC (%)"), ("itr_bits_per_minute", "ITR (bits/min)")):
                matrix = [[results[window, subject, seed][key] for seed in cfg.seeds] for subject in cfg.test_subjects]
                stats = aggregate_matrix(matrix, cfg.variance_ddof)
                text += [f"- {label}：{format_stat(stats['across_subjects'])}；"
                         f"种子 LOSO 均值间：{format_stat(stats['across_repeats'])}。"]
    else:
        text += ["", "M=1：没有独立种子重复，种子间方差为 NA；上表反映被试间差异。"]
    text += ["", "ITR 先逐被试计算再平均。折间训练集重叠，不将所有折/种子视为独立重复。",
             "", "<details>", "<summary>分折结果与训练进度</summary>", "",
             "| 窗口 ms | 测试被试 | seed | 状态 | epoch | ACC % | ITR bits/min | 报告 |",
             "| --- | --- | --- | --- | --- | ---: | ---: | --- |", *rows, "", "</details>"]
    text += timing_markdown(timing_totals(timing_runs, cfg.epochs))
    links = ["[实验日志](details/experiment.log)", "[配置与复现资料](details/experiment.json)"]
    if (root / "summary.csv").exists():
        links.insert(0, "[完整统计](summary.csv)")
    text += ["", " · ".join(links)]
    write_text(root / "report.md", "\n".join(text) + "\n")
    return completed == total


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
    names = ["logits", "embedding", "labels"]
    if cfg.model_variant != "temporal_only":
        names.append("spectral_logits")
    if cfg.model_variant != "spectral_only":
        names.append("temporal_logits")
    if cfg.model_variant == "full":
        names.append("gate")
    collected = {name: [] for name in names}
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
                  spike_counts=(torch.stack(recorder.counts).cpu().numpy() if recorder.counts
                                else np.empty(0, dtype=np.int64)),
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


def metric_stats(values, ddof):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or ddof not in (0, 1):
        raise ValueError("Invalid metric observations or variance ddof")
    return {"count": len(values), "mean": float(np.mean(values)),
            "variance": float(np.var(values, ddof=ddof)) if len(values) >= 2 else None}


def format_stat(item, digits=2):
    variance = f"{item['variance']:.{digits}f}" if item["variance"] is not None else "NA"
    return f"{item['mean']:.{digits}f} ± {variance}"


def aggregate_matrix(matrix, ddof):
    """matrix[subject, seed]；分开报告跨被试与跨种子差异。"""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Invalid subject-by-seed metric matrix")

    def stats(values):
        return metric_stats(values, ddof)

    return {"across_repeats": stats(matrix.mean(axis=0)),
            "across_subjects": stats(matrix.mean(axis=1)),
            "subjects": [stats(row) for row in matrix],
            "repeats": [stats(column) for column in matrix.T]}


def summarize_experiment(root, cfg):
    rows = []
    all_timing_runs = []
    for window in cfg.windows_ms:
        window_timing_runs = []
        values = {metric: np.zeros((cfg.N, cfg.M)) for metric in ("acc_percent", "itr_bits_per_minute")}
        for i, subject in enumerate(cfg.test_subjects):
            for j, seed in enumerate(cfg.seeds):
                folder = run_path(root, window, subject, seed)
                provenance = json.loads((folder / "details" / "provenance.json").read_text(encoding="utf-8"))
                if provenance["status"] != "complete" or provenance["test_subject"] != subject or provenance["seed"] != seed:
                    raise ValueError(f"Incomplete or mismatched run: {folder}")
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                window_timing_runs.append((cfg.model_variant, read_history(folder / "details/history.csv")))
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
            main = format_stat(stats["across_subjects"], 6)
            print(f"{window} ms | {metric} | across subjects (mean ± variance, N={cfg.N}, M={cfg.M}): {main}", flush=True)
        rows.extend(timing_rows(timing_totals(window_timing_runs, cfg.epochs), window, member="all"))
        all_timing_runs.extend(window_timing_runs)
    rows.extend(timing_rows(timing_totals(all_timing_runs, cfg.epochs), "all", member="all"))
    destination = Path(root) / "summary.csv"
    with destination.with_suffix(".tmp").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    destination.with_suffix(".tmp").replace(destination)
    return rows


def refresh_saved_reports(root):
    """仅重建现有报告和统计，不加载 EEG、模型或覆盖原训练日志。"""
    root = Path(root)
    manifest = json.loads((root / "details/experiment.json").read_text(encoding="utf-8"))
    cfg = Config.from_dict(manifest["config"])
    for window in cfg.windows_ms:
        for subject in cfg.test_subjects:
            for seed in cfg.seeds:
                folder = run_path(root, window, subject, seed)
                path = folder / "details/provenance.json"
                if path.exists():
                    provenance = json.loads(path.read_text(encoding="utf-8"))
                    write_run_report(folder, cfg, read_history(folder / "details/history.csv"), provenance)
    if refresh_progress(root, cfg):
        summarize_experiment(root, cfg)
        refresh_progress(root, cfg)


def reevaluate_experiment(root, prepared=None, data_root=None, device_name=None):
    root = Path(root)
    manifest = json.loads((root / "details" / "experiment.json").read_text(encoding="utf-8"))
    cfg = Config.from_dict(manifest["config"])
    if data_root is not None:
        cfg = replace(cfg, data_root=Path(data_root))
    if device_name is not None:
        cfg = replace(cfg, evaluation_device=device_name)
    if file_sha256(Path(__file__).with_name("model.py")) != manifest["code_sha256"]["model.py"]:
        known_native = (manifest.get("format_version", 1) <= 9 and cfg.t2_conv_execution == "native"
                        and verified_source_upgrade("model.py", manifest["code_sha256"]["model.py"],
                                                    root / "details", CONV_EXECUTION_V9_SOURCES))
        same_code = verified_line_endings("model.py", "model.py", manifest["code_sha256"]["model.py"],
                                         file_sha256(Path(__file__).with_name("model.py")), root / "details")
        if not known_native and not same_code:
            raise ValueError("Model source changed; use the archived source to replay this experiment")
    variant_key = "experiments/variants.py" if manifest.get("format_version", 1) >= 6 else "ablation_model.py"
    if manifest.get("format_version", 1) >= 4 and file_sha256(
            PROJECT_ROOT / "experiments/variants.py") != manifest["code_sha256"][variant_key] and not verified_line_endings(
                variant_key, "experiments/variants.py", manifest["code_sha256"][variant_key],
                file_sha256(PROJECT_ROOT / "experiments/variants.py"), root / "details"):
        raise ValueError("Ablation model source changed; use the archived source")
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
                model = build_model(cfg).to(device)
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
