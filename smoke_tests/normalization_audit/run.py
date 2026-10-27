"""固定服务器权重，只用完整源训练池重估 BN；所有产物与正式实验隔离。"""

import copy
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from spikingjelly.activation_based import base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from data import file_sha256, prepare_data
from evaluate import itr_bits_per_minute
from experiments.variants import build_model

EXPERIMENT = ROOT / "results_server/temporal_tuning_v1_500s/temporal_only"
DATA_ROOT = Path(r"G:\SSVEP\data\benchmark")
OUTPUT = HERE / "results"
SUBJECTS = tuple(range(1, 8))
SEEDS = (42, 43)
BATCH = 128
THREADS = 4
ORDER_SEED = 20261026
MODES = ("original", "shared_reestimated", "per_step_reestimated")
CORE = ("config.py", "utils/configuration.py", "data.py", "model.py", "train.py", "evaluate.py",
        "experiments/variants.py", "utils/provenance.py")


def digest_tensors(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Moments:
    """按实际元素数合并总体方差；末批96条不会与128条等权。"""

    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, None

    def add(self, count, mean, variance):
        mean, variance = mean.double(), variance.double()
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean.clone(), variance * count
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += variance * count + delta.square() * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def tensors(self):
        return self.mean.float(), (self.m2 / self.count).clamp_min(0).float()


class StepStats(base.MemoryModule):
    """仅用于推理诊断：四个调用各用一套统计，仿射参数仍共享。"""

    def __init__(self, original, means, variances):
        super().__init__()
        self.eps = original.eps
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())
        self.register_buffer("running_mean", means.clone())
        self.register_buffer("running_var", variances.clone())
        self.register_memory("index", 0)

    def forward(self, x):
        if self.training or self.index >= len(self.running_mean):
            raise RuntimeError("StepStats requires frozen evaluation and reset state")
        step = self.index
        self.index += 1
        return nn.functional.batch_norm(x, self.running_mean[step], self.running_var[step],
                                        self.weight, self.bias, False, 0., self.eps)


def forward(model, x, cfg):
    # temporal_only 已删除频域分支；占位频谱不参与计算。
    return model(x, x.new_zeros((len(x), len(cfg.channels), cfg.spectral_features)))


def estimate(model, source_time, cfg, order_seed=ORDER_SEED):
    """仅接收源输入，不接受标签或测试输入；BN用批统计前向，其余层保持eval。"""
    net = copy.deepcopy(model).eval()
    before = digest_tensors(net.named_parameters())
    moments, indices, handles = {}, {}, []

    def capture(name):
        def hook(module, inputs):
            step = indices.get(name, 0)
            indices[name] = step + 1
            x = inputs[0].detach()
            variance, mean = torch.var_mean(x, dim=(0, 2), unbiased=False)
            moments.setdefault((name, step), Moments()).add(x.shape[0] * x.shape[2], mean, variance)
        return hook

    names = []
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            names.append(name)
            module.train()
            handles.append(module.register_forward_pre_hook(capture(name)))
        elif isinstance(module, nn.Dropout):
            assert not module.training
    order = np.random.default_rng(order_seed).permutation(len(source_time))
    try:
        with torch.inference_mode():
            for start in range(0, len(order), BATCH):
                indices.clear()
                out = forward(net, source_time[order[start:start + BATCH]], cfg)
                assert torch.isfinite(out["logits"]).all()
                assert all(indices[name] == (1 if name == "temporal.current.1" else cfg.simulation_steps)
                           for name in names)
    finally:
        for handle in handles:
            handle.remove()
    assert digest_tensors(net.named_parameters()) == before
    expected_count = len(source_time) * source_time.shape[-1]
    assert all(moment.count == expected_count for moment in moments.values())
    statistics = {}
    for name in names:
        groups = [moments[name, step] for step in range(1 if name == "temporal.current.1" else cfg.simulation_steps)]
        pooled = Moments()
        for group in groups:
            pooled.add(group.count, group.mean, group.m2 / group.count)
        statistics[name] = {"step_mean": torch.stack([g.tensors()[0] for g in groups]),
                            "step_var": torch.stack([g.tensors()[1] for g in groups]),
                            "mean": pooled.tensors()[0], "var": pooled.tensors()[1],
                            "count_per_step": expected_count}
    return statistics


def apply_statistics(original, statistics, per_step):
    model = copy.deepcopy(original)
    for name, stats in statistics.items():
        module = model.get_submodule(name)
        if per_step and len(stats["step_mean"]) > 1:
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child, StepStats(module, stats["step_mean"], stats["step_var"]))
        else:
            module.running_mean.copy_(stats["mean"])
            module.running_var.copy_(stats["var"])
    model.eval()
    assert digest_tensors(model.named_parameters()) == digest_tensors(original.named_parameters())
    return model


def score(model, time_data, labels, cfg):
    model.eval()
    buffers = digest_tensors(model.named_buffers())
    parts = []
    with torch.inference_mode():
        for start in range(0, len(time_data), BATCH):
            parts.append(forward(model, time_data[start:start + BATCH], cfg)["logits"].cpu().numpy())
    assert buffers == digest_tensors(model.named_buffers())
    logits = np.concatenate(parts)
    assert np.isfinite(logits).all()
    acc = float((logits.argmax(1) == labels).mean())
    return {"acc_percent": 100 * acc,
            "itr_bits_per_minute": itr_bits_per_minute(acc, cfg.num_classes, .5 + cfg.gaze_shift_seconds)}, logits


def save(evidence, predictions, statistics):
    details = OUTPUT / "details"
    for name, payload in (("evidence.json", evidence),):
        target = details / name
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
    target = details / "predictions.npz"
    with target.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **predictions)
    target.with_suffix(".tmp").replace(target)
    target = details / "statistics.pt"
    torch.save(statistics, target.with_suffix(".tmp"))
    target.with_suffix(".tmp").replace(target)


def write_report(evidence):
    lines = ["# 500毫秒模型：固定权重的归一化统计审查", "",
             "校准仅使用每折全部8160条源训练输入，关闭所有Dropout，不更新权重。",
             "采集阶段仅BN使用每批统计；按实际样本数及时间长度合并总体均值/方差。",
             "共享重估保持原结构；按步重估仅将仿真步的统计分开，是额外推理诊断。",
             "训练监测为源训练被试block1（1360条）；测试被试240条从未参与统计估计。", "",
             "| 被试 | 种子 | 原监测 | 共享重估监测 | 按步重估监测 | 原测试 | 共享重估测试 | 按步重估测试 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, run in evidence["runs"].items():
        values = [run[mode][split]["acc_percent"] for split in ("monitor", "test") for mode in MODES]
        lines.append(f"| S{run['subject']:02d} | {run['seed']} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    if evidence["runs"]:
        values = [np.mean([run[mode][split]["acc_percent"] for run in evidence["runs"].values()])
                  for split in ("monitor", "test") for mode in MODES]
        lines.append("| 平均 | — | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", f"已完成 {len(evidence['runs'])}/14 个模型；不按测试成绩选择校准方式。",
              "详细指标、CPU与服务器原预测差异、源身份和完整性检查保存在details/evidence.json。",
              "本地CPU诊断，不代表重新训练结果；未覆盖原始服务器权重或结果。"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    details = OUTPUT / "details"
    details.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((EXPERIMENT / "details/experiment.json").read_text(encoding="utf-8"))
    cfg = replace(Config.from_dict(manifest["config"]), data_root=DATA_ROOT, device="cpu",
                  evaluation_device="cpu", cache_root=details / "cache")
    assert cfg.model_variant == "temporal_only" and cfg.windows_ms == (500,)
    for name in ("model.py", "data.py", "experiments/variants.py", "utils/configuration.py"):
        archived = EXPERIMENT / "details/source" / name
        assert file_sha256(archived) == manifest["code_sha256"][name]
        assert archived.read_bytes().replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    plan = {"subjects": list(SUBJECTS), "seeds": list(SEEDS), "modes": list(MODES),
            "batch": BATCH, "calibration_order_seed": ORDER_SEED, "window_ms": 500,
            "calibration": "dropout-off, BN-only batch-stat forward; population moments pooled over all source inputs",
            "core_sha256": {n: file_sha256(ROOT / n) for n in CORE}, "runner_sha256": file_sha256(__file__)}
    evidence, predictions, statistics = {"plan": plan, "runs": {}, "checks": {}}, {}, {}
    if (details / "evidence.json").exists():
        evidence = json.loads((details / "evidence.json").read_text(encoding="utf-8"))
        assert evidence["plan"] == plan, "Do not mix changed audit implementations"
        with np.load(details / "predictions.npz") as archive:
            predictions = {k: archive[k] for k in archive.files}
        statistics = torch.load(details / "statistics.pt", weights_only=True)
    else:
        save(evidence, predictions, statistics)  # 先冻结方案，再读取实验数据。
    prepared = prepare_data(cfg)
    original_data = json.loads((EXPERIMENT / "details/dataset.json").read_text(encoding="utf-8"))
    assert prepared.metadata["sources"] == original_data["sources"]
    # 服务器与本地源码换行符不同会改变原始文件哈希；上面已验证归档源码仅换行不同。
    assert original_data["code"] == manifest["code_sha256"]["data.py"]
    for name in original_data.keys() | prepared.metadata.keys():
        if name not in ("code", "fingerprint"):
            assert prepared.metadata.get(name) == original_data.get(name), name
    evidence["checks"]["raw_data_and_preprocessing_match_server"] = True
    evidence["checks"]["preprocessing_source_differs_only_in_line_endings"] = True
    time_data = torch.from_numpy(prepared.time)
    for subject in SUBJECTS:
        for seed in SEEDS:
            key = f"S{subject:02d}_seed{seed}"
            if key in evidence["runs"]:
                continue
            print(f"START {key}", flush=True)
            started = perf_counter()
            folder = EXPERIMENT / f"window_0500ms/test_S{subject:02d}/seed_{seed}/details"
            checkpoint_hash = file_sha256(folder / "model.pt")
            model = build_model(cfg).eval()
            model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
            parameter_hash = digest_tensors(model.named_parameters())
            source = np.flatnonzero(prepared.ids[:, 0] != subject)
            monitor = source[prepared.ids[source, 1] == 1]
            target = np.flatnonzero(prepared.ids[:, 0] == subject)
            with np.load(folder / "evaluation.npz") as archive:
                assert np.array_equal(prepared.ids[source], archive["training_ids"])
                assert np.array_equal(prepared.ids[target], archive["test_ids"])
                saved_logits = archive["logits"].copy()
                np.testing.assert_array_equal(prepared.time[target[:4]], archive["probe_time"])
            assert len(source) == 8160 and len(monitor) == 1360 and len(target) == 240
            assert not set(source) & set(target)
            row = {"subject": subject, "seed": seed, "checkpoint_sha256": checkpoint_hash,
                   "parameter_sha256": parameter_hash, "calibration_subjects": sorted(set(prepared.ids[source, 0].tolist())),
                   "source_id_sha256": hashlib.sha256(prepared.ids[source].tobytes()).hexdigest()}
            for split, indices in (("monitor", monitor), ("test", target)):
                predictions[f"{key}_{split}_ids"] = prepared.ids[indices]
            for mode in MODES:
                if mode == "shared_reestimated":
                    print(f"CALIBRATE {key}: all 8160 source inputs", flush=True)
                    statistics[key] = estimate(model, time_data[source], cfg)
                net = model if mode == "original" else apply_statistics(model, statistics[key], mode == "per_step_reestimated")
                row[mode] = {}
                for split, indices in (("monitor", monitor), ("test", target)):
                    result, logits = score(net, time_data[indices], prepared.ids[indices, 2], cfg)
                    row[mode][split] = result
                    predictions[f"{key}_{mode}_{split}_logits"] = logits
                    if mode == "original" and split == "test":
                        row["cpu_server_replay"] = {"max_logit_difference": float(np.abs(logits - saved_logits).max()),
                                                   "prediction_disagreements": int((logits.argmax(1) != saved_logits.argmax(1)).sum())}
                assert digest_tensors(net.named_parameters()) == parameter_hash
                print(f"{key} {mode}: monitor={row[mode]['monitor']['acc_percent']:.3f} test={row[mode]['test']['acc_percent']:.3f}", flush=True)
            assert file_sha256(folder / "model.pt") == checkpoint_hash
            row["seconds"] = perf_counter() - started
            row["weights_unchanged_and_evaluation_buffers_frozen"] = True
            evidence["runs"][key] = row
            save(evidence, predictions, statistics)
            write_report(evidence)
    assert plan["core_sha256"] == {n: file_sha256(ROOT / n) for n in CORE}
    evidence["checks"]["production_sources_unchanged"] = True
    evidence["checks"]["all_completed"] = len(evidence["runs"]) == 14
    save(evidence, predictions, statistics)
    write_report(evidence)
    print("AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
