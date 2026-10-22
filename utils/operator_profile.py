"""单 batch 的底层算子采样；只保存 T2 归属的汇总，不导出大体积 trace。"""

import os
from collections import defaultdict

import torch


MARKER = "T2::"


def model_parameters(model):
    def count(module):
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"trainable": count(model), "branches": {name: count(module) for name, module in model.named_children()},
            "submodules": {f"{branch}.{name}": count(child) for branch, module in model.named_children()
                           for name, child in module.named_children()}}


def runtime_settings(device):
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_threads": torch.get_num_threads(),
        "profiler_kineto_available": torch.profiler.kineto_available(),
        "profiler_supported_activities": sorted(str(value) for value in torch.profiler.supported_activities()),
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUBLAS_WORKSPACE_CONFIG",
            "TORCH_CUDNN_V8_API_DISABLED", "NVIDIA_TF32_OVERRIDE")},
    }


def summarize_events(events, device_type):
    """按最近的 T2 标记归属；CPU self 与直接关联 kernel 各累计一次。"""
    events = list(events)
    operators, kernels = {}, {}
    owners = {}

    def owner(event):
        if event is None:
            return None
        if event.id not in owners:
            if event.name.startswith(MARKER):
                _, layer, phase, *_ = event.name.split("::")
                owners[event.id] = layer, phase
            else:
                owners[event.id] = owner(event.cpu_parent)
        return owners[event.id]

    for event in events:
        if event.device_type != torch.autograd.DeviceType.CPU or event.name.startswith(MARKER):
            continue
        scope = owner(event)
        if scope is None:
            continue
        key = (*scope, event.name)
        row = operators.setdefault(key, {"layer": scope[0], "phase": scope[1], "operator": event.name,
                                         "calls": 0, "self_cpu_seconds": 0.0, "kernel_seconds": 0.0})
        row["calls"] += 1
        row["self_cpu_seconds"] += event.self_cpu_time_total / 1e6
        # kernels 属于当前 CPU 事件，不能再累计父事件的 inclusive CUDA time。
        for kernel in event.kernels:
            seconds = kernel.duration / 1e6
            row["kernel_seconds"] += seconds
            key = (*scope, kernel.name)
            item = kernels.setdefault(key, {"layer": scope[0], "phase": scope[1], "kernel": kernel.name,
                                            "calls": 0, "kernel_seconds": 0.0})
            item["calls"] += 1
            item["kernel_seconds"] += seconds
    observed = bool(kernels)
    coverage = {
        "cpu_events": sum(event.device_type == torch.autograd.DeviceType.CPU for event in events),
        "cuda_events": sum(event.device_type == torch.autograd.DeviceType.CUDA for event in events),
        "linked_kernels_all_scopes": sum(len(event.kernels) for event in events
                                         if event.device_type == torch.autograd.DeviceType.CPU),
        "linked_kernels_t2": sum(row["calls"] for row in kernels.values()),
    }
    if not observed:
        for row in operators.values():
            row["kernel_seconds"] = None
    return {"operators": list(operators.values()), "kernels": list(kernels.values()),
            "coverage": coverage,
            "cuda_kernels_observed": observed,
            "note": ("Self CPU and linked CUDA kernels; parent inclusive times are not added."
                     if observed else "CUDA kernel timing unavailable; CPU operator times only."
                     if device_type == "cuda" else "CPU smoke: no CUDA kernel timing.")}


class OperatorSample:
    def __init__(self, device, batch):
        self.device, self.batch = device, batch
        self.profiler = None
        self.result = None
        self.activities = []

    def start(self):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda" and torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.activities = [str(value) for value in activities]
        try:
            self.profiler = torch.profiler.profile(activities=activities, record_shapes=False,
                                                  profile_memory=False, with_stack=False)
            self.profiler.start()
        except Exception as exc:
            self.profiler = None
            self.result = {"batch": self.batch, "error": f"Profiler start failed: {type(exc).__name__}: {exc}"}

    def marker(self, layer, phase, step, node):
        if self.profiler is None:
            return None
        context = torch.profiler.record_function(f"{MARKER}{layer}::{phase}::step={step}::{node}")
        context.__enter__()
        return context

    def stop(self):
        if self.profiler is None:
            return
        try:
            self.profiler.stop()
            self.result = {"batch": self.batch, "requested_activities": self.activities,
                           **summarize_events(self.profiler.events(), self.device.type)}
        except Exception as exc:
            self.result = {"batch": self.batch, "error": f"Profiler collection failed: {type(exc).__name__}: {exc}"}
        finally:
            self.profiler = None


def sample_log(sample):
    if not sample:
        return []
    lines = [f"T2 SAMPLE batch={sample['batch']} (one batch only; profiler overhead included)"]
    if "error" in sample:
        return lines + [sample["error"]]
    lines.append(f"T2 SAMPLE {sample['note']}")
    if "coverage" in sample:
        lines.append(f"T2 PROFILER activities={sample.get('requested_activities')} coverage={sample['coverage']}")
        if not sample["cuda_kernels_observed"] and "CUDA" in str(sample.get("requested_activities")):
            lines.append("T2 PROFILER CUDA requested but no T2 kernels linked; inspect CUDA event/"
                         "global link counts and profiler warnings (CUPTI/runtime permissions). "
                         "Layer CUDA Event timings remain separate measurements.")
    grouped = defaultdict(list)
    for row in sample["operators"]:
        if row["operator"].startswith(("aten::", "prim::", "cuda")):
            grouped[row["layer"], row["phase"]].append(row)
    for (layer, phase), rows in sorted(grouped.items()):
        # CPU self 能揭示逐通道 copy_/select 等调度；另列 kernel 排名避免只看 CPU。
        by_cpu = sorted(rows, key=lambda row: row["self_cpu_seconds"], reverse=True)[:4]
        by_gpu = sorted((row for row in rows if row["kernel_seconds"]),
                        key=lambda row: row["kernel_seconds"], reverse=True)[:4]
        diagnostic_ops = [row for row in rows if row["operator"] in (
            "aten::fill_", "aten::copy_", "aten::select", "aten::item", "aten::_local_scalar_dense")]
        chosen = {row["operator"]: row for row in by_cpu + by_gpu + diagnostic_ops if row["calls"]}
        text = "; ".join(f"{row['operator']} n={row['calls']} CPUself={row['self_cpu_seconds']:.6f}s "
                         f"kernels={row['kernel_seconds']:.6f}s" if row["kernel_seconds"] is not None else
                         f"{row['operator']} n={row['calls']} CPUself={row['self_cpu_seconds']:.6f}s kernels=NA"
                         for row in chosen.values())
        lines.append(f"T2 SAMPLE {layer} {phase}: {text}")
        backends = [row for row in rows if any(token in row["operator"] for token in (
            "cudnn", "slow_conv", "mkldnn", "thnn"))]
        if backends:
            lines.append(f"T2 BACKEND {layer} {phase}: " + ", ".join(
                f"{row['operator']} n={row['calls']}" for row in backends))
        layer_kernels = [row for row in sample["kernels"] if row["layer"] == layer and row["phase"] == phase]
        for row in sorted(layer_kernels, key=lambda row: row["kernel_seconds"], reverse=True)[:4]:
            lines.append(f"T2 KERNEL {layer} {phase} n={row['calls']} "
                         f"time={row['kernel_seconds']:.6f}s {row['kernel']}")
    return lines
