"""时域残差模块卷积/LIF 的前向、反向诊断计时；不改模型计算。"""

import math
import time
from collections import defaultdict

import torch
from torch import nn
from spikingjelly.activation_based import neuron

from utils.operator_profile import OperatorSample, sample_log


PARTS = ("temporal2_conv_forward", "temporal2_conv_backward", "temporal2_lif_forward", "temporal2_lif_backward")
TIME_KEYS = tuple(f"{name}_seconds" for name in PARTS)
TIME_LABELS = ("T2 Conv Forward", "T2 Conv Backward", "T2 LIF Forward", "T2 LIF Backward")
HISTORY_KEYS = ("epoch", "loss", "train_acc_percent", "train_trials", *TIME_KEYS,
                *(key + "_per_trial" for key in TIME_KEYS), "monitor_loss", "monitor_acc_percent")


def graph_nodes(tensors, stop=()):
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    pending = [tensor.grad_fn for tensor in tensors if tensor.grad_fn is not None]
    nodes = set()
    while pending:
        node = pending.pop()
        if node in nodes or node in stop:
            continue
        nodes.add(node)
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
    return nodes


class BranchTimer:
    """只诊断 temporal.blocks 的 Conv1d/LIF；CUDA 事件在 epoch 末统一解析。"""

    def __init__(self, model, device, enabled=True, profile_batch=0):
        self.device, self.enabled = device, enabled
        blocks = getattr(getattr(model, "temporal", None), "blocks", None)
        self.modules, self.names, self.layer_metadata = {}, {}, {}
        if blocks is not None:
            for name, module in blocks.named_modules():
                if isinstance(module, nn.Conv1d):
                    self.modules[module] = "conv"
                elif isinstance(module, neuron.LIFNode):
                    self.modules[module] = "lif"
                else:
                    continue
                self.names[module] = name
                self.layer_metadata[name] = {"kind": self.modules[module], "module": type(module).__name__}
                if isinstance(module, nn.Conv1d):
                    self.layer_metadata[name].update(in_channels=module.in_channels, out_channels=module.out_channels,
                                                    kernel=list(module.kernel_size), dilation=list(module.dilation),
                                                    padding=list(module.padding), stride=list(module.stride),
                                                    groups=module.groups, bias=module.bias is not None)
                else:
                    self.layer_metadata[name].update(tau=module.tau, backend=module.backend,
                                                    step_mode=module.step_mode, detach_reset=module.detach_reset)
        self.forward_handles, self.backward_handles = [], []
        self.owners, self.starts = {}, {}
        self.intervals = {(kind, phase): [] for kind in ("conv", "lif") for phase in ("forward", "backward")}
        self.calls = defaultdict(int)
        self.batch, self.batch_size = 1, 0
        self.profile_batch, self.sample = profile_batch, None
        self.profile_active = False
        self.resolved = None

    def begin_batch(self, index, batch_size):
        self.batch, self.batch_size = index, batch_size
        self.calls.clear()
        if self.enabled and self.modules and index == self.profile_batch:
            self.sample = OperatorSample(self.device, index)
            self.sample.start()
            self.profile_active = self.sample.profiler is not None

    def end_batch(self):
        if self.profile_active:
            self.sample.stop()
            self.profile_active = False

    def start_interval(self, owner, phase):
        kind, layer, step, node = owner
        marker = self.sample.marker(layer, phase, step, node) if self.profile_active else None
        return self.stamp(), time.perf_counter(), marker

    def end_interval(self, owner, phase, started):
        end, host_end = self.stamp(), time.perf_counter()
        start, host_start, marker = started
        if marker is not None:
            marker.__exit__(None, None, None)
        kind, layer, step, node = owner
        self.intervals[kind, phase].append((start, end, {
            "layer": layer, "step": step, "phase": phase, "node": node,
            "batch_size": self.batch_size, "batch_group": "first" if self.batch == 1 else "later",
            "profiled": self.profile_active, "host_seconds": host_end - host_start,
        }))

    def stamp(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def __enter__(self):
        if self.enabled:
            for module, kind in self.modules.items():
                self.forward_handles.append(module.register_forward_pre_hook(self.forward_start))
                self.forward_handles.append(module.register_forward_hook(self.forward_end))
        return self

    def forward_start(self, module, inputs):
        # 明确排除输入上游的 BN、残差、其他分支，计时不包含归属图遍历。
        self.owners.update({node: None for node in graph_nodes(inputs, self.owners)})
        self.calls[module] += 1
        name = self.names[module]
        if "input_shape" not in self.layer_metadata[name]:
            value = inputs[0]
            self.layer_metadata[name].update(input_shape=list(value.shape), input_stride=list(value.stride()),
                                             dtype=str(value.dtype), contiguous=value.is_contiguous())
        owner = (self.modules[module], name, self.calls[module], "forward")
        self.starts[module] = owner, self.start_interval(owner, "forward")

    def forward_end(self, module, inputs, output):
        owner, started = self.starts.pop(module)
        self.end_interval(owner, "forward", started)
        kind = self.modules[module]
        outputs = (output,)
        # LIF 重置后的膜电位要到后续仿真步才参与 loss；也必须标注其反向归属。
        if kind == "lif" and isinstance(module.v, torch.Tensor):
            outputs += (module.v,)
        for node in graph_nodes(outputs, self.owners):
            node_name = node.name()
            # 参数 AccumulateGrad 跨仿真步共享，不能硬分给首次遇到它的 step。
            step = 0 if node_name.endswith("AccumulateGrad") else owner[2]
            self.owners[node] = kind, owner[1], step, node_name

    def attach_backward(self, node, owner):
        started = []

        def before(grad_outputs):
            started.append(self.start_interval(owner, "backward"))

        def after(grad_inputs, grad_outputs):
            self.end_interval(owner, "backward", started.pop())

        # 每个归属节点独立包围，避免把相邻 BN/其他节点或节点间调度空隙算入。
        self.backward_handles.append(node.register_prehook(before))
        self.backward_handles.append(node.register_hook(after))

    def prepare_backward(self, loss):
        if self.enabled:
            for node in graph_nodes(loss):
                kind = self.owners.get(node)
                if kind is not None:
                    self.attach_backward(node, kind)

    def end_backward(self):
        for handle in self.backward_handles:
            handle.remove()
        self.backward_handles.clear()
        self.owners.clear()

    def totals(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result = {}
        self.resolved = []
        for (kind, phase), pairs in self.intervals.items():
            if kind not in self.modules.values():
                seconds = 0.0
            elif not self.enabled:
                seconds = None
            else:
                seconds = 0.0
                for start, end, record in pairs:
                    value = start.elapsed_time(end) / 1000 if self.device.type == "cuda" else end - start
                    seconds += value
                    self.resolved.append({**record, "seconds": value})
            result[f"temporal2_{kind}_{phase}_seconds"] = seconds
        return result

    def details(self):
        if not self.enabled or not self.modules:
            return None
        if self.resolved is None:
            raise RuntimeError("Read totals before detailed timing")
        grouped = {}
        dimensions = ("layer", "step", "phase", "node", "batch_size", "batch_group", "profiled")
        for record in self.resolved:
            key = tuple(record[name] for name in dimensions)
            row = grouped.setdefault(key, {name: record[name] for name in dimensions} | {
                "calls": 0, "seconds": 0.0, "host_seconds": 0.0, "max_seconds": 0.0})
            row["calls"] += 1
            row["seconds"] += record["seconds"]
            row["host_seconds"] += record["host_seconds"]
            row["max_seconds"] = max(row["max_seconds"], record["seconds"])
        return {"schema": 1, "clock": "cuda_event" if self.device.type == "cuda" else "cpu_wall",
                "layers": self.layer_metadata, "records": list(grouped.values()),
                "sample": self.sample.result if self.sample is not None else None}

    def __exit__(self, *exc):
        for _, (_, _, marker) in self.starts.values():
            if marker is not None:
                marker.__exit__(*exc)
        self.end_batch()
        self.end_backward()
        for handle in self.forward_handles:
            handle.remove()
        self.forward_handles.clear()
        self.starts.clear()


def normalize_history(history, training_trials=None):
    """升级旧历史，保留可恢复的计时；批均摊耗时不冒充单 trial 独立延迟。"""
    result = []
    for original in history:
        row = {key: original.get(key) for key in HISTORY_KEYS}
        if original.get("_t2_diagnostics") is not None:
            row["_t2_diagnostics"] = original["_t2_diagnostics"]
        row["train_trials"] = original.get("train_trials") or training_trials
        for key in TIME_KEYS:
            value, count = row[key], row["train_trials"]
            row[key + "_per_trial"] = value / count if value is not None and count else None
        result.append(row)
    return result


def timing_log(row):
    fields = []
    for key, label in zip(TIME_KEYS, TIME_LABELS):
        value = row.get(key)
        seconds = f"{value:.3f}s" if value is not None else "NA"
        fields.append(f"{label}={seconds}")
    return " | ".join(fields)


def detailed_timing_log(details):
    if not details:
        return []
    lines = [f"T2 DETAIL clock={details['clock']}; F=layer calls, B=autograd nodes; "
             "Host=submission interval (overlaps CUDA); Bshared=parameter gradient accumulation."]
    for layer, metadata in details["layers"].items():
        records = [row for row in details["records"] if row["layer"] == layer]
        def total(phase, field):
            return sum(row[field] for row in records if row["phase"] == phase)
        steps = sorted({row["step"] for row in records if row["step"]})
        def step_times(phase):
            return ",".join(f"{sum(row['seconds'] for row in records if row['phase'] == phase and row['step'] == step):.6f}"
                            for step in steps)
        shape = "x".join(map(str, metadata["input_shape"]))
        settings = (f"k={metadata['kernel'][0]} d={metadata['dilation'][0]} bias={metadata['bias']}"
                    if metadata["kind"] == "conv" else f"tau={metadata['tau']} backend={metadata['backend']}")
        maximum = max((row["max_seconds"] for row in records if row["phase"] == "forward"), default=0)
        shared = sum(row["seconds"] for row in records if row["phase"] == "backward" and row["step"] == 0)
        lines.append(f"T2 LAYER {layer} {metadata['kind']} {settings} shape={shape} "
                     f"F={total('forward', 'seconds'):.6f}s B={total('backward', 'seconds'):.6f}s "
                     f"FHost={total('forward', 'host_seconds'):.6f}s BHost={total('backward', 'host_seconds'):.6f}s "
                     f"Fn={total('forward', 'calls')} Bn={total('backward', 'calls')} Fmax={maximum:.6f}s "
                     f"Fstep=[{step_times('forward')}]s Bstep=[{step_times('backward')}]s Bshared={shared:.6f}s")
    lines.extend(sample_log(details.get("sample")))
    return lines


def timing_totals(runs, epochs):
    """runs 为 (variant, history) 列表；缺失计时不冒充 0 或完整总时长。"""
    result = {}
    for key in TIME_KEYS:
        values, expected, trials = [], 0, []
        for variant, history in runs:
            absent = variant == "spectral_only"
            if absent:
                continue
            expected += epochs
            for row in history:
                value = row.get(key)
                if value is not None and math.isfinite(value) and value >= 0:
                    values.append(value)
                    count = row.get("train_trials")
                    if count is not None and math.isfinite(count) and count > 0:
                        trials.append(count)
        observed = sum(values) if values or expected == 0 else None
        result[key] = {"recorded_seconds": observed,
                       "total_seconds": observed if len(values) == expected else None,
                       "recorded_epochs": len(values), "expected_epochs": expected,
                       "recorded_trials": sum(trials),
                       "seconds_per_trial": (observed / sum(trials) if trials and len(trials) == len(values)
                                             else 0.0 if expected == 0 else None)}
    return result


def timing_markdown(totals):
    lines = ["", "| 训练时间累计 | 总时长（秒 / 小时） | 已记录部分平均（秒/trial） | 有计时的 epoch / 应记录 epoch |",
             "| --- | ---: | ---: | ---: |"]
    for key, label in zip(TIME_KEYS, TIME_LABELS):
        item = totals[key]
        value = item["total_seconds"]
        if item["expected_epochs"] == 0:
            display = "未启用该分支：0"
        elif value is not None:
            display = f"{value:.3f} / {value / 3600:.4f}"
        elif item["recorded_seconds"] is not None:
            display = f"完整总时长未知；已记录 {item['recorded_seconds']:.3f} 秒"
        else:
            display = "未记录"
        trial = item["seconds_per_trial"]
        trial_text = f"{trial:.6f}" if trial is not None else "未记录"
        lines.append(f"| {label} | {display} | {trial_text} | {item['recorded_epochs']} / {item['expected_epochs']} |")
    lines += ["", "仅统计第二模块 temporal.blocks：所有膨胀/逐点卷积及所有 LIF，跨残差块和仿真步累计。",
              "前向按层调用包围；反向按归属 autograd 节点的执行前后包围，含 LIF 跨步膜电位与重置梯度。",
              "BN、Dropout、残差相加、其他模块、loss 和优化器不计入这四项。四项之和不代表整模块或整轮时间。",
              "CUDA 使用同一流的 Event，epoch 末同步后读数；区间可能包含提交/调度空闲，不等于纯 kernel 时间。",
              "秒/trial 是对应已记录样本处理次数上的均摊值；多任务按次数加权，不是单样本延迟或双卡墙钟时间。",
              "旧版未记录的项保持缺失，v8 已有四项保留；中断重做但未提交的 epoch 不计入。",
              "采样 epoch 包含 profiler 额外开销；逐层和底层算子明细见任务 details/t2_diagnostics.json。"]
    return lines


def timing_rows(totals, window, **identity):
    return [dict(window_ms=window, metric=key, scope="training_time_total", **identity, **totals[key])
            for key in TIME_KEYS]
