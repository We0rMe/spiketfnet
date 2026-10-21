"""时域残差模块卷积/LIF 的前向、反向诊断计时；不改模型计算。"""

import math
import time

import torch
from torch import nn
from spikingjelly.activation_based import neuron


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

    def __init__(self, model, device, enabled=True):
        self.device, self.enabled = device, enabled
        blocks = getattr(getattr(model, "temporal", None), "blocks", None)
        self.modules = {}
        if blocks is not None:
            for module in blocks.modules():
                if isinstance(module, nn.Conv1d):
                    self.modules[module] = "conv"
                elif isinstance(module, neuron.LIFNode):
                    self.modules[module] = "lif"
        self.forward_handles, self.backward_handles = [], []
        self.owners, self.starts = {}, {}
        self.intervals = {(kind, phase): [] for kind in ("conv", "lif") for phase in ("forward", "backward")}

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
        self.starts[module] = self.stamp()

    def forward_end(self, module, inputs, output):
        end = self.stamp()
        kind = self.modules[module]
        self.intervals[kind, "forward"].append((self.starts.pop(module), end))
        outputs = (output,)
        # LIF 重置后的膜电位要到后续仿真步才参与 loss；也必须标注其反向归属。
        if kind == "lif" and isinstance(module.v, torch.Tensor):
            outputs += (module.v,)
        self.owners.update({node: kind for node in graph_nodes(outputs, self.owners)})

    def attach_backward(self, node, kind):
        started = []

        def before(grad_outputs):
            started.append(self.stamp())

        def after(grad_inputs, grad_outputs):
            self.intervals[kind, "backward"].append((started.pop(), self.stamp()))

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
        for (kind, phase), pairs in self.intervals.items():
            if kind not in self.modules.values():
                seconds = 0.0
            elif not self.enabled:
                seconds = None
            else:
                seconds = sum(start.elapsed_time(end) / 1000 if self.device.type == "cuda" else end - start
                              for start, end in pairs)
            result[f"temporal2_{kind}_{phase}_seconds"] = seconds
        return result

    def __exit__(self, *exc):
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
              "旧 epoch 未记录这四项，保持缺失；中断重做但未提交的 epoch 不计入。"]
    return lines


def timing_rows(totals, window, **identity):
    return [dict(window_ms=window, metric=key, scope="training_time_total", **identity, **totals[key])
            for key in TIME_KEYS]
