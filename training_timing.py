"""分支前向/反向区间计时与累计；不改计算图、梯度或优化步骤。"""

import math
import time

import torch


TIME_KEYS = ("spectral_train_seconds", "temporal_train_seconds", "train_seconds")
TIME_LABELS = ("频域分支前向+反向", "时域分支前向+反向", "整体训练阶段")


def graph_nodes(tensors):
    pending = [tensor.grad_fn for tensor in tensors if tensor.grad_fn is not None]
    nodes = set()
    while pending:
        node = pending.pop()
        if node in nodes:
            continue
        nodes.add(node)
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
    return nodes


class BranchTimer:
    """CUDA 用事件记录当前流区间；CPU 用 perf_counter。只在 epoch 末同步 CUDA。"""

    def __init__(self, model, device, enabled=True):
        self.device, self.enabled = device, enabled
        self.branches = {name: getattr(model, name) for name in ("spectral", "temporal") if hasattr(model, name)}
        self.forward_handles, self.backward_handles = [], []
        self.nodes, self.starts = {}, {}
        self.intervals = {name: [] for name in self.branches}
        self.active, self.backward_start = None, None

    def stamp(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def __enter__(self):
        if self.enabled:
            for name, module in self.branches.items():
                self.forward_handles.append(module.register_forward_pre_hook(self.forward_start(name)))
                self.forward_handles.append(module.register_forward_hook(self.forward_end(name)))
        return self

    def forward_start(self, name):
        def hook(module, inputs):
            self.starts[name] = self.stamp()
        return hook

    def forward_end(self, name):
        def hook(module, inputs, outputs):
            self.intervals[name].append((self.starts.pop(name), self.stamp()))
            self.nodes[name] = graph_nodes(outputs)
        return hook

    def switch_backward(self, name):
        if name == self.active:
            return
        now = self.stamp()
        if self.active is not None:
            self.intervals[self.active].append((self.backward_start, now))
        self.active, self.backward_start = name, now

    def backward_hook(self, name):
        def hook(grad_outputs):
            self.switch_backward(name)
        return hook

    def prepare_backward(self, loss):
        if not self.enabled:
            return
        owners = {}
        for name, nodes in self.nodes.items():
            for node in nodes:
                owners[node] = name if node not in owners else None
        # 每个 Node 都标注归属；分支梯度若交错执行，切换区间而不重复归入两个分支。
        # Module full-backward-hook 在原始输入不需梯度时会过早触发，不能作为分支反向终点。
        for node in graph_nodes((loss,)):
            self.backward_handles.append(node.register_prehook(self.backward_hook(owners.get(node))))

    def end_backward(self):
        if self.enabled:
            self.switch_backward(None)
        for handle in self.backward_handles:
            handle.remove()
        self.backward_handles.clear()
        self.nodes.clear()

    def totals(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result = {}
        for name in ("spectral", "temporal"):
            if name not in self.branches:
                seconds = 0.0
            elif not self.enabled:
                seconds = None
            else:
                pairs = self.intervals[name]
                seconds = sum(start.elapsed_time(end) / 1000 if self.device.type == "cuda" else end - start
                              for start, end in pairs)
            result[f"{name}_train_seconds"] = seconds
        return result

    def __exit__(self, *exc):
        self.end_backward()
        for handle in self.forward_handles:
            handle.remove()
        self.forward_handles.clear()


def timing_totals(runs, epochs):
    """runs 为 (variant, history) 列表；缺失计时不冒充 0 或完整总时长。"""
    result = {}
    for key in TIME_KEYS:
        values, expected = [], 0
        for variant, history in runs:
            absent = ((key == "spectral_train_seconds" and variant == "temporal_only") or
                      (key == "temporal_train_seconds" and variant == "spectral_only"))
            if absent:
                continue
            expected += epochs
            for row in history:
                value = row.get(key)
                if value is not None and math.isfinite(value) and value >= 0:
                    values.append(value)
        observed = sum(values) if values or expected == 0 else None
        result[key] = {"recorded_seconds": observed,
                       "total_seconds": observed if len(values) == expected else None,
                       "recorded_epochs": len(values), "expected_epochs": expected}
    return result


def timing_markdown(totals):
    lines = ["", "| 训练时间累计 | 总时长（秒 / 小时） | 有计时的 epoch / 应记录 epoch |",
             "| --- | ---: | ---: |"]
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
        lines.append(f"| {label} | {display} | {item['recorded_epochs']} / {item['expected_epochs']} |")
    lines += ["", "分支仅计自身模块前向+反向；融合、loss、Adam、数据搬运等计入整体训练阶段。",
              "不含监测、测试、保存和绘图。多任务时间相加，不是双卡运行经过的墙钟时间；中断重做但未提交的 epoch 不计入。"]
    return lines


def timing_rows(totals, window, **identity):
    return [dict(window_ms=window, metric=key, scope="training_time_total", **identity, **totals[key])
            for key in TIME_KEYS]
