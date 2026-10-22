"""完整频域/时域分支区间计时及三项摘要；细粒度诊断独立保留。"""

import time
import torch
from utils.timing import graph_nodes, SUMMARY_KEYS, SUMMARY_LABELS, timing_rows
from utils import timing as diagnostic


class EpochBranchTimer:
    """CUDA 用事件记录当前流区间；CPU 用 perf_counter。只在 epoch 末同步 CUDA。"""

    def __init__(self, model, device, enabled=True):
        self.device, self.enabled = device, enabled
        self.branches = {name: getattr(model, name) for name in ("spectral", "temporal") if hasattr(model, name)}
        self.forward_handles, self.backward_handles = [], []
        self.nodes, self.starts = {}, {}
        self.input_nodes = {}
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
            self.input_nodes[name] = graph_nodes(inputs)
            self.starts[name] = self.stamp()
        return hook

    def forward_end(self, name):
        def hook(module, inputs, outputs):
            self.intervals[name].append((self.starts.pop(name), self.stamp()))
            self.nodes[name] = graph_nodes(outputs, self.input_nodes.pop(name))
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


def timing_log(row):
    return diagnostic.timing_log(row, SUMMARY_KEYS, SUMMARY_LABELS)


def timing_totals(runs, epochs):
    return diagnostic.timing_totals(runs, epochs, SUMMARY_KEYS)


def timing_markdown(totals):
    return diagnostic.timing_markdown(totals, SUMMARY_KEYS, SUMMARY_LABELS)
