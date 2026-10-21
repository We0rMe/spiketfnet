"""分支前向/反向区间计时与累计；不改计算图、梯度或优化步骤。"""

import math
import time

import torch


PARTS = ("spectral", "temporal1", "temporal2", "temporal3")
TIME_KEYS = tuple(f"{name}_train_seconds" for name in PARTS) + ("train_seconds",)
TIME_LABELS = ("Spectral Times", "Temporal Times1", "Temporal Times2", "Temporal Times3", "Total Time")
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
    """CUDA 用事件记录当前流区间；CPU 用 perf_counter。只在 epoch 末同步 CUDA。"""

    def __init__(self, model, device, enabled=True):
        self.device, self.enabled = device, enabled
        self.model = model
        self.parts = [name for name in PARTS if hasattr(model, "spectral" if name == "spectral" else "temporal")]
        self.forward_handles, self.backward_handles = [], []
        self.owners = {}
        self.intervals = {name: [] for name in self.parts}
        self.forward_active, self.forward_stamp = None, None
        self.active, self.backward_start = None, None

    def stamp(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def __enter__(self):
        if self.enabled:
            if hasattr(self.model, "spectral"):
                self.attach(self.model.spectral, "spectral", root=True, finish=True)
            if hasattr(self.model, "temporal"):
                temporal = self.model.temporal
                self.attach(temporal, "temporal1", root=True, finish=True, output_part="temporal3")
                self.attach(temporal.current, "temporal1")
                self.attach(temporal.encoder, "temporal1", capture=False)
                self.attach(temporal.dropout, "temporal1")
                self.attach(temporal.blocks, "temporal2")
                self.attach(temporal.attention, "temporal3")
                self.attach(temporal.embedding, "temporal3")
                self.attach(temporal.classifier, "temporal3")
        return self

    def switch_forward(self, name):
        if name == self.forward_active:
            return
        now = self.stamp()
        if self.forward_active is not None:
            self.intervals[self.forward_active].append((self.forward_stamp, now))
        self.forward_active, self.forward_stamp = name, now

    def attach(self, module, name, root=False, finish=False, output_part=None, capture=True):
        def start(module, inputs):
            if root:
                self.owners.update({node: None for node in graph_nodes(inputs, self.owners)})
            self.switch_forward(name)

        def end(module, inputs, outputs):
            # 在已标注的输入图处停止，避免把上游模块的反向重复归入下游。
            # 图遍历本身不算进模块前向时间；池化/跨步平均由外围连续区间覆盖。
            self.switch_forward(None)
            self.owners.update({node: output_part or name for node in graph_nodes(outputs, self.owners)})
            if not finish:
                self.switch_forward(name)

        self.forward_handles.append(module.register_forward_pre_hook(start))
        if capture:
            self.forward_handles.append(module.register_forward_hook(end))

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
        # 每个 Node 都标注归属；分支梯度若交错执行，切换区间而不重复归入两个分支。
        # Module full-backward-hook 在原始输入不需梯度时会过早触发，不能作为分支反向终点。
        for node in graph_nodes((loss,)):
            self.backward_handles.append(node.register_prehook(self.backward_hook(self.owners.get(node))))

    def end_backward(self):
        if self.enabled:
            self.switch_backward(None)
        for handle in self.backward_handles:
            handle.remove()
        self.backward_handles.clear()
        self.owners.clear()

    def totals(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result = {}
        for name in PARTS:
            if name not in self.parts:
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
        value, count = row.get(key), row.get("train_trials")
        seconds = f"{value:.3f}s" if value is not None else "NA"
        trial = f"{1000 * value / count:.3f}ms/trial" if value is not None and count else "NA/trial"
        fields.append(f"{label}={seconds} ({trial})")
    return " | ".join(fields)


def timing_totals(runs, epochs):
    """runs 为 (variant, history) 列表；缺失计时不冒充 0 或完整总时长。"""
    result = {}
    for key in TIME_KEYS:
        values, expected, trials = [], 0, []
        for variant, history in runs:
            absent = ((key == "spectral_train_seconds" and variant == "temporal_only") or
                      (key.startswith("temporal") and variant == "spectral_only"))
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
    lines += ["", "Temporal Times1：电流映射与 LIF 编码；Times2：全部残差块；Times3：注意力、率读出与输出头。",
              "各模块累计所有仿真步的前向+反向；融合、loss、Adam、数据搬运等只计入 Total Time。",
              "Total Time 为训练阶段，不含监测、测试、保存和绘图。模块耗时之和无需等于 Total Time。",
              "秒/trial = 已记录耗时 / 对应实际处理 trial 次数（重复 epoch 重复计数），是 batch 均摊值，不是单样本延迟。",
              "多任务汇总按 trial 次数加权，不是各任务平均值的简单平均；累计时长不是双卡墙钟时间。",
              "旧模块耗时未测量，不能从旧时域总耗时拆分；中断重做但未提交的 epoch 不计入。"]
    return lines


def timing_rows(totals, window, **identity):
    return [dict(window_ms=window, metric=key, scope="training_time_total", **identity, **totals[key])
            for key in TIME_KEYS]
