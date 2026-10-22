"""独立验证/测速 native 与 split_bias；不读取或修改正式训练断点。"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from config import CONFIG
from evaluate import resolve_device
from model import SplitBiasConv1d
from train import seed_everything
from utils.operator_profile import OperatorSample, runtime_settings


DEVICE = "auto"  # 服务器可指定空闲卡 cuda:0/cuda:1；不要与正式任务争用 GPU
WINDOWS_MS = (500, 1000)
BATCH_SIZES = (128,)  # CPU 冒烟自动使用 batch=2；GPU 保留此值
REPEATS = 10
WARMUP = 2
PROFILE_OPERATORS = True  # 测速结束后单独采样，不混入速度统计
OUTPUT = ROOT / "smoke_tests/results/conv_benchmark/report.json"


def comparison(native, split, x, gradient):
    values = []
    for layer in (native, split):
        sample = x.detach().clone().requires_grad_()
        layer.zero_grad(set_to_none=True)
        output = layer(sample)
        output.backward(gradient)
        values.append((output.detach(), sample.grad, layer.weight.grad, layer.bias.grad))
    results = {}
    for name, a, b in zip(("output", "input_gradient", "weight_gradient", "bias_gradient"), *values):
        results[name] = {"max_abs_error": (a - b).abs().max().item(),
                         "passed": torch.allclose(a, b, rtol=1e-4, atol=1e-5)}
    return results


def measure(layer, x, gradient, device, repeats):
    def stamp():
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def elapsed(a, b):
        return a.elapsed_time(b) if device.type == "cuda" else (b - a) * 1000

    def clear():
        layer.zero_grad(set_to_none=True)
        x.grad = None

    for _ in range(WARMUP):
        for _ in range(CONFIG.simulation_steps):
            clear()
            layer(x).backward(gradient)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    intervals = []
    wall = time.perf_counter()
    for _ in range(repeats):
        for _ in range(CONFIG.simulation_steps):
            clear()
            a = stamp()
            y = layer(x)
            b = stamp()
            c = stamp()
            y.backward(gradient)
            d = stamp()
            intervals.append((a, b, c, d))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall) * 1000 / repeats
    return {"forward_ms_per_batch_Ts": sum(elapsed(a, b) for a, b, _, _ in intervals) / repeats,
            "backward_ms_per_batch_Ts": sum(elapsed(c, d) for _, _, c, d in intervals) / repeats,
            "wall_ms_per_batch_Ts": wall_ms}


def sample_operators(layer, x, gradient, device):
    sample = OperatorSample(device, 1)
    sample.start()
    try:
        for step in range(1, CONFIG.simulation_steps + 1):
            layer.zero_grad(set_to_none=True)
            x.grad = None
            marker = sample.marker("conv", "forward", step, "forward")
            try:
                output = layer(x)
            finally:
                if marker is not None:
                    marker.__exit__(None, None, None)
            marker = sample.marker("conv", "backward", step, "backward")
            try:
                output.backward(gradient)
            finally:
                if marker is not None:
                    marker.__exit__(None, None, None)
    finally:
        sample.stop()
    return sample.result


def main():
    device = resolve_device(DEVICE)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(CONFIG.cpu_threads)
    seed_everything(CONFIG.seed, CONFIG)
    batches = BATCH_SIZES if device.type == "cuda" else (2,)
    repeats = REPEATS if device.type == "cuda" else 3
    report = {"runtime": runtime_settings(device), "simulation_steps": CONFIG.simulation_steps,
              "repeats": repeats, "warmup_batches": WARMUP, "cases": [],
              "note": "Synthetic spike inputs; convolution only; no claim about full-model accuracy or speed. "
                      "CPU smoke is not a server GPU benchmark. Profiler runs separately after timings."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        for window in WINDOWS_MS:
            for batch in batches:
                for dilation in sorted(set(d for d in CONFIG.dilations if d > 1)):
                    width = CONFIG.temporal_width
                    args = dict(in_channels=width, out_channels=width, kernel_size=CONFIG.residual_kernel,
                                padding=dilation * (CONFIG.residual_kernel // 2), dilation=dilation)
                    native, split = nn.Conv1d(**args).to(device), SplitBiasConv1d(**args).to(device)
                    split.load_state_dict(native.state_dict())
                    x = (torch.rand(batch, width, CONFIG.samples(window), device=device) < 0.2).float().requires_grad_()
                    gradient = torch.randn(batch, width, CONFIG.samples(window), device=device) / x.numel()
                    row = {"window_ms": window, "batch": batch, "dilation": dilation,
                           "input_shape": list(x.shape), "dtype": str(x.dtype),
                           "kernel_size": CONFIG.residual_kernel, "bias": True,
                           "checks": comparison(native, split, x, gradient)}
                    report["cases"].append(row)
                    if not all(result["passed"] for result in row["checks"].values()):
                        raise AssertionError("Output/gradient comparison failed; see report before using split_bias")
                    for mode, layer in (("native", native), ("split_bias", split)):
                        row[mode] = measure(layer, x, gradient, device, repeats)
                    row["forward_speedup"] = row["native"]["forward_ms_per_batch_Ts"] / row["split_bias"]["forward_ms_per_batch_Ts"]
                    if PROFILE_OPERATORS:
                        for mode, layer in (("native", native), ("split_bias", split)):
                            row[mode]["operators"] = sample_operators(layer, x, gradient, device)
                    print(f"{device} {window}ms B={batch} d={dilation}: checks PASS; "
                          f"forward ratio native/split={row['forward_speedup']:.2f}", flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {OUTPUT}", flush=True)
    return report


if __name__ == "__main__":
    main()
