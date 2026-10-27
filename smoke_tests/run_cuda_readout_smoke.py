"""服务器快速检查：严格确定性下的32段读出及完整时间分支更新，无需EEG文件。"""

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG
from evaluate import write_json
from experiments.variants import build_model
from model import TemporalBranch
from train import seed_everything, train_epoch, training_devices
from utils.operator_profile import runtime_settings


def check_device(device):
    cfg = replace(CONFIG, model_variant="temporal_only", windows_ms=(500,), temporal_readout_bins=32,
                  deterministic=True, check_gradients=True, device=str(device))
    torch.cuda.set_device(device)
    torch.set_num_threads(cfg.cpu_threads)
    seed_everything(cfg.seed, cfg)
    branch = TemporalBranch(cfg).to(device)
    for length in (125, 250):
        source = torch.randn(2, cfg.temporal_width, length, requires_grad=True)
        weights = torch.randn(2, cfg.temporal_width * 32)
        reference = torch.nn.functional.adaptive_avg_pool1d(source, 32).flatten(1)
        reference_grad = torch.autograd.grad(reference, source, weights)[0]
        repeats = []
        for _ in range(2):
            inputs = source.detach().to(device).requires_grad_()
            actual = branch.pool_features(inputs)
            gradient = torch.autograd.grad(actual, inputs, weights.to(device))[0]
            torch.testing.assert_close(actual.cpu(), reference, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(gradient.cpu(), reference_grad, rtol=1e-5, atol=1e-6)
            repeats.append((actual.detach().cpu(), gradient.cpu()))
        for a, b in zip(*repeats):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    del branch
    data = TensorDataset(torch.randn(cfg.batch_size * 2, 9, 125),
                         torch.randn(cfg.batch_size * 2, 9, cfg.spectral_features),
                         torch.arange(cfg.batch_size * 2) % cfg.num_classes)
    states = []
    for _ in range(2):
        seed_everything(cfg.seed, cfg)
        model = build_model(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=False)
        train_epoch(model, loader, optimizer, cfg, device)
        torch.cuda.synchronize(device)
        states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        del model, optimizer
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    return {"runtime": runtime_settings(device), "pool_forward_backward_matches_cpu_reference": True,
            "two_optimizer_steps_repeated_exactly": True, "batch_size": cfg.batch_size}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires the server CUDA environment; CPU cannot verify this failure.")
    devices = training_devices(CONFIG)
    if any(device.type != "cuda" for device in devices):
        raise ValueError("Set the intended CUDA device in config.py before this server check")
    results = []
    for device in devices:
        print(f"Checking {device}: strict deterministic pooling + temporal_only training...", flush=True)
        results.append(check_device(device))
        print(f"PASS {device}", flush=True)
    report = ROOT / "smoke_tests/results/cuda_readout_check/report.json"
    write_json(report, {"status": "passed", "checks": results})
    print(json.dumps({"status": "passed", "report": str(report)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
