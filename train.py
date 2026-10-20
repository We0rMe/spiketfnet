"""固定末轮 checkpoint 的 LOSO 训练；每折 M 个独立连续种子。"""

import csv
import importlib.metadata
import json
import os
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

from config import CONFIG, PROJECT_ROOT
from data import WindowDataset, file_sha256, loso_indices, prepare_data
from evaluate import evaluate_model, resolve_device, run_path, save_evaluation, summarize_experiment, write_json
from model import SpikeTFNet, joint_loss


def seed_everything(seed, cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg.deterministic


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def environment_info(device):
    packages = ("torch", "torchvision", "spikingjelly", "numpy", "scipy", "matplotlib")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for time, spectrum, labels in loader:
        time, spectrum, labels = time.to(device), spectrum.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(time, spectrum)
        loss = joint_loss(outputs, labels, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        finite_gradients = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
        if not finite_gradients:
            raise FloatingPointError("Nonfinite training gradients")
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += int((outputs["logits"].argmax(1) == labels).sum().item())
        count += len(labels)
    if count == 0:
        raise ValueError("Empty training loader")
    return {"loss": total_loss / count, "train_acc_percent": 100 * correct / count}


def plot_history(history, destination, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    axes[0].plot(epochs, [row["loss"] for row in history], marker="." if len(history) < 10 else None)
    axes[0].set(xlabel="Epoch", ylabel="Joint training loss")
    axes[1].plot(epochs, [row["train_acc_percent"] for row in history], marker="." if len(history) < 10 else None)
    axes[1].set(xlabel="Epoch", ylabel="Training ACC (%)", ylim=(0, 100))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def training_devices(cfg):
    """显式 CUDA 配置不可静默回落；auto 在没有 GPU 时允许 CPU。"""
    cfg.validate()
    if cfg.device == "cpu" or cfg.device.startswith("cuda:"):
        return [resolve_device(cfg.device)]
    if not torch.cuda.is_available():
        return [resolve_device("cpu" if cfg.device == "auto" else "cuda")]
    count = torch.cuda.device_count()
    selected = [index for index in cfg.gpu_ids if index < count]
    if cfg.device == "cuda" and len(selected) != len(cfg.gpu_ids):
        raise ValueError(f"Requested gpu_ids={cfg.gpu_ids}, but only {count} GPUs are visible")
    if not selected:
        raise ValueError("No configured GPU is visible; check gpu_ids and CUDA_VISIBLE_DEVICES")
    return [resolve_device(f"cuda:{index}") for index in selected]


def experiment_jobs(cfg):
    return [(window, subject, seed) for window in cfg.windows_ms
            for subject in cfg.test_subjects for seed in cfg.seeds]


def partition_jobs(jobs, rank, workers):
    if workers < 1 or not 0 <= rank < workers:
        raise ValueError("Invalid worker partition")
    return jobs[rank::workers]


def train_run(cfg, prepared, dataset, job, device, root, purpose):
    window, subject, seed = job
    training_rows, test_rows = loso_indices(prepared.ids, subject, cfg.subject_ids)
    seed_everything(seed, cfg)
    folder = run_path(root, window, subject, seed)
    private = folder / "details"
    private.mkdir(parents=True, exist_ok=False)
    provenance = {"status": "training", "purpose": purpose, "test_subject": subject,
                  "training_subjects": sorted(set(prepared.ids[training_rows, 0].tolist())),
                  "seed": seed, "window_ms": window, "epochs": cfg.epochs,
                  "device": str(device), "worker_pid": os.getpid(),
                  "training_trials": len(training_rows), "test_trials": len(test_rows)}
    write_json(private / "provenance.json", provenance)
    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers,
              "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    train_loader = DataLoader(Subset(dataset, training_rows.tolist()), shuffle=True, generator=generator, **common)
    test_loader = DataLoader(Subset(dataset, test_rows.tolist()), shuffle=False, **common)
    model = SpikeTFNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history = []
    with (private / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "loss", "train_acc_percent"))
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            row = {"epoch": epoch, **train_epoch(model, train_loader, optimizer, cfg, device)}
            history.append(row)
            writer.writerow(row)
            handle.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
                print(f"[{device}] {window}ms S{subject:02d} seed={seed} epoch={epoch}/{cfg.epochs} "
                      f"loss={row['loss']:.4f} train_ACC={row['train_acc_percent']:.2f}%", flush=True)
    # 不保存每轮 checkpoint；测试只接触固定末轮权重。
    torch.save(model.state_dict(), private / "model.pt")
    plot_history(history, folder / "training.png", f"{window} ms | test S{subject:02d} | seed {seed} | {purpose}")
    metrics, arrays = evaluate_model(model, test_loader, cfg, window, device)
    save_evaluation(folder, metrics, arrays, prepared.ids[training_rows], prepared.ids[test_rows])
    provenance["status"] = "complete"
    write_json(private / "provenance.json", provenance)
    print(f"[{device}] {window}ms S{subject:02d} seed={seed} final test: "
          f"ACC={metrics['acc_percent']:.2f}%, ITR={metrics['itr_bits_per_minute']:.2f} bits/min", flush=True)
    del model, optimizer, arrays


def train_worker(rank, devices, cfg, prepared, root, purpose, jobs):
    device = resolve_device(devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(cfg.cpu_threads)
    os.environ["MPLCONFIGDIR"] = str(Path(root) / "details" / "plot_cache" / f"worker_{rank}")
    dataset, current_window = None, None
    for job in partition_jobs(jobs, rank, len(devices)):
        if job[0] != current_window:
            dataset = WindowDataset(prepared, cfg, job[0])
            current_window = job[0]
        train_run(cfg, prepared, dataset, job, device, root, purpose)


def execute_jobs(devices, cfg, prepared, root, purpose, jobs):
    names = [str(device) for device in devices]
    args = (names, cfg, prepared, root, purpose, jobs)
    if len(names) == 1:
        train_worker(0, *args)
    else:
        # spawn 隔离 CUDA、BN 和 LIF 状态；子进程失败时由 PyTorch 终止其余 worker。
        mp.spawn(train_worker, args=args, nprocs=len(names), join=True)


def run_experiment(cfg=CONFIG, prepared=None, purpose="formal"):
    cfg.validate()
    if purpose not in ("formal", "smoke"):
        raise ValueError("purpose must be formal or smoke")
    if purpose == "formal" and cfg.subject_ids != tuple(range(1, 36)):
        raise ValueError("Formal Benchmark LOSO requires all 35 source-pool subjects; change N to limit test folds")
    root = cfg.output_root / cfg.experiment_name
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}; change experiment_name")
    jobs = experiment_jobs(cfg)
    devices = training_devices(cfg)[:len(jobs)]
    print(f"Training devices: {[str(d) for d in devices]}; independent jobs: {len(jobs)}", flush=True)
    torch.set_num_threads(cfg.cpu_threads)
    prepared = prepare_data(cfg) if prepared is None else prepared.validate(cfg)
    if purpose == "formal" and len(prepared.ids) != len(cfg.subject_ids) * cfg.num_blocks * cfg.num_classes:
        raise ValueError("Formal experiment must use all trials")
    details = root / "details"
    details.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(details / "plot_cache"))
    source_names = ("config.py", "data.py", "model.py", "train.py", "evaluate.py", "requirements.txt")
    code_hashes = {name: file_sha256(PROJECT_ROOT / name) for name in source_names}
    (details / "source").mkdir()
    for name in source_names:
        shutil.copy2(PROJECT_ROOT / name, details / "source" / name)
    decision_file = PROJECT_ROOT / "docs" / "IMPLEMENTATION_NOTES.zh-CN.md"
    if decision_file.exists():
        shutil.copy2(decision_file, details / "source" / decision_file.name)
    manifest = {"config": cfg.to_dict(), "purpose": purpose, "status": "running",
                "created_utc": datetime.now(timezone.utc).isoformat(), "environment": environment_info(devices[0]),
                "training_devices": [str(d) for d in devices],
                "gpu_names": {str(d): torch.cuda.get_device_name(d) for d in devices if d.type == "cuda"},
                "parallel_strategy": "independent window/subject/seed jobs, one worker per device",
                "dataset_fingerprint": prepared.metadata["fingerprint"], "code_sha256": code_hashes,
                "selected_trial_count": len(prepared.ids), "selection": "final epoch only; no target-dependent selection",
                "statistics": "Population variance by default, not standard deviation; ITR computed per subject/seed first"}
    write_json(details / "experiment.json", manifest)
    write_json(details / "dataset.json", prepared.metadata)
    try:
        execute_jobs(devices, cfg, prepared, root, purpose, jobs)
        summarize_experiment(root, cfg)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(details / "experiment.json", manifest)
    return root


if __name__ == "__main__":
    run_experiment()
