"""固定末轮 checkpoint 的 LOSO 训练；每折 M 个独立连续种子。"""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

from config import CONFIG, PROJECT_ROOT, Config
from data import WindowDataset, file_sha256, loso_indices, prepare_data
from evaluate import (evaluate_model, plot_confusion, plot_training, read_history, refresh_progress,
                      resolve_device, run_path, save_evaluation, summarize_experiment, write_json, write_run_report)
from ablation_model import build_model, joint_loss

SOURCE_NAMES = ("config.py", "data.py", "model.py", "ablation_model.py", "ablation.py",
                "train.py", "evaluate.py", "requirements.txt")


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
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0
    for time, spectrum, labels in loader:
        time, spectrum, labels = (value.to(device, non_blocking=True) for value in (time, spectrum, labels))
        optimizer.zero_grad(set_to_none=True)
        outputs = model(time, spectrum)
        loss = joint_loss(outputs, labels, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if cfg.check_gradients:
            finite_gradients = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                                            if p.grad is not None]).all()
            if not finite_gradients:
                raise FloatingPointError("Nonfinite training gradients")
        optimizer.step()
        total_loss += loss.detach().double() * len(labels)
        correct += (outputs["logits"].argmax(1) == labels).sum()
        count += len(labels)
    if count == 0:
        raise ValueError("Empty training loader")
    finite_parameters = torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
    if not finite_parameters:
        raise FloatingPointError("Nonfinite model parameters at epoch end")
    return {"loss": total_loss.item() / count, "train_acc_percent": 100 * correct.item() / count}


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.console.write(text)

    def flush(self):
        self.log.flush()
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def log_output(path):
    with Path(path).open("a", encoding="utf-8", buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            print(f"\nSESSION {datetime.now(timezone.utc).isoformat()}", flush=True)
            yield
        except BaseException:
            traceback.print_exc(file=log)
            log.flush()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def plot_history(history, destination, title):
    plot_training(history, destination, title)


def monitor_training(model, dataset, rows, cfg, device, generator):
    """只读源训练子集；恢复随机状态与模式，不影响后续优化轨迹。"""
    state, was_training = rng_state(generator, device), model.training
    started = time.perf_counter()
    try:
        loader = DataLoader(Subset(dataset, rows.tolist()), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(0))
        model.eval()
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        with torch.inference_mode():
            for time_data, spectrum, labels in loader:
                time_data, spectrum, labels = (x.to(device, non_blocking=True) for x in (time_data, spectrum, labels))
                output = model(time_data, spectrum)
                loss_sum += joint_loss(output, labels, cfg).double() * len(labels)
                correct += (output["logits"].argmax(1) == labels).sum()
        if not len(rows) or not torch.isfinite(loss_sum):
            raise ValueError("Empty or nonfinite training monitor")
        return {"monitor_loss": loss_sum.item() / len(rows),
                "monitor_acc_percent": correct.item() * 100 / len(rows),
                "monitor_seconds": time.perf_counter() - started}
    finally:
        model.train(was_training)
        restore_rng(state, generator, device)


def recover_log(folder, history):
    path = folder / "training.log"
    if path.exists():
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.write("RECOVERED history below is reconstructed from saved CSV/checkpoint, not original console output.\n")
        for row in history:
            handle.write(f"RECOVERED epoch={row['epoch']} loss={row['loss']:.6f} "
                         f"train_ACC={row['train_acc_percent']:.4f}% train_seconds={row.get('train_seconds')}\n")


def update_run_artifacts(folder, cfg, history, provenance):
    title = f"{provenance['window_ms']} ms | test S{provenance['test_subject']:02d} | seed {provenance['seed']}"
    plot_history(history, folder / "training.png", title)
    write_run_report(folder, cfg, history, provenance)


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


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def rng_state(generator, device):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
    elif (device.type == "cuda") != (state["cuda"] is not None):
        print("Resume changed CPU/CUDA backend; bitwise reproducibility is not guaranteed.", flush=True)


@contextmanager
def preserved_rng(generator, device):
    state = rng_state(generator, device)
    try:
        yield
    finally:
        restore_rng(state, generator, device)


def write_history(path, history):
    with path.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "loss", "train_acc_percent", "train_seconds",
                                                    "monitor_loss", "monitor_acc_percent", "monitor_seconds"))
        writer.writeheader()
        writer.writerows(history)
    path.with_suffix(".tmp").replace(path)


def save_resume(path, model, optimizer, generator, device, history, job, prepared):
    # 膜电位在 forward 后已重置；保存 epoch 边界可恢复的随机数和数据顺序。
    atomic_torch_save({"format_version": 1, "job": job, "epoch": len(history), "history": history,
                       "model_variant": getattr(model, "model_variant", "full"),
                       "dataset_fingerprint": prepared.metadata["fingerprint"],
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": rng_state(generator, device)}, path)


def completed_job(folder, job, prepared, cfg):
    private = folder / "details"
    provenance_file = private / "provenance.json"
    if not provenance_file.exists():
        return False
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if provenance.get("model_variant", "full") != cfg.model_variant:
        raise ValueError(f"Run model variant mismatch: {folder}")
    if (provenance["window_ms"], provenance["test_subject"], provenance["seed"]) != job:
        raise ValueError(f"Run identity mismatch: {folder}")
    if provenance["status"] != "complete":
        return False
    required = ("model.pt", "history.csv", "evaluation.npz")
    if not all((private / name).is_file() for name in required) or not all(
            (folder / name).is_file() for name in ("metrics.json", "training.png")):
        raise ValueError(f"Completed run has missing files: {folder}")
    training_rows, test_rows = loso_indices(prepared.ids, job[1], cfg.subject_ids)
    with np.load(private / "evaluation.npz", allow_pickle=False) as arrays:
        if not np.array_equal(arrays["training_ids"], prepared.ids[training_rows]) or not np.array_equal(
                arrays["test_ids"], prepared.ids[test_rows]):
            raise ValueError(f"Completed run has a different LOSO split: {folder}")
    history = read_history(private / "history.csv")
    if not (folder / "report.md").exists():
        update_run_artifacts(folder, cfg, history, provenance)
    if not (folder / "confusion.png").exists():
        plot_confusion(folder, cfg)
    recover_log(folder, history)
    return True


def train_run(cfg, prepared, dataset, job, device, root, purpose):
    folder = run_path(root, *job)
    folder.mkdir(parents=True, exist_ok=True)
    reconstruct_log = not (folder / "training.log").exists()
    with log_output(folder / "training.log"):
        return train_run_logged(cfg, prepared, dataset, job, device, root, purpose, reconstruct_log)


def train_run_logged(cfg, prepared, dataset, job, device, root, purpose, reconstruct_log=False):
    window, subject, seed = job
    training_rows, test_rows = loso_indices(prepared.ids, subject, cfg.subject_ids)
    monitor_rows = training_rows[prepared.ids[training_rows, 1] == 1]
    seed_everything(seed, cfg)
    folder = run_path(root, window, subject, seed)
    private = folder / "details"
    private.mkdir(parents=True, exist_ok=True)
    resume_file = private / "resume.pt"
    provenance = {"status": "training", "purpose": purpose, "test_subject": subject,
                  "model_variant": cfg.model_variant,
                  "training_subjects": sorted(set(prepared.ids[training_rows, 0].tolist())),
                  "seed": seed, "window_ms": window, "epochs": cfg.epochs,
                  "device": str(device), "worker_pid": os.getpid(),
                  "training_trials": len(training_rows), "test_trials": len(test_rows),
                  "monitor": "Source training subjects, block 1, eval mode; NOT validation",
                  "monitor_trials": len(monitor_rows), "monitor_every": cfg.monitor_every}
    write_json(private / "provenance.json", provenance)
    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers,
              "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    train_loader = DataLoader(Subset(dataset, training_rows.tolist()), shuffle=True, generator=generator, **common)
    test_loader = DataLoader(Subset(dataset, test_rows.tolist()), shuffle=False, **common)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history = []
    if resume_file.exists():
        state = torch.load(resume_file, map_location="cpu", weights_only=True)
        if state.get("model_variant", "full") != cfg.model_variant:
            raise ValueError(f"Resume model variant mismatch: {resume_file}")
        if state["format_version"] != 1 or tuple(state["job"]) != job or state[
                "dataset_fingerprint"] != prepared.metadata["fingerprint"]:
            raise ValueError(f"Resume checkpoint identity mismatch: {resume_file}")
        history = state["history"]
        if state["epoch"] != len(history) or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
            raise ValueError("Resume history does not match the checkpoint epoch")
        if len(history) > cfg.epochs:
            raise ValueError("Checkpoint exceeds configured epochs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], generator, device)
        del state
        print(f"[{device}] {window}ms S{subject:02d} seed={seed}: resume after epoch {len(history)}", flush=True)
    elif (private / "model.pt").exists():
        # 兼容旧版：已写出末轮权重但测试/绘图中断，仅补完评估。
        model.load_state_dict(torch.load(private / "model.pt", map_location=device, weights_only=True))
        history = read_history(private / "history.csv")
        if [row["epoch"] for row in history] != list(range(1, cfg.epochs + 1)):
            raise ValueError("Final model has no matching full training history")
        print(f"[{device}] S{subject:02d} seed={seed}: final weights found; finishing evaluation only", flush=True)
    else:
        if (private / "history.csv").exists():
            shutil.copy2(private / "history.csv", private / "history_before_restart.csv")
            print(f"[{device}] S{subject:02d} seed={seed}: no saved weights; restart from epoch 1", flush=True)
        save_resume(resume_file, model, optimizer, generator, device, history, job, prepared)
    write_history(private / "history.csv", history)
    if reconstruct_log and history:
        print("RECOVERED rows below come from saved history, not original console output.", flush=True)
        for row in history:
            print(f"RECOVERED epoch={row['epoch']} loss={row['loss']:.6f} "
                  f"train_ACC={row['train_acc_percent']:.4f}% train_seconds={row.get('train_seconds')}", flush=True)
    provenance["resumed_after_epoch"] = len(history)
    write_json(private / "provenance.json", provenance)
    print(f"TRAIN {len(training_rows)} trials; MONITOR {len(monitor_rows)} source block-1 trials (NOT validation); "
          f"TEST {len(test_rows)} trials after epoch {cfg.epochs} only.", flush=True)
    with preserved_rng(generator, device):
        update_run_artifacts(folder, cfg, history, provenance)
    try:
        for epoch in range(len(history) + 1, cfg.epochs + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            row = {"epoch": epoch, **train_epoch(model, train_loader, optimizer, cfg, device)}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row["train_seconds"] = time.perf_counter() - started
            if cfg.monitor_every and (epoch == 1 or epoch % cfg.monitor_every == 0 or epoch == cfg.epochs):
                row.update(monitor_training(model, dataset, monitor_rows, cfg, device, generator))
            history.append(row)
            saved_at = time.perf_counter()
            save_resume(resume_file, model, optimizer, generator, device, history, job, prepared)
            write_history(private / "history.csv", history)
            save_seconds = time.perf_counter() - saved_at
            plot_started = time.perf_counter()
            if epoch % cfg.plot_every == 0 or epoch == 1 or epoch == cfg.epochs:
                with preserved_rng(generator, device):
                    update_run_artifacts(folder, cfg, history, provenance)
            plot_seconds = time.perf_counter() - plot_started
            monitor_text = (f"monitor_loss={row['monitor_loss']:.4f} "
                            f"monitor_ACC={row['monitor_acc_percent']:.2f}% (NOT validation) "
                            if "monitor_loss" in row else "")
            print(f"[{device}] {window}ms S{subject:02d} seed={seed} epoch={epoch}/{cfg.epochs} "
                  f"loss={row['loss']:.4f} train_ACC={row['train_acc_percent']:.2f}% "
                  f"{monitor_text}train={row['train_seconds']:.2f}s monitor={row.get('monitor_seconds', 0):.2f}s "
                  f"save={save_seconds:.2f}s plot={plot_seconds:.2f}s "
                  f"total={time.perf_counter() - started:.2f}s", flush=True)
        atomic_torch_save(model.state_dict(), private / "model.pt")
        plot_history(history, folder / "training.png", f"{window} ms | test S{subject:02d} | seed {seed} | {purpose}")
        metrics, arrays = evaluate_model(model, test_loader, cfg, window, device)
        save_evaluation(folder, metrics, arrays, prepared.ids[training_rows], prepared.ids[test_rows])
        provenance["status"] = "complete"
        plot_confusion(folder, cfg)
        write_run_report(folder, cfg, history, provenance)
        write_json(private / "provenance.json", provenance)
        resume_file.unlink(missing_ok=True)
    except BaseException as exc:
        provenance.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=f"{type(exc).__name__}: {exc}")
        write_json(private / "provenance.json", provenance)
        write_run_report(folder, cfg, read_history(private / "history.csv"), provenance)
        print(f"STOPPED: {provenance['status']}; resume uses the last committed checkpoint.", flush=True)
        raise
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
        if len(devices) == 1:
            refresh_progress(root, cfg)


def execute_jobs(devices, cfg, prepared, root, purpose, jobs):
    if not jobs:
        return
    names = [str(device) for device in devices]
    args = (names, cfg, prepared, root, purpose, jobs)
    if len(names) == 1:
        train_worker(0, *args)
    else:
        context = mp.spawn(train_worker, args=args, nprocs=len(names), join=False)
        try:
            while not context.join(timeout=5):
                refresh_progress(root, cfg)
        except BaseException:
            # 父进程 Ctrl+C 时也清理所有训练子进程，避免它们继续写同一实验。
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join()
            raise


@contextmanager
def experiment_lock(details):
    with (details / "training.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("This experiment is already running in another process") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


REPORTING_V2_SOURCES = {
    "train.py": "67e00a4b9018a5be210cd3967760c6203cee4055668bd4569ad98ef65fd40704",
    "evaluate.py": "1c800e8cc51dfae59ce18ef64af81150011805afceaae5d1e5ed318409cb36dd",
}

ABLATION_V3_SOURCES = {
    "train.py": "4fff6ebc49e0b31e23dbccbe598e4a2bac1fb5390b75a361f042431646144edc",
    "evaluate.py": "dca51e7f17af5f882844a8f9156aca5165d666f311e12b7feae9b347ff9423c3",
}


def verified_reporting_upgrade(name, old_hash, details):
    return verified_source_upgrade(name, old_hash, details, REPORTING_V2_SOURCES)


def verified_source_upgrade(name, old_hash, details, approved):
    if details is None or name not in approved:
        return False
    for directory in ("source_resume", "source"):
        source = Path(details) / directory / name
        if source.is_file() and file_sha256(source) == old_hash:
            normalized = source.read_bytes().replace(b"\r\n", b"\n")
            return hashlib.sha256(normalized).hexdigest() == approved[name]
    return False


def validate_resume(manifest, cfg, prepared, purpose, code_hashes, details=None):
    previous = Config.from_dict(manifest["config"]).to_dict()
    allowed = {"N", "M", "seed", "windows_ms", "device", "gpu_ids", "evaluation_device", "num_workers",
               "cpu_threads", "data_root", "cache_root", "output_root", "experiment_name",
               "resume", "check_gradients", "variance_ddof", "monitor_every", "plot_every"}
    differences = [key for key, value in cfg.to_dict().items() if key not in allowed and previous[key] != value]
    if differences:
        raise ValueError(f"Resume would change training/model settings: {differences}; use a new experiment_name")
    if manifest["purpose"] != purpose or manifest["dataset_fingerprint"] != prepared.metadata["fingerprint"]:
        raise ValueError("Resume dataset fingerprint or experiment purpose differs")
    selection_hash = hashlib.sha256(prepared.ids.tobytes()).hexdigest()
    if manifest["selected_trial_count"] != len(prepared.ids) or manifest.get("selection_sha256", selection_hash) != selection_hash:
        raise ValueError("Resume trial selection differs")
    # 旧版可升级训练/评估基础设施；网络和预处理代码必须完全相同。
    checked_sources = ("model.py", "data.py") if manifest.get("format_version", 1) == 1 else (
        "model.py", "data.py", "train.py", "evaluate.py")
    if manifest.get("format_version", 1) >= 4:
        checked_sources += ("ablation_model.py",)
    elif cfg.model_variant != "full":
        raise ValueError("Legacy experiments can only resume as full models")
    for name in checked_sources:
        previous_hash = manifest["code_sha256"][name]
        if previous_hash == code_hashes[name]:
            continue
        if manifest.get("format_version") == 2 and verified_reporting_upgrade(name, previous_hash, details):
            print(f"Verified reporting-only v2 upgrade: {name}; training semantics preserved.", flush=True)
            continue
        if manifest.get("format_version") == 3 and verified_source_upgrade(
                name, previous_hash, details, ABLATION_V3_SOURCES):
            print(f"Verified v3 full-model upgrade: {name}; original model and training preserved.", flush=True)
            continue
        raise ValueError(f"Resume source code differs: {name}; unverified changes cannot resume this experiment")


def run_experiment(cfg=CONFIG, prepared=None, purpose="formal"):
    cfg.validate()
    if purpose not in ("formal", "smoke"):
        raise ValueError("purpose must be formal or smoke")
    if purpose == "formal" and cfg.subject_ids != tuple(range(1, 36)):
        raise ValueError("Formal Benchmark LOSO requires all 35 source-pool subjects; change N to limit test folds")
    root = cfg.output_root / cfg.experiment_name
    if root.exists() and not cfg.resume:
        raise FileExistsError(f"Refusing to overwrite {root}; change experiment_name")
    details = root / "details"
    details.mkdir(parents=True, exist_ok=True)
    with experiment_lock(details):
        existed = (details / "experiment.json").is_file()
        if existed and not cfg.resume:
            raise FileExistsError(f"Refusing to overwrite {root}; change experiment_name")
        if not existed:
            # 准备数据时中断只留下锁文件，可安全重新准备；未知文件不覆盖。
            if any(path.relative_to(root).as_posix() not in ("details", "details/training.lock", "details/experiment.log")
                   for path in root.rglob("*")):
                raise ValueError(f"Existing experiment has no manifest: {root}; use a new experiment_name")
        with log_output(details / "experiment.log"):
            return run_locked_experiment(cfg, prepared, purpose, root, existed)


def run_locked_experiment(cfg, prepared, purpose, root, existed):
    jobs = experiment_jobs(cfg)
    devices = training_devices(cfg)
    torch.set_num_threads(cfg.cpu_threads)
    prepared = prepare_data(cfg) if prepared is None else prepared.validate(cfg)
    if purpose == "formal" and len(prepared.ids) != len(cfg.subject_ids) * cfg.num_blocks * cfg.num_classes:
        raise ValueError("Formal experiment must use all trials")
    details = root / "details"
    os.environ.setdefault("MPLCONFIGDIR", str(details / "plot_cache"))
    source_names = SOURCE_NAMES
    code_hashes = {name: file_sha256(PROJECT_ROOT / name) for name in source_names}
    previous = json.loads((details / "experiment.json").read_text(encoding="utf-8")) if existed else None
    if previous is not None:
        validate_resume(previous, cfg, prepared, purpose, code_hashes, details)
        if previous.get("format_version", 1) >= 2:
            for job in jobs:
                private = run_path(root, *job) / "details"
                if read_history(private / "history.csv") and not any(
                        (private / name).is_file() for name in ("resume.pt", "model.pt")):
                    raise FileNotFoundError(f"Saved training history has no recoverable weights: {private}; "
                                            "refusing to restart this task silently")
    pending = [job for job in jobs if not completed_job(run_path(root, *job), job, prepared, cfg)]
    devices = devices[:max(1, len(pending))]
    print(f"Training devices: {[str(d) for d in devices]}; jobs={len(jobs)}, "
          f"completed={len(jobs) - len(pending)}, pending={len(pending)}", flush=True)
    source_dir = details / ("source_resume" if existed else "source")
    source_dir.mkdir(exist_ok=True)
    for name in source_names:
        shutil.copy2(PROJECT_ROOT / name, source_dir / name)
    decision_file = PROJECT_ROOT / "docs" / "IMPLEMENTATION_NOTES.zh-CN.md"
    if decision_file.exists():
        shutil.copy2(decision_file, source_dir / decision_file.name)
    manifest = {"config": cfg.to_dict(), "purpose": purpose, "status": "running",
                "created_utc": datetime.now(timezone.utc).isoformat(), "environment": environment_info(devices[0]),
                "training_devices": [str(d) for d in devices],
                "gpu_names": {str(d): torch.cuda.get_device_name(d) for d in devices if d.type == "cuda"},
                "parallel_strategy": "independent window/subject/seed jobs, one worker per device",
                "dataset_fingerprint": prepared.metadata["fingerprint"], "code_sha256": code_hashes,
                "selected_trial_count": len(prepared.ids), "selection": "final epoch only; no target-dependent selection",
                "statistics": "Population variance by default, not standard deviation; ITR computed per subject/seed first"}
    manifest.update(format_version=4, selection_sha256=hashlib.sha256(prepared.ids.tobytes()).hexdigest(),
                    monitoring="Source training block 1 in eval mode; no validation set or test-based selection")
    if previous is not None:
        manifest["created_utc"] = previous["created_utc"]
        manifest["resumptions"] = previous.get("resumptions", []) + [{
            "utc": datetime.now(timezone.utc).isoformat(), "previous_config": previous["config"],
            "previous_code_sha256": previous["code_sha256"], "previous_environment": previous["environment"]}]
    write_json(details / "experiment.json", manifest)
    write_json(details / "dataset.json", prepared.metadata)
    # 运行范围可从 M=3 缩为 M=1；防止旧范围的汇总在恢复期间被误读。
    (root / "summary.csv").unlink(missing_ok=True)
    refresh_progress(root, cfg)
    try:
        execute_jobs(devices, cfg, prepared, root, purpose, pending)
        summarize_experiment(root, cfg)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(details / "experiment.json", manifest)
        refresh_progress(root, cfg)
    return root


if __name__ == "__main__":
    run_experiment()
