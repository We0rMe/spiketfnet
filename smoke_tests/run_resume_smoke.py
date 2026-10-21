"""真实数据下，验证中断恢复、双进程、旧结果升级和 M 缩减。"""

import csv
import io
import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from config import CONFIG
from data import prepare_data
from evaluate import reevaluate_experiment, run_path, write_json
import train
from smoke_tests.assertions import metric_rows


def compare_runs(reference, actual):
    assert metric_rows(reference / "summary.csv") == metric_rows(actual / "summary.csv")
    folders = sorted(actual.glob("window_*/test_*/seed_*"))
    assert len(folders) == 4
    for folder in folders:
        other = reference / folder.relative_to(actual)
        assert (folder / "metrics.json").read_bytes() == (other / "metrics.json").read_bytes()
        states = [torch.load(p / "details/model.pt", map_location="cpu", weights_only=True) for p in (folder, other)]
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
        with np.load(folder / "details/evaluation.npz") as a, np.load(other / "details/evaluation.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            assert not set(a["training_ids"][:, 0]) & set(a["test_ids"][:, 0])
        histories = []
        for path in (folder, other):
            with (path / "details/history.csv").open() as handle:
                histories.append(list(csv.DictReader(handle)))
        assert len(histories[0]) == len(histories[1]) == 3
        for a, b in zip(*histories):
            assert all(a[key] == b[key] for key in ("epoch", "loss", "train_acc_percent"))
            assert float(a["train_seconds"]) > 0
        assert not (folder / "details/resume.pt").exists()


def main():
    base = ROOT / "smoke_tests/results/resume_check"
    base.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(base / "plot_cache")
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "smoke_tests"), pattern="test_*.py")
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (base / "checks.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(stream.getvalue(), flush=True)
    if not outcome.wasSuccessful():
        raise SystemExit("Core checks failed")
    cfg = replace(CONFIG, data_root=Path(r"G:\SSVEP\data\benchmark"), subject_ids=(1, 2, 3),
                  N=2, M=2, seed=42, windows_ms=(500,), epochs=3, batch_size=8,
                  device="cpu", evaluation_device="cpu", resume=True, num_workers=0, cpu_threads=2,
                  cache_root=base.parent / "cache", output_root=base, experiment_name="reference")
    prepared = prepare_data(cfg)
    selected = prepared.subset(np.flatnonzero((prepared.ids[:, 1] == 1) & (prepared.ids[:, 2] % 5 == 0)))
    reference = train.run_experiment(cfg, selected, purpose="smoke")
    resumed_cfg = replace(cfg, experiment_name="resumed")
    original_save = train.save_resume

    def interrupt_after_epoch_one(path, model, optimizer, generator, device, history, job, data):
        original_save(path, model, optimizer, generator, device, history, job, data)
        if len(history) == 1:
            raise KeyboardInterrupt("Simulated interruption after committed checkpoint")

    with patch("train.save_resume", side_effect=interrupt_after_epoch_one):
        try:
            train.run_experiment(resumed_cfg, selected, purpose="smoke")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("Interruption was not injected")
    partial = run_path(base / "resumed", 500, 1, 42) / "details"
    checkpoint = torch.load(partial / "resume.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1 and checkpoint["optimizer"]["state"]
    # 模拟 CSV 写入滞后；恢复必须以原子 checkpoint 的历史为准。
    (partial / "history.csv").write_text("epoch,loss,train_acc_percent,train_seconds\n", encoding="utf-8")
    with patch("train.training_devices", return_value=[torch.device("cpu"), torch.device("cpu")]):
        resumed = train.run_experiment(resumed_cfg, selected, purpose="smoke")
    compare_runs(reference, resumed)
    pids = {json.loads(path.read_text(encoding="utf-8"))["worker_pid"]
            for path in resumed.glob("window_*/test_*/seed_*/details/provenance.json")}
    assert len(pids) == 2 and os.getpid() not in pids
    with patch("train.train_epoch", side_effect=AssertionError("Completed run trained again")):
        train.run_experiment(resumed_cfg, selected, purpose="smoke")
        train.run_experiment(replace(resumed_cfg, M=1), selected, purpose="smoke")
    assert len(list(resumed.glob("window_*/test_*/seed_*"))) == 4
    with (resumed / "summary.csv").open(encoding="utf-8-sig") as handle:
        summary = list(csv.DictReader(handle))
    assert all(row["member"] == "42" for row in summary if row["scope"] == "within_repeat")
    assert all(float(row["variance"]) == 0 for row in summary if row["scope"] == "across_repeats")
    reevaluate_experiment(resumed, prepared=selected, device_name="cpu")

    # 旧版无 resume 文件：完成任务可升级复用，未完成任务明确从零重训。
    legacy = train.run_experiment(replace(cfg, N=1, M=1, experiment_name="legacy"), selected, purpose="smoke")
    manifest_path = legacy / "details/experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("format_version")
    manifest["config"].pop("resume")
    manifest["config"].pop("check_gradients")
    manifest["code_sha256"]["train.py"] = "legacy_train"
    write_json(manifest_path, manifest)
    with patch("train.train_epoch", side_effect=AssertionError("Legacy completed run trained again")):
        train.run_experiment(replace(cfg, N=1, M=1, experiment_name="legacy"), selected, purpose="smoke")
    legacy_private = run_path(legacy, 500, 1, 42) / "details"
    provenance = json.loads((legacy_private / "provenance.json").read_text(encoding="utf-8"))
    provenance["status"] = "training"
    write_json(legacy_private / "provenance.json", provenance)
    (legacy_private / "model.pt").unlink()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("format_version")
    write_json(manifest_path, manifest)
    train.run_experiment(replace(cfg, N=1, M=1, experiment_name="legacy"), selected, purpose="smoke")
    assert (legacy_private / "history_before_restart.csv").is_file()
    report = (f"# Resume smoke: PASS\n\n{outcome.testsRun} core/device/recovery checks passed.\n"
              "Real Benchmark S1-S3, 8 trials per subject; 2 folds x 2 seeds x 3 epochs, full model.\n"
              "Interrupted after epoch 1, restored model + Adam + RNG + loader order, then used two CPU workers.\n"
              "Weights, training metrics, evaluation arrays and summary matched uninterrupted training exactly.\n"
              "Timing columns were positive and excluded from numerical equality checks.\n"
              "Completed jobs skipped; M reduction excluded extra seeds from summary without deleting files.\n"
              "Legacy completion reused; legacy partial job without weights restarted explicitly.\n"
              "Atomic write failure kept the previous checkpoint; concurrent writer lock verified.\n\n"
              "CPU-only verification; GPU performance and CUDA recovery still require server verification.\n")
    (base / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
