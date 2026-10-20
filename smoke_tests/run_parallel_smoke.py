"""用两个真实 CPU spawn 进程验证调度、隔离及串行一致性，不代替 GPU 验证。"""

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
from evaluate import reevaluate_experiment
from train import run_experiment


def main():
    base = ROOT / "smoke_tests" / "results" / "parallel_device_check"
    base.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(base / "plot_cache")
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "smoke_tests"), pattern="test_*.py")
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (base / "checks.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(stream.getvalue(), flush=True)
    if not outcome.wasSuccessful():
        raise SystemExit("Device/core checks failed")

    cfg = replace(CONFIG, data_root=Path(r"G:\SSVEP\data\benchmark"), subject_ids=(1, 2, 3),
                  N=2, M=2, seed=42, windows_ms=(500,), epochs=2, batch_size=8,
                  device="cpu", evaluation_device="cpu", num_workers=0, cpu_threads=2,
                  cache_root=base.parent / "cache", output_root=base, experiment_name="serial")
    prepared = prepare_data(cfg)
    rows = np.flatnonzero((prepared.ids[:, 1] == 1) & (prepared.ids[:, 2] % 5 == 0))
    selected = prepared.subset(rows)
    serial = run_experiment(cfg, selected, purpose="smoke")
    # 只替换设备发现；spawn、任务分配、模型训练与文件写入均走生产路径。
    with patch("train.training_devices", return_value=[torch.device("cpu"), torch.device("cpu")]):
        parallel = run_experiment(replace(cfg, experiment_name="parallel"), selected, purpose="smoke")
    assert (serial / "summary.csv").read_bytes() == (parallel / "summary.csv").read_bytes()
    pids = set()
    runs = sorted(parallel.glob("window_*/test_*/seed_*"))
    assert len(runs) == 4
    for folder in runs:
        reference = serial / folder.relative_to(parallel)
        provenance = json.loads((folder / "details" / "provenance.json").read_text(encoding="utf-8"))
        pids.add(provenance["worker_pid"])
        assert provenance["status"] == "complete"
        assert provenance["test_subject"] not in provenance["training_subjects"]
        for name in ("metrics.json", "details/history.csv"):
            assert (folder / name).read_bytes() == (reference / name).read_bytes()
        with np.load(folder / "details" / "evaluation.npz") as actual, np.load(reference / "details" / "evaluation.npz") as expected:
            assert set(actual.files) == set(expected.files)
            for name in actual.files:
                np.testing.assert_array_equal(actual[name], expected[name])
        weights = torch.load(folder / "details" / "model.pt", map_location="cpu", weights_only=True)
        expected_weights = torch.load(reference / "details" / "model.pt", map_location="cpu", weights_only=True)
        assert weights.keys() == expected_weights.keys()
        for name in weights:
            torch.testing.assert_close(weights[name], expected_weights[name], rtol=0, atol=0)
    assert len(pids) == 2 and os.getpid() not in pids
    reevaluate_experiment(parallel, prepared=selected, device_name="cpu")
    assert (serial / "summary.csv").read_bytes() == (parallel / "summary.csv").read_bytes()
    for folder in runs:
        with np.load(folder / "details" / "evaluation.npz") as actual, np.load(serial / folder.relative_to(parallel) / "details" / "evaluation.npz") as expected:
            np.testing.assert_array_equal(actual["logits"], expected["logits"])
    report = (f"# Parallel smoke: PASS\n\n{outcome.testsRun} core/device checks passed.\n"
              "Real Benchmark S1-S3: 720 trials preprocessed, 8 per subject selected.\n"
              "Serial and two spawned CPU workers each completed 2 folds x 2 seeds x 2 epochs.\n"
              "Separate worker PIDs confirmed; held-out subjects never entered training.\n"
              "All weights, histories, evaluation arrays, metrics and summaries matched exactly.\n"
              "All four parallel checkpoints replayed with identical logits and summary.\n\n"
              "GPU selection was mocked; real CUDA execution and dual-GPU speed were not tested.\n"
              "Scores are software checks, not reproduction results.\n")
    (base / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
