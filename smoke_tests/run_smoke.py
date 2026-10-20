"""仅在本地执行：数值检查 + 真实 Benchmark 小子集的 2 折×2 种子。"""

import io
import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / "smoke_tests" / "results" / "plot_cache")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import CONFIG
from data import channel_indices, prepare_data
from evaluate import reevaluate_experiment
from train import run_experiment


def main():
    results = ROOT / "smoke_tests" / "results"
    results.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "smoke_tests"), pattern="test_*.py")
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (results / "checks.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(stream.getvalue(), flush=True)
    if not outcome.wasSuccessful():
        raise SystemExit("Core checks failed; see smoke_tests/results/checks.txt")

    cfg = replace(CONFIG, subject_ids=(1, 2, 3), N=2, M=2, windows_ms=(500,), epochs=2,
                  batch_size=8, device="cpu", num_workers=0, cache_root=results / "cache",
                  output_root=results, experiment_name="integration")
    assert channel_indices(cfg) == [60, 61, 62, 54, 55, 56, 47, 53, 57]
    prepared = prepare_data(cfg)
    assert prepared.time.shape == (720, 9, 125)
    assert prepared.frequencies[:9].tolist() == [8., 9., 10., 11., 12., 13., 14., 15., 8.2]
    # 仅缩小样本和训练轮数；保留完整 40 类输出、560 频谱维和默认网络结构。
    rows = np.flatnonzero((prepared.ids[:, 1] == 1) & (prepared.ids[:, 2] % 5 == 0))
    selected = prepared.subset(rows)
    root = run_experiment(cfg, selected, purpose="smoke")
    summary_before = (root / "summary.csv").read_bytes()
    predictions_before = {}
    for path in root.glob("window_*/test_*/seed_*/details/evaluation.npz"):
        with np.load(path, allow_pickle=False) as archive:
            predictions_before[path] = archive["logits"].copy()
            assert not set(archive["training_ids"][:, 0]) & set(archive["test_ids"][:, 0])
            assert len(archive["test_ids"]) == 8
    assert len(predictions_before) == 4
    reevaluate_experiment(root, prepared=selected)
    assert (root / "summary.csv").read_bytes() == summary_before
    for path, before in predictions_before.items():
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["logits"], before)
    manifest = json.loads((root / "details" / "experiment.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    report = (f"# Smoke test: PASS\n\nCore checks: {outcome.testsRun} passed.\n\n"
              "Real data: S1-S3, 720 trials fully preprocessed; 8 trials per subject used for integration.\n"
              "Integration: 2 held-out subjects x 2 seeds (42, 43), 2 epochs, 0.5 s window, CPU.\n"
              "Default model dimensions and Ts=4 retained; all six window lengths checked separately.\n"
              "All four final checkpoints replayed with exactly identical logits and summary.\n"
              "Only source subjects entered training; eval BatchNorm buffers remained unchanged.\n\n"
              "These scores are pipeline checks, not scientific reproduction results.\n"
              "Detailed checks: checks.txt. Integration outputs: integration/.\n")
    (results / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
