"""由冻结的 v2 训练代码生成旧断点，验证输出升级后仍保持训练轨迹。"""

import csv
import importlib.util
import io
import json
import os
import shutil
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
from evaluate import run_path, read_history
import train


def load_v2(source):
    spec = importlib.util.spec_from_file_location("spiketfnet_train_v2", source / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = source
    return module


def main(output_name="reporting_check"):
    base = ROOT / "smoke_tests/results" / output_name
    base.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(base / "plot_cache")
    source = base / "v2_source"
    source.mkdir()
    for name in ("config.py", "data.py", "model.py", "requirements.txt"):
        shutil.copy2(ROOT / name, source / name)
    for name in ("train.py", "evaluate.py"):
        shutil.copy2(ROOT / "smoke_tests/fixtures" / name.replace(".py", "_v2.py"), source / name)
    old = load_v2(source)
    cfg = replace(CONFIG, data_root=Path(r"G:\SSVEP\data\benchmark"), subject_ids=(1, 2, 3),
                  N=2, M=1, seed=42, windows_ms=(500,), epochs=3, batch_size=8,
                  device="cpu", evaluation_device="cpu", resume=True, num_workers=0, cpu_threads=2,
                  monitor_every=1, plot_every=1, cache_root=base.parent / "cache",
                  output_root=base, experiment_name="old_reference")
    prepared = prepare_data(cfg)
    selected = prepared.subset(np.flatnonzero((prepared.ids[:, 1] == 1) & (prepared.ids[:, 2] % 5 == 0)))
    reference = old.run_experiment(cfg, selected, purpose="smoke")
    updated_cfg = replace(cfg, experiment_name="upgraded")
    original_save = old.save_resume

    def interrupt_old(path, model, optimizer, generator, device, history, job, data):
        original_save(path, model, optimizer, generator, device, history, job, data)
        if len(history) == 1:
            raise KeyboardInterrupt("Injected old-version interruption after epoch 1")

    with patch.object(old, "save_resume", side_effect=interrupt_old):
        try:
            old.run_experiment(updated_cfg, selected, purpose="smoke")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("Old-version interruption not injected")
    upgraded = base / "upgraded"
    private = run_path(upgraded, 500, 1, 42) / "details"
    state = torch.load(private / "resume.pt", map_location="cpu", weights_only=True)
    assert state["epoch"] == 1 and "monitor_loss" not in state["history"][0]
    # 旧保存点成功、CSV 尚未同步的情形，仍应使用 checkpoint 中的历史。
    original_save_new = train.save_resume

    def interrupt_new(path, model, optimizer, generator, device, history, job, data):
        original_save_new(path, model, optimizer, generator, device, history, job, data)
        if len(history) == 3:
            prior = read_history(Path(path).parent / "history.csv")
            assert prior[-1]["epoch"] == 2
            assert (Path(path).parent.parent / "training.png").is_file()
            assert "epoch=2/3" in (Path(path).parent.parent / "training.log").read_text(encoding="utf-8")
            raise KeyboardInterrupt("Injected after live epoch 2 artifacts and committed epoch 3")

    with patch("train.save_resume", side_effect=interrupt_new):
        try:
            train.run_experiment(updated_cfg, selected, purpose="smoke")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("New-version interruption not injected")
    assert json.loads((upgraded / "details/experiment.json").read_text(encoding="utf-8"))["format_version"] == 3
    with patch("train.training_devices", return_value=[torch.device("cpu"), torch.device("cpu")]):
        train.run_experiment(updated_cfg, selected, purpose="smoke")
    assert (reference / "summary.csv").read_bytes() == (upgraded / "summary.csv").read_bytes()
    for subject in (1, 2):
        a, b = (run_path(root, 500, subject, 42) for root in (reference, upgraded))
        for name in ("training.png", "confusion.png", "report.md", "training.log"):
            assert (b / name).is_file()
        weights_a, weights_b = [torch.load(folder / "details/model.pt", map_location="cpu", weights_only=True)
                                for folder in (a, b)]
        for name in weights_a:
            torch.testing.assert_close(weights_a[name], weights_b[name], rtol=0, atol=0)
        for row_a, row_b in zip(read_history(a / "details/history.csv"), read_history(b / "details/history.csv")):
            for name in ("epoch", "loss", "train_acc_percent"):
                assert row_a[name] == row_b[name]
        with np.load(a / "details/evaluation.npz") as arrays_a, np.load(b / "details/evaluation.npz") as arrays_b:
            for key in arrays_a.files:
                np.testing.assert_array_equal(arrays_a[key], arrays_b[key])
        assert not (b / "details/resume.pt").exists()
    first_history = read_history(run_path(upgraded, 500, 1, 42) / "details/history.csv")
    assert "RECOVERED epoch=1" in (run_path(upgraded, 500, 1, 42) / "training.log").read_text(encoding="utf-8")
    assert first_history[0]["monitor_loss"] is None
    assert all(row["monitor_loss"] is not None for row in first_history[1:])
    # 已完成旧任务升级只生成新展示文件，不做训练或再次测试。
    with patch("train.train_epoch", side_effect=AssertionError("Completed legacy task trained again")):
        train.run_experiment(cfg, selected, purpose="smoke")
    assert (run_path(reference, 500, 1, 42) / "confusion.png").is_file()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "smoke_tests"), pattern="test_*.py")
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (base / "checks.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(stream.getvalue(), flush=True)
    if not outcome.wasSuccessful():
        raise AssertionError("Unit checks failed")
    report = (f"# Reporting upgrade: PASS\n\n{outcome.testsRun} core/device/resume/reporting checks passed.\n"
              "Frozen v2 code trained uninterrupted references and produced a genuine epoch-1 checkpoint.\n"
              "New code resumed it, added monitoring/live artifacts, was interrupted again, then resumed with two CPU workers.\n"
              "All final weights, training metrics, prediction/evidence arrays and summary matched old uninterrupted training exactly.\n"
              "Live log/plot existed before completion; old monitor values remained missing, not fabricated.\n"
              "Completed old tasks gained reports/confusion/log reconstruction without retraining.\n"
              "Monitor uses source training block 1, never the held-out subject. It is NOT validation.\n\n"
              "CPU smoke verification only; real CUDA checkpoint migration must still be checked on the server.\n")
    (base / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
