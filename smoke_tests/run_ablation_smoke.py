"""真实 Benchmark 小样本：四组恢复、双进程调度及 v3 旧断点迁移。"""

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

from ablation import run_ablation
from config import CONFIG
from data import prepare_data
from evaluate import read_history, reevaluate_experiment, run_path
import train
from smoke_tests.assertions import metric_rows

MODEL_VARIANTS = ("full", "spectral_only", "temporal_only", "equal_fusion")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_runs(expected, actual, cfg):
    assert metric_rows(expected / "summary.csv") == metric_rows(actual / "summary.csv")
    for subject in cfg.test_subjects:
        a, b = [run_path(root, 500, subject, cfg.seed) for root in (expected, actual)]
        weights = [torch.load(folder / "details/model.pt", weights_only=True, map_location="cpu") for folder in (a, b)]
        assert weights[0].keys() == weights[1].keys()
        for key in weights[0]:
            torch.testing.assert_close(weights[0][key], weights[1][key], rtol=0, atol=0)
        assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()
        for old, new in zip(read_history(a / "details/history.csv"), read_history(b / "details/history.csv")):
            for key in ("epoch", "loss", "train_acc_percent", "monitor_loss", "monitor_acc_percent"):
                assert old[key] == new[key], (key, old, new)
        with np.load(a / "details/evaluation.npz") as x, np.load(b / "details/evaluation.npz") as y:
            assert set(x.files) == set(y.files)
            for key in x.files:
                np.testing.assert_array_equal(x[key], y[key])
            assert not set(y["training_ids"][:, 0]) & set(y["test_ids"][:, 0])
            assert set(y["test_ids"][:, 0]) == {subject}
        for artifact in ("training.log", "training.png", "confusion.png", "report.md"):
            assert (b / artifact).is_file()
        assert not (b / "details/resume.pt").exists()


def main(output_name="ablation_check"):
    base = ROOT / "smoke_tests/results" / output_name
    base.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(base / "plot_cache")
    stream = io.StringIO()
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.discover(str(ROOT / "smoke_tests"), pattern="test_*.py"))
    (base / "checks.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(stream.getvalue(), flush=True)
    if not outcome.wasSuccessful():
        raise AssertionError("Unit checks failed")
    cfg = replace(CONFIG, data_root=Path(r"G:\SSVEP\data\benchmark"), subject_ids=(1, 2, 3), N=2, M=1,
                  seed=42, windows_ms=(500,), epochs=2, batch_size=8, device="cpu", evaluation_device="cpu",
                  num_workers=0, cpu_threads=2, monitor_every=1, plot_every=1, resume=True,
                  output_root=base / "reference", cache_root=base.parent / "cache", model_variant="full")
    prepared = prepare_data(cfg)
    selected = prepared.subset(np.flatnonzero((prepared.ids[:, 1] == 1) & (prepared.ids[:, 2] % 5 == 0)))
    run_ablation(cfg, MODEL_VARIANTS, prepared=selected, purpose="smoke")
    resumed = replace(cfg, output_root=base / "resumed")
    save = train.save_resume

    def interrupt(path, model, optimizer, generator, device, history, job, data):
        save(path, model, optimizer, generator, device, history, job, data)
        if len(history) == 1:
            raise KeyboardInterrupt("Injected after first committed epoch")

    for variant in MODEL_VARIANTS:
        group = replace(resumed, model_variant=variant, experiment_name=variant)
        with patch("train.save_resume", side_effect=interrupt):
            try:
                train.run_experiment(group, selected, purpose="smoke")
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Interruption was not injected")
        state = torch.load(run_path(resumed.output_root / variant, 500, 1, cfg.seed) / "details/resume.pt",
                           weights_only=True, map_location="cpu")
        assert state["epoch"] == 1 and state["model_variant"] == variant
        del state
    with patch("train.training_devices", return_value=[torch.device("cpu"), torch.device("cpu")]):
        run_ablation(resumed, MODEL_VARIANTS, prepared=selected, purpose="smoke")
    for variant in MODEL_VARIANTS:
        compare_runs(cfg.output_root / variant, resumed.output_root / variant, cfg)
        reevaluate_experiment(resumed.output_root / variant, selected, device_name="cpu")
        compare_runs(cfg.output_root / variant, resumed.output_root / variant, cfg)
    assert metric_rows(cfg.output_root / "comparison.csv") == metric_rows(resumed.output_root / "comparison.csv")
    with patch("train.train_epoch", side_effect=AssertionError("Completed task trained again")):
        run_ablation(resumed, MODEL_VARIANTS, prepared=selected, purpose="smoke")

    # 从冻结的旧训练/评估代码产生真实 v3 checkpoint；删除新配置字段以模拟旧快照。
    source = base / "v3_source"
    source.mkdir()
    for name in ("config.py", "model.py", "data.py", "requirements.txt"):
        shutil.copy2(ROOT / name, source / name)
    for name in ("train.py", "evaluate.py"):
        shutil.copy2(ROOT / "smoke_tests/fixtures" / name.replace(".py", "_v3.py"), source / name)
    old_eval = load_module("ablation_smoke_evaluate_v3", source / "evaluate.py")
    with patch.dict(sys.modules, {"evaluate": old_eval}):
        old = load_module("ablation_smoke_train_v3", source / "train.py")
    old.PROJECT_ROOT = source
    legacy_cfg = replace(cfg, output_root=base, experiment_name="legacy_full")
    original_save = old.save_resume

    def interrupt_old(*args):
        original_save(*args)
        if len(args[5]) == 1:
            raise KeyboardInterrupt("Injected v3 checkpoint interruption")

    with patch.object(old, "save_resume", side_effect=interrupt_old):
        try:
            old.run_experiment(legacy_cfg, selected, purpose="smoke")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("Legacy interruption was not injected")
    manifest_file = base / "legacy_full/details/experiment.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    del manifest["config"]["model_variant"]
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    train.run_experiment(legacy_cfg, selected, purpose="smoke")
    compare_runs(cfg.output_root / "full", base / "legacy_full", cfg)

    report = (f"# Ablation smoke: PASS\n\n{outcome.testsRun} unit checks passed.\n\n"
              "Real Benchmark S1-S3, 8 trials/subject, 2 LOSO folds, seed 42, 500 ms, 2 epochs.\n"
              "Original model widths/depths and Ts=4 retained. This subset is only a smoke test.\n\n"
              "All four variants: uninterrupted vs epoch-1 interruption + two CPU worker resume matched exactly:\n"
              "final weights, loss/ACC histories, evaluation evidence, metrics and metric summaries (runtime excluded).\n"
              "Independent reevaluation and completed-task skipping passed for every variant.\n"
              "Frozen v3 checkpoint resumed as full and matched uninterrupted full training exactly.\n"
              "Same initial shared weights; full vs constant-gate equal-fusion matched even with dropout ON.\n"
              "Single branches kept projection/classifier, absent inputs had no effect; no held-out subject in training.\n"
              "500/1000 ms model checks passed. CPU verification only; CUDA execution was not tested locally.\n")
    (base / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()

