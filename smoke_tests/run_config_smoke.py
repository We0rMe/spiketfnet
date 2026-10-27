"""真实小样本验证新候选的训练、断点恢复及重评估，不测量模型效果。"""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config import CONFIG
from data import prepare_data
from evaluate import reevaluate_experiment, run_path
import train


def main():
    base = ROOT / "smoke_tests/results/config_check" / CONFIG.experiment_name
    base.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    with redirect_stdout(stream):
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.discover(str(ROOT / "smoke_tests"), pattern="test_*.py"))
    if not tests.wasSuccessful():
        (base / "checks.txt").write_text(stream.getvalue(), encoding="utf-8")
        raise AssertionError(stream.getvalue())
    cfg = replace(CONFIG, data_root=Path(r"G:\SSVEP\data\benchmark"), subject_ids=(1, 2, 3),
                  N=1, M=1, epochs=2, batch_size=8, cpu_threads=2, device="cpu", evaluation_device="cpu",
                  monitor_every=1, model_variant="temporal_only", experiment_name="reference",
                  output_root=base / "details/runs", cache_root=base / "details/cache")
    with redirect_stdout(stream):
        prepared = prepare_data(cfg)
        selected = prepared.subset(np.flatnonzero((prepared.ids[:, 1] == 1) & (prepared.ids[:, 2] % 5 == 0)))
        reference = train.run_experiment(cfg, selected, purpose="smoke")
        resumed_cfg = replace(cfg, experiment_name="resumed")
        original_save = train.save_resume

        def interrupt(path, model, optimizer, generator, device, history, job, data):
            original_save(path, model, optimizer, generator, device, history, job, data)
            if len(history) == 1:
                raise KeyboardInterrupt("Smoke: stop after committed epoch 1")

        with patch("train.save_resume", side_effect=interrupt):
            try:
                train.run_experiment(resumed_cfg, selected, purpose="smoke")
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Interruption was not triggered")
        resumed = train.run_experiment(resumed_cfg, selected, purpose="smoke")
        folders = [run_path(root, cfg.windows_ms[0], 1, cfg.seed) for root in (reference, resumed)]
        weights = [torch.load(p / "details/model.pt", weights_only=True) for p in folders]
        for key, value in weights[0].items():
            torch.testing.assert_close(value, weights[1][key], rtol=0, atol=0)
        with np.load(folders[0] / "details/evaluation.npz") as a, np.load(folders[1] / "details/evaluation.npz") as b:
            saved = {key: b[key].copy() for key in b.files}
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            assert not set(a["training_ids"][:, 0]) & set(a["test_ids"][:, 0])
        reevaluate_experiment(resumed, prepared=selected, device_name="cpu")
        with np.load(folders[1] / "details/evaluation.npz") as archive:
            for key, value in saved.items():
                np.testing.assert_array_equal(archive[key], value)
    report = {"unit_tests_passed": tests.testsRun, "device": "cpu", "data": "S1-S3, block1, 8 trials/subject",
              "training_epochs": 2, "window_ms": cfg.windows_ms[0], "resume_weights_and_predictions_exact": True,
              "reevaluation_exact": True, "loso_subjects_disjoint": True,
              "candidate": {key: getattr(CONFIG, key) for key in (
                  "temporal_dropout", "dropout", "stem_kernel", "residual_kernel", "t2_conv_execution",
                  "temporal_readout_bins", "simulation_steps")},
              "scope": "software verification only; no evidence of accuracy improvement"}
    manifest = json.loads((reference / "details/experiment.json").read_text(encoding="utf-8"))
    report["code_sha256"] = manifest["code_sha256"]
    (base / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
