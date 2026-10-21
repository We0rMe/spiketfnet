"""监测状态隔离和旧版本升级边界。"""

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from config import CONFIG, PROJECT_ROOT
from data import PreparedData, file_sha256
from evaluate import write_json
from model import SpikeTFNet
from train import monitor_training, rng_state, run_experiment
from utils.provenance import verified_reporting_upgrade


class ReportingTests(unittest.TestCase):
    def test_missing_checkpoint_blocks_without_restarting(self):
        results = PROJECT_ROOT / "smoke_tests/results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temporary:
            cfg = replace(CONFIG, subject_ids=(1, 2), N=1, M=1, windows_ms=(500,), device="cpu",
                          output_root=Path(temporary), experiment_name="missing", resume=True)
            prepared = PreparedData(np.zeros((2, 9, 125), dtype=np.float32),
                                    np.array([[1, 1, 0], [2, 1, 0]], dtype=np.int16),
                                    np.arange(40), np.arange(40), {"fingerprint": "test"})
            root = cfg.output_root / cfg.experiment_name
            hashes = {name: file_sha256(PROJECT_ROOT / name)
                      for name in ("train.py", "evaluate.py", "model.py", "data.py")}
            write_json(root / "details/experiment.json",
                       {"format_version": 3, "config": cfg.to_dict(), "purpose": "smoke",
                        "dataset_fingerprint": "test", "selected_trial_count": 2, "code_sha256": hashes})
            private = root / "window_0500ms/test_S01" / f"seed_{cfg.seed}" / "details"
            private.mkdir(parents=True)
            (private / "history.csv").write_text("epoch,loss,train_acc_percent,train_seconds\n1,1.0,10.0,1.0\n")
            with self.assertRaisesRegex(FileNotFoundError, "refusing to restart"):
                run_experiment(cfg, prepared, purpose="smoke")
            self.assertFalse((private / "resume.pt").exists())

    def test_monitor_preserves_parameters_buffers_and_rng(self):
        cfg = replace(CONFIG, device="cpu", N=1, windows_ms=(500,), batch_size=2)
        model = SpikeTFNet(cfg).train()
        dataset = TensorDataset(torch.randn(4, 9, 125), torch.randn(4, 9, 560), torch.arange(4))
        generator = torch.Generator().manual_seed(19)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        random_before = rng_state(generator, torch.device("cpu"))
        output = monitor_training(model, dataset, np.array([0, 2]), cfg, torch.device("cpu"), generator)
        self.assertTrue(model.training)
        self.assertTrue(np.isfinite(output["monitor_loss"]))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        random_after = rng_state(generator, torch.device("cpu"))
        for name in ("python", "numpy"):
            self.assertEqual(random_before[name], random_after[name])
        for name in ("torch", "loader"):
            torch.testing.assert_close(random_before[name], random_after[name], rtol=0, atol=0)

    def test_upgrade_requires_verified_archive_and_known_version(self):
        results = PROJECT_ROOT / "smoke_tests/results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temporary:
            details = Path(temporary)
            source = details / "source"
            source.mkdir()
            fixture = PROJECT_ROOT / "smoke_tests/fixtures/train_v2.py"
            for ending in (b"\n", b"\r\n"):
                raw = fixture.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending)
                (source / "train.py").write_bytes(raw)
                self.assertTrue(verified_reporting_upgrade("train.py", hashlib.sha256(raw).hexdigest(), details))
                self.assertFalse(verified_reporting_upgrade("train.py", "wrong hash", details))
            (source / "train.py").write_bytes(raw + b"\n# unknown edit\n")
            self.assertFalse(verified_reporting_upgrade("train.py", file_sha256(source / "train.py"), details))
            self.assertFalse(verified_reporting_upgrade("model.py", file_sha256(source / "train.py"), details))
