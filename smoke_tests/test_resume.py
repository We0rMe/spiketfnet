"""恢复验证中的兼容性、锁及原子写入检查。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from config import CONFIG, PROJECT_ROOT
from data import PreparedData
from train import atomic_torch_save, experiment_lock, restore_rng, rng_state, run_experiment, validate_resume


class ResumeTests(unittest.TestCase):
    def test_preparation_failure_can_be_retried(self):
        folder = PROJECT_ROOT / "smoke_tests" / "results"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            cfg = replace(CONFIG, device="cpu", resume=True, output_root=Path(temp), experiment_name="retry")
            with patch("train.prepare_data", side_effect=FileNotFoundError("Dataset temporarily unavailable")):
                for _ in range(2):
                    with self.assertRaises(FileNotFoundError):
                        run_experiment(cfg, purpose="smoke")

    def test_rng_roundtrip(self):
        import random
        generator = torch.Generator().manual_seed(10)
        state = rng_state(generator, torch.device("cpu"))
        expected = (random.random(), np.random.rand(), torch.rand(3), torch.randperm(8, generator=generator))
        restore_rng(state, generator, torch.device("cpu"))
        actual = (random.random(), np.random.rand(), torch.rand(3), torch.randperm(8, generator=generator))
        self.assertEqual(actual[:2], expected[:2])
        for a, b in zip(actual[2:], expected[2:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_resume_rejects_scientific_changes_but_allows_scope_and_device(self):
        cfg = replace(CONFIG, N=2, M=3, device="cpu")
        prepared = PreparedData(np.zeros((2, 9, 250)), np.array([[1, 1, 0], [2, 1, 0]], dtype=np.int16),
                                np.arange(40), np.arange(40), {"fingerprint": "example"})
        hashes = {name: name for name in ("model.py", "data.py", "train.py", "evaluate.py")}
        manifest = {"format_version": 2, "config": cfg.to_dict(), "purpose": "smoke",
                    "dataset_fingerprint": "example", "selected_trial_count": 2, "code_sha256": hashes}
        validate_resume(manifest, replace(cfg, M=1, N=1, device="cuda"), prepared, "smoke", hashes)
        for changes in ({"learning_rate": .02}, {"batch_size": 64}, {"epochs": 90}, {"simulation_steps": 2}):
            with self.assertRaises(ValueError):
                validate_resume(manifest, replace(cfg, **changes), prepared, "smoke", hashes)
        with self.assertRaises(ValueError):
            validate_resume(manifest, cfg, prepared, "formal", hashes)
        with self.assertRaises(ValueError):
            validate_resume(manifest, cfg, prepared, "smoke", {**hashes, "model.py": "changed"})

    def test_atomic_checkpoint_failure_keeps_previous(self):
        folder = PROJECT_ROOT / "smoke_tests" / "results"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            destination = Path(temp) / "resume.pt"
            atomic_torch_save({"epoch": 1}, destination)
            with patch("train.torch.save", side_effect=OSError("disk write failed")):
                with self.assertRaises(OSError):
                    atomic_torch_save({"epoch": 2}, destination)
            self.assertEqual(torch.load(destination, weights_only=True)["epoch"], 1)

    def test_experiment_lock_excludes_second_writer_and_releases(self):
        folder = PROJECT_ROOT / "smoke_tests" / "results"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            with experiment_lock(Path(temp)):
                with self.assertRaises(RuntimeError):
                    with experiment_lock(Path(temp)):
                        pass
            with experiment_lock(Path(temp)):
                pass
