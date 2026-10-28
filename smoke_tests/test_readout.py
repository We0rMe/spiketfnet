"""分段读出的轴、旧模型逐位兼容和恢复边界检查。"""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from config import CONFIG, PROJECT_ROOT, Config
from data import PreparedData, file_sha256
from experiments.variants import build_model, joint_loss
from model import SpikeTFNet, TemporalBranch, joint_loss as full_loss
from experiments.ablation import run_ablation
from evaluate import run_path
import train
from train import seed_everything
from utils.provenance import (POOLING_V14_SOURCES, READOUT_V13_SOURCES, SOURCE_NAMES,
                              pooling_upgrade_allowed, validate_resume, verified_source_upgrade)


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_default_and_invalid_bins(self):
        snapshot = CONFIG.to_dict()
        del snapshot["temporal_readout_bins"]
        self.assertEqual(Config.from_dict(snapshot).temporal_readout_bins, 1)
        for bins in (0, -1, 126, True, 1.5):
            with self.assertRaises(ValueError):
                replace(CONFIG, temporal_readout_bins=bins).validate()
        replace(CONFIG, temporal_readout_bins=125).validate()
        with self.assertRaises(ValueError):
            replace(CONFIG, windows_ms=(1000, 100), temporal_readout_bins=32).validate()

    def test_v13_gap_initialization_forward_backward_and_rng_are_exact(self):
        spec = importlib.util.spec_from_file_location("v13_model", PROJECT_ROOT / "smoke_tests/fixtures/v13/model.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        cfg = replace(CONFIG, temporal_readout_bins=1, device="cpu")
        seed_everything(42, cfg)
        reference = old.SpikeTFNet(cfg)
        initial_rng = torch.get_rng_state().clone()
        seed_everything(42, cfg)
        current = SpikeTFNet(cfg)
        torch.testing.assert_close(initial_rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(reference.state_dict().keys(), current.state_dict().keys())
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
        time, spectrum = torch.randn(2, 9, 125), torch.randn(2, 9, 560)
        for training in (True, False):
            results, rng = [], []
            for net in (reference, current):
                net.train(training)
                net.zero_grad(set_to_none=True)
                seed_everything(91, cfg)
                output = net(time, spectrum)
                full_loss(output, torch.tensor([0, 1]), cfg).backward()
                results.append(output)
                rng.append(torch.get_rng_state())
            for key in results[0]:
                torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
            for (name, a), (_, b) in zip(reference.named_parameters(), current.named_parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
            torch.testing.assert_close(rng[0], rng[1], rtol=0, atol=0)

    def test_pool_keeps_time_order_and_never_mixes_channels_or_trials(self):
        branch = TemporalBranch(CONFIG)
        features = torch.zeros(2, 64, 125, requires_grad=True)
        with torch.no_grad():
            features[0, 0, 0] = 1
            features[1, 0, -1] = 1
        pooled = branch.pool_features(features).reshape(2, 64, 32)
        torch.testing.assert_close(features[0].mean(-1), features[1].mean(-1), rtol=0, atol=0)
        self.assertFalse(torch.equal(pooled[0], pooled[1]))
        self.assertGreater(pooled[0, 0, 0].item(), 0)
        self.assertGreater(pooled[1, 0, -1].item(), 0)
        self.assertEqual(pooled[:, 1:].count_nonzero().item(), 0)
        torch.testing.assert_close(pooled[0], branch.pool_features(features[:1]).reshape(64, 32))
        pooled.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue((features.grad > 0).all())
        with self.assertRaisesRegex(ValueError, "upsample"):
            branch.pool_features(features[..., :31])

    def test_candidate_keeps_four_steps_fusion_and_gradients(self):
        cfg = replace(CONFIG, model_variant="temporal_only", device="cpu")
        seed_everything(42, cfg)
        model = build_model(cfg)
        self.assertTrue(hasattr(model.fusion, "temporal"))
        self.assertTrue(hasattr(model.fusion, "classifier"))
        shapes = []
        hook = model.temporal.embedding[0].register_forward_pre_hook(
            lambda module, inputs: shapes.append(tuple(inputs[0].shape)))
        try:
            result = model(torch.randn(2, 9, 125), torch.randn(2, 9, 560))
            joint_loss(result, torch.tensor([0, 1]), cfg).backward()
        finally:
            hook.remove()
        self.assertEqual(shapes, [(2, 64 * 32)] * 4)
        self.assertEqual(result["logits"].shape, (2, 40))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.temporal.embedding[0].weight.grad.abs().sum().item(), 0)
        baseline = build_model(replace(cfg, temporal_readout_bins=1))
        self.assertEqual(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()),
                         64 * 31 * 128)

    def test_segment_values_and_gradients_match_adaptive_pooling(self):
        for dtype in (torch.float32, torch.float64):
            tolerance = 1e-6 if dtype == torch.float32 else 1e-14
            for length, bins in ((125, 32), (250, 32), (128, 32), (125, 125), (125, 1)):
                branch = TemporalBranch(replace(CONFIG, temporal_readout_bins=bins))
                # 非连续输入也必须保持相同分段，特别检查125→32的重叠边界。
                features = torch.randn(2, 3, length * 2, dtype=dtype)[..., ::2].requires_grad_()
                expected = torch.nn.functional.adaptive_avg_pool1d(features, bins).flatten(1)
                actual = branch.pool_features(features)
                weights = torch.randn_like(expected)
                expected_grad = torch.autograd.grad(expected, features, weights)[0]
                actual_grad = torch.autograd.grad(actual, features, weights)[0]
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)

    def test_pooling_upgrade_only_accepts_untrained_checkpoints(self):
        folder = PROJECT_ROOT / "smoke_tests/results"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            root = Path(temp)
            details = root / "details"
            details.mkdir()
            private = root / "window_0500ms/test_S01/seed_42/details"
            private.mkdir(parents=True)
            (private / "history.csv").write_text("epoch,loss\n", encoding="utf-8")
            state = {"epoch": 0, "history": [], "optimizer": {"state": {}}}
            torch.save(state, private / "resume.pt")
            self.assertTrue(pooling_upgrade_allowed(CONFIG, details))
            for changes in ({"epoch": 1}, {"history": [{"epoch": 1}]}, {"optimizer": {"state": {0: {"step": 1}}}}):
                torch.save({**state, **changes}, private / "resume.pt")
                self.assertFalse(pooling_upgrade_allowed(CONFIG, details))
            torch.save(state, private / "resume.pt")
            (private / "history.csv").write_text("epoch,loss\n1,3.0\n", encoding="utf-8")
            self.assertFalse(pooling_upgrade_allowed(CONFIG, details))
            (private / "history.csv").write_text("epoch,loss\n", encoding="utf-8")
            for name in ("model.pt", "evaluation.npz", "metrics.json"):
                path = private / name
                path.touch()
                self.assertFalse(pooling_upgrade_allowed(CONFIG, details))
                path.unlink()
            self.assertFalse(pooling_upgrade_allowed(CONFIG, None))
            self.assertTrue(pooling_upgrade_allowed(replace(CONFIG, temporal_readout_bins=1), details))

    def test_v14_failed_first_batch_retries_through_ablation_entry(self):
        folder = PROJECT_ROOT / "smoke_tests/results"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg = replace(CONFIG, subject_ids=(1, 2), N=1, M=1, windows_ms=(500,),
                          epochs=1, batch_size=2, device="cpu", evaluation_device="cpu", cpu_threads=2,
                          output_root=root / "retry", experiment_name="pooling_test")
            ids = np.array([(s, 1, c) for s in (1, 2) for c in (0, 1)], dtype=np.int16)
            prepared = PreparedData(np.random.default_rng(1).normal(size=(4, 9, 125)).astype(np.float32),
                                    ids, np.arange(40), np.arange(40), {"fingerprint": "synthetic-pooling-retry"})
            with patch("train.train_epoch", side_effect=RuntimeError("adaptive_avg_pool2d_backward_cuda: deterministic")):
                with self.assertRaisesRegex(RuntimeError, "adaptive_avg_pool"):
                    run_ablation(cfg, ("temporal_only",), prepared, purpose="smoke")
            experiment = cfg.output_root / "temporal_only"
            details = experiment / "details"
            path = details / "experiment.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            old_source = details / "source/model.py"
            old_source.write_bytes((PROJECT_ROOT / "smoke_tests/fixtures/v14/model.py").read_bytes())
            old_hash = file_sha256(old_source)
            self.assertTrue(verified_source_upgrade("model.py", old_hash, details, POOLING_V14_SOURCES))
            manifest["code_sha256"]["model.py"] = old_hash
            manifest["format_version"] = 14
            path.write_text(json.dumps(manifest), encoding="utf-8")
            run_ablation(cfg, ("temporal_only",), prepared, purpose="smoke")
            upgraded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(upgraded["format_version"], 16)
            self.assertEqual(upgraded["status"], "complete")
            run_ablation(replace(cfg, output_root=root / "fresh"), ("temporal_only",), prepared, purpose="smoke")
            runs = [run_path(p / "temporal_only", 500, 1, cfg.seed) for p in (cfg.output_root, root / "fresh")]
            weights = [torch.load(p / "details/model.pt", weights_only=True) for p in runs]
            for name in weights[0]:
                torch.testing.assert_close(weights[0][name], weights[1][name], rtol=0, atol=0)
            with np.load(runs[0] / "details/evaluation.npz") as a, np.load(runs[1] / "details/evaluation.npz") as b:
                np.testing.assert_array_equal(a["logits"], b["logits"])
            # 已有结果的v14多段模型不允许悄悄改用新计算顺序。
            blocked = {**upgraded, "format_version": 14,
                       "code_sha256": {**upgraded["code_sha256"], "model.py": old_hash}}
            group_cfg = replace(cfg, model_variant="temporal_only", experiment_name="temporal_only")
            with self.assertRaisesRegex(ValueError, "saved training/results"):
                validate_resume(blocked, group_cfg, prepared, "smoke", upgraded["code_sha256"], details)

    def test_resume_requires_same_readout_and_authentic_legacy_source(self):
        cfg = replace(CONFIG, temporal_readout_bins=1, device="cpu")
        prepared = PreparedData(np.zeros((2, 9, 125)), np.array([[1, 1, 0], [2, 1, 0]]),
                                np.arange(40), np.arange(40), {"fingerprint": "example"})
        snapshot = cfg.to_dict()
        del snapshot["temporal_readout_bins"]
        hashes = {name: file_sha256(PROJECT_ROOT / name) for name in SOURCE_NAMES}
        folder = PROJECT_ROOT / "smoke_tests/results"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            details = Path(temp)
            path = details / "source/model.py"
            path.parent.mkdir()
            path.write_bytes((PROJECT_ROOT / "smoke_tests/fixtures/v13/model.py").read_bytes())
            old_hash = file_sha256(path)
            manifest = {"format_version": 13, "config": snapshot, "purpose": "smoke",
                        "dataset_fingerprint": "example", "selected_trial_count": 2,
                        "code_sha256": {**hashes, "model.py": old_hash}}
            self.assertTrue(verified_source_upgrade("model.py", old_hash, details, READOUT_V13_SOURCES))
            validate_resume(manifest, cfg, prepared, "smoke", hashes, details)
            with self.assertRaisesRegex(ValueError, "temporal_readout_bins"):
                validate_resume(manifest, replace(cfg, temporal_readout_bins=32), prepared, "smoke", hashes, details)
            path.write_bytes(path.read_bytes() + b"\n# unknown modification\n")
            with self.assertRaisesRegex(ValueError, "model.py"):
                validate_resume(manifest, cfg, prepared, "smoke", hashes, details)
            current = {**manifest, "format_version": 14, "config": CONFIG.to_dict(), "code_sha256": hashes}
            for name in ("model.py", "utils/configuration.py"):
                with self.assertRaisesRegex(ValueError, name):
                    validate_resume(current, CONFIG, prepared, "smoke", {**hashes, name: "changed"}, details)


if __name__ == "__main__":
    unittest.main()
