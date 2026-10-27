"""分段读出的轴、旧模型逐位兼容和恢复边界检查。"""

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import CONFIG, PROJECT_ROOT, Config
from data import PreparedData, file_sha256
from experiments.variants import build_model, joint_loss
from model import SpikeTFNet, TemporalBranch, joint_loss as full_loss
from train import seed_everything
from utils.provenance import READOUT_V13_SOURCES, SOURCE_NAMES, validate_resume, verified_source_upgrade


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
