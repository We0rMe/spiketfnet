"""简化配置、历史快照和分支独立 dropout 的回归检查。"""

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import ABLATION_CONFIG, ABLATION_GROUPS, CONFIG, PROJECT_ROOT, Config
from data import PreparedData
from model import SpikeTFNet, joint_loss
from train import seed_everything
from utils.provenance import CONFIG_V12_SOURCES, SOURCE_NAMES, validate_resume, verified_source_upgrade


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_candidate_scope_and_snapshot_roundtrip(self):
        CONFIG.validate()
        self.assertEqual((CONFIG.N, CONFIG.M, CONFIG.seeds), (3, 2, (42, 43)))
        self.assertEqual(ABLATION_GROUPS, ("temporal_only",))
        self.assertEqual(ABLATION_CONFIG.windows_ms, (500,))
        self.assertEqual(CONFIG.temporal_readout_bins, 32)
        self.assertEqual(len(CONFIG.subject_ids), 35)
        self.assertEqual(Config.from_dict(json.loads(json.dumps(CONFIG.to_dict()))), CONFIG)
        self.assertIn("utils/configuration.py", SOURCE_NAMES)

    def test_legacy_missing_fields_never_inherit_candidate_defaults(self):
        snapshot = Config(dropout=0.2).to_dict()
        del snapshot["temporal_dropout"], snapshot["t2_conv_execution"]
        restored = Config.from_dict(snapshot)
        self.assertEqual(restored.temporal_backbone_dropout, 0.2)
        self.assertEqual(restored.t2_conv_execution, "native")
        for value in (-0.1, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                replace(CONFIG, temporal_dropout=value).validate()

    def test_dropout_is_separated_only_in_temporal_backbone(self):
        model = SpikeTFNet(CONFIG)
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                is_backbone = name == "temporal.dropout" or name.startswith("temporal.blocks.")
                self.assertEqual(module.p, 0.1 if is_backbone else 0.5, name)

    def test_original_dropout_path_matches_v12_forward_backward_and_buffers(self):
        spec = importlib.util.spec_from_file_location("v12_model", PROJECT_ROOT / "smoke_tests/fixtures/v12/model.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for execution in ("native", "split_bias"):
            cfg = Config(device="cpu", t2_conv_execution=execution)
            seed_everything(42, cfg)
            reference = old.SpikeTFNet(cfg)
            seed_everything(42, cfg)
            current = SpikeTFNet(cfg)
            time, spectrum = torch.randn(2, 9, 125), torch.randn(2, 9, 560)
            labels = torch.tensor([1, 3])
            for training in (True, False):
                results = []
                for net in (reference, current):
                    net.train(training)
                    net.zero_grad(set_to_none=True)
                    seed_everything(90, cfg)
                    result = net(time, spectrum)
                    joint_loss(result, labels, cfg).backward()
                    results.append(result)
                for key in results[0]:
                    torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
                for (_, a), (_, b) in zip(reference.named_parameters(), current.named_parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_new_schema_and_dropout_are_guarded_on_resume(self):
        cfg = replace(CONFIG, device="cpu")
        prepared = PreparedData(np.zeros((2, 9, 250)), np.array([[1, 1, 0], [2, 1, 0]]),
                                np.arange(40), np.arange(40), {"fingerprint": "example"})
        hashes = {name: name for name in SOURCE_NAMES}
        manifest = {"format_version": 13, "config": cfg.to_dict(), "purpose": "smoke",
                    "dataset_fingerprint": "example", "selected_trial_count": 2, "code_sha256": hashes}
        validate_resume(manifest, cfg, prepared, "smoke", hashes)
        with self.assertRaisesRegex(ValueError, "temporal_dropout"):
            validate_resume(manifest, replace(cfg, temporal_dropout=0.5), prepared, "smoke", hashes)
        with self.assertRaisesRegex(ValueError, "utils/configuration.py"):
            validate_resume(manifest, cfg, prepared, "smoke", {**hashes, "utils/configuration.py": "changed"})

    def test_v12_model_upgrade_requires_authentic_source(self):
        folder = PROJECT_ROOT / "smoke_tests/results"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            details = Path(temp)
            path = details / "source/model.py"
            path.parent.mkdir()
            path.write_bytes((PROJECT_ROOT / "smoke_tests/fixtures/v12/model.py").read_bytes())
            from data import file_sha256
            self.assertTrue(verified_source_upgrade("model.py", file_sha256(path), details, CONFIG_V12_SOURCES))
            path.write_bytes(path.read_bytes() + b"\n# unknown modification\n")
            self.assertFalse(verified_source_upgrade("model.py", file_sha256(path), details, CONFIG_V12_SOURCES))


if __name__ == "__main__":
    unittest.main()
