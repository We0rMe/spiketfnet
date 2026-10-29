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
        self.assertEqual(CONFIG.seeds, tuple(range(CONFIG.seed, CONFIG.seed + CONFIG.M)))
        self.assertEqual(CONFIG.test_subjects, tuple(range(1, CONFIG.N + 1)))
        self.assertEqual(ABLATION_GROUPS, ("equal_supervised",))
        self.assertEqual(CONFIG.lambda_fusion_branch, 0.3)
        self.assertEqual(ABLATION_CONFIG.windows_ms, (500,))
        self.assertEqual(CONFIG.temporal_readout_bins, 32)
        self.assertEqual(len(CONFIG.subject_ids), 35)
        self.assertEqual(Config.from_dict(json.loads(json.dumps(CONFIG.to_dict()))), CONFIG)
        self.assertIn("utils/configuration.py", SOURCE_NAMES)

    def test_legacy_missing_fields_never_inherit_candidate_defaults(self):
        snapshot = Config(dropout=0.2).to_dict()
        del snapshot["temporal_dropout"], snapshot["t2_conv_execution"], snapshot["lambda_fusion_branch"]
        restored = Config.from_dict(snapshot)
        self.assertEqual(restored.temporal_backbone_dropout, 0.2)
        self.assertEqual(restored.t2_conv_execution, "native")
        self.assertEqual(restored.lambda_fusion_branch, 0.0)
        for value in (-0.1, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "lambda_fusion_branch"):
                replace(CONFIG, lambda_fusion_branch=value).validate()
        for value in (-0.1, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                replace(CONFIG, temporal_dropout=value).validate()

    def test_dropout_is_separated_only_in_temporal_backbone(self):
        model = SpikeTFNet(CONFIG)
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                is_backbone = name == "temporal.dropout" or name.startswith("temporal.blocks.")
                self.assertEqual(module.p, 0.1 if is_backbone else 0.5, name)

    def test_v12_weights_cannot_load_into_time_position_attention(self):
        spec = importlib.util.spec_from_file_location("v12_model", PROJECT_ROOT / "smoke_tests/fixtures/v12/model.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        cfg = replace(CONFIG, device="cpu", windows_ms=(500,))
        reference = old.SpikeTFNet(cfg)
        current = SpikeTFNet(cfg)
        with self.assertRaises(RuntimeError):
            current.load_state_dict(reference.state_dict(), strict=True)

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
