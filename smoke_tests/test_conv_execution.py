"""偏置外置的数值等价、参数兼容与 native 历史轨迹验证。"""

import copy
import importlib.util
import io
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import CONFIG, PROJECT_ROOT, Config
from data import PreparedData, file_sha256
from experiments.variants import build_model, joint_loss
from model import SplitBiasConv1d
from train import seed_everything
from utils.provenance import CONV_EXECUTION_V9_SOURCES, SOURCE_NAMES, validate_resume, verified_source_upgrade


class ConvExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_layer_output_and_all_gradients(self):
        for dilation in (2, 4):
            for groups, mode, batched in ((1, "zeros", True), (2, "reflect", True), (1, "zeros", False)):
                torch.manual_seed(7)
                native = nn.Conv1d(4, 6, 3, padding=dilation, dilation=dilation,
                                   groups=groups, padding_mode=mode).double()
                split = SplitBiasConv1d(4, 6, 3, padding=dilation, dilation=dilation,
                                        groups=groups, padding_mode=mode).double()
                split.load_state_dict(native.state_dict(), strict=True)
                x = torch.randn((2, 4, 17) if batched else (4, 17), dtype=torch.double)
                a, b = x.requires_grad_(), x.detach().clone().requires_grad_()
                ya, yb = native(a), split(b)
                gradient = torch.randn_like(ya)
                ya.backward(gradient)
                yb.backward(gradient)
                for expected, actual in ((ya, yb), (a.grad, b.grad),
                                         (native.weight.grad, split.weight.grad), (native.bias.grad, split.bias.grad)):
                    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)

    def test_legacy_config_remains_native_after_user_changes_default(self):
        @dataclass(frozen=True)
        class FutureDefault(Config):
            t2_conv_execution: str = "split_bias"

        legacy = CONFIG.to_dict()
        legacy.pop("t2_conv_execution")
        self.assertEqual(FutureDefault.from_dict(legacy).t2_conv_execution, "native")

    def test_native_matches_frozen_v9_model_and_rng_exactly(self):
        spec = importlib.util.spec_from_file_location("v9_model", PROJECT_ROOT / "smoke_tests/fixtures/v9/model.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        cfg = replace(CONFIG, device="cpu", windows_ms=(500,), t2_conv_execution="native")
        states, gradients, rngs = [], [], []
        for cls in (old.SpikeTFNet, build_model):
            seed_everything(17, cfg)
            model = cls(cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            output = model(torch.randn(2, 9, 125), torch.randn(2, 9, 560))
            joint_loss(output, torch.tensor([1, 2]), cfg).backward()
            gradients.append({k: p.grad.clone() for k, p in model.named_parameters()})
            optimizer.step()
            states.append(copy.deepcopy(model.state_dict()))
            rngs.append(torch.get_rng_state())
        for a, b in ((states[0], states[1]), (gradients[0], gradients[1])):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(*rngs, rtol=0, atol=0)

    def test_all_variants_keep_state_keys_and_only_replace_two_convolutions(self):
        for variant in ("full", "spectral_only", "temporal_only", "equal_fusion"):
            cfg = replace(CONFIG, model_variant=variant)
            native = build_model(cfg)
            split = build_model(replace(cfg, t2_conv_execution="split_bias"))
            split.load_state_dict(native.state_dict(), strict=True)
            self.assertEqual(list(native.state_dict()), list(split.state_dict()))
            self.assertEqual(sum(p.numel() for p in native.parameters()), sum(p.numel() for p in split.parameters()))
            names = [name for name, module in split.named_modules() if isinstance(module, SplitBiasConv1d)]
            self.assertEqual(names, [] if variant == "spectral_only" else
                             ["temporal.blocks.1.dilated.0", "temporal.blocks.2.dilated.0"])

    def test_split_model_optimizer_rng_resume(self):
        cfg = replace(CONFIG, t2_conv_execution="split_bias", model_variant="temporal_only", windows_ms=(500,))
        seed_everything(9, cfg)
        model = build_model(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        def step(net, opt):
            opt.zero_grad(set_to_none=True)
            outputs = net(torch.randn(2, 9, 125), torch.randn(2, 9, 560))
            loss = joint_loss(outputs, torch.tensor([1, 2]), cfg)
            loss.backward()
            opt.step()
            self.assertTrue(torch.isfinite(loss))

        step(model, optimizer)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, buffer)
        step(model, optimizer)
        restored = build_model(cfg)
        restored_opt = torch.optim.Adam(restored.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        restored.load_state_dict(state["model"])
        restored_opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        step(restored, restored_opt)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_execution_switch_cannot_silently_resume_and_v9_hashes_are_authenticated(self):
        prepared = PreparedData(np.zeros((2, 9, 250)), np.array([[1, 1, 0], [2, 1, 0]]),
                                np.arange(40), np.arange(40), {"fingerprint": "test"})
        hashes = {name: name for name in SOURCE_NAMES}
        legacy = CONFIG.to_dict()
        legacy.pop("t2_conv_execution")
        manifest = {"format_version": 9, "config": legacy, "purpose": "smoke",
                    "dataset_fingerprint": "test", "selected_trial_count": 2, "code_sha256": hashes}
        with self.assertRaisesRegex(ValueError, "t2_conv_execution"):
            validate_resume(manifest, replace(CONFIG, t2_conv_execution="split_bias"), prepared, "smoke", hashes)
        base = PROJECT_ROOT / "smoke_tests/results"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            details = Path(temp)
            old_hashes = dict(hashes)
            for name in CONV_EXECUTION_V9_SOURCES:
                path = details / "source" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = (PROJECT_ROOT / "smoke_tests/fixtures/v9" / name).read_bytes().replace(b"\r\n", b"\n")
                for ending in (b"\n", b"\r\n"):
                    path.write_bytes(raw.replace(b"\n", ending))
                    self.assertTrue(verified_source_upgrade(name, file_sha256(path), details, CONV_EXECUTION_V9_SOURCES))
                old_hashes[name] = file_sha256(path)
            validate_resume({**manifest, "code_sha256": old_hashes}, CONFIG, prepared, "smoke", hashes, details)
            (details / "source/model.py").write_bytes(raw + b"\n# unknown change\n")
            with self.assertRaisesRegex(ValueError, "model.py"):
                validate_resume({**manifest, "code_sha256": old_hashes}, CONFIG, prepared, "smoke", hashes, details)
