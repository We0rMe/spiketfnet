"""消融变量控制、输出缺项和旧断点边界；CPU 数值验证。"""

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ablation import summarize_ablation
from ablation_model import build_model, joint_loss
from config import CONFIG, PROJECT_ROOT, Config
from data import PreparedData, file_sha256
from evaluate import evaluate_model, run_path, write_json
from model import SpikeTFNet
from train import ABLATION_V3_SOURCES, seed_everything, validate_resume, verified_source_upgrade

MODEL_VARIANTS = ("full", "spectral_only", "temporal_only", "equal_fusion")


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = replace(CONFIG, device="cpu", evaluation_device="cpu", windows_ms=(500, 1000))

    def test_shared_initialization_and_original_full_model(self):
        seed_everything(42, self.cfg)
        original = SpikeTFNet(self.cfg)
        expected_rng = torch.get_rng_state()
        for variant in MODEL_VARIANTS:
            with self.subTest(variant=variant):
                seed_everything(42, self.cfg)
                model = build_model(replace(self.cfg, model_variant=variant))
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
                if variant == "full":
                    self.assertIs(type(model), SpikeTFNet)
                    self.assertEqual(set(model.state_dict()), set(original.state_dict()))
                else:
                    self.assertFalse(hasattr(model.fusion, "gate"))
                self.assertEqual(hasattr(model, "spectral"), variant != "temporal_only")
                self.assertEqual(hasattr(model, "temporal"), variant != "spectral_only")

    def test_equal_fusion_matches_full_with_constant_gate_even_during_training(self):
        cfg = self.cfg
        time, spectrum = torch.randn(2, 9, 125), torch.randn(2, 9, 560)
        seed_everything(42, cfg)
        full = build_model(cfg).train()
        seed_everything(42, cfg)
        equal = build_model(replace(cfg, model_variant="equal_fusion")).train()
        handle = full.fusion.gate.register_forward_hook(lambda module, inputs, output: torch.full_like(output, 0.5))
        seed_everything(90, cfg)
        reference = full(time, spectrum)
        seed_everything(90, cfg)
        actual = equal(time, spectrum)
        handle.remove()
        for key in actual:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        self.assertNotIn("gate", actual)

    def test_single_branch_preserves_projection_and_final_classifier(self):
        for variant, branch in (("spectral_only", "spectral"), ("temporal_only", "temporal")):
            model = build_model(replace(self.cfg, model_variant=variant)).eval()
            captured = []
            handle = getattr(model.fusion, branch).register_forward_hook(
                lambda module, inputs, output: captured.append(output.detach().clone()))
            with torch.no_grad():
                for window in self.cfg.windows_ms:
                    time, spectrum = torch.randn(2, 9, self.cfg.samples(window)), torch.randn(2, 9, 560)
                    output = model(time, spectrum)
                    torch.testing.assert_close(output["embedding"], captured[-1], rtol=0, atol=0)
                    torch.testing.assert_close(output["logits"], model.fusion.classifier(captured[-1]), rtol=0, atol=0)
                    # 不存在的输入分支不得影响预测，且时域膜电位不跨调用残留。
                    other_time = time if branch == "temporal" else torch.full_like(time, float("nan"))
                    other_spectrum = spectrum if branch == "spectral" else torch.full_like(spectrum, float("nan"))
                    torch.testing.assert_close(model(other_time, other_spectrum)["logits"], output["logits"], rtol=0, atol=0)
            handle.remove()

    def test_loss_gradients_and_evidence_for_all_variants(self):
        for variant in MODEL_VARIANTS:
            with self.subTest(variant=variant):
                cfg = replace(self.cfg, model_variant=variant)
                model = build_model(cfg)
                time, spectrum, labels = torch.randn(2, 9, 125), torch.randn(2, 9, 560), torch.tensor([0, 1])
                outputs = model(time, spectrum)
                expected = F.cross_entropy(outputs["logits"], labels)
                for key, weight in (("spectral_logits", cfg.lambda_spectral), ("temporal_logits", cfg.lambda_temporal)):
                    if key in outputs:
                        expected = expected + weight * F.cross_entropy(outputs[key], labels)
                loss = joint_loss(outputs, labels, cfg)
                torch.testing.assert_close(loss, expected, rtol=0, atol=0)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                for branch in ("spectral", "temporal", "fusion"):
                    if hasattr(model, branch):
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in getattr(model, branch).parameters()), 0)
                loader = DataLoader(TensorDataset(time, spectrum, labels), batch_size=2)
                metrics, arrays = evaluate_model(model, loader, cfg, 500, torch.device("cpu"))
                self.assertEqual(arrays["logits"].shape, (2, 40))
                self.assertEqual("gate" in arrays, variant == "full")
                self.assertEqual("spectral_logits" in arrays, variant != "temporal_only")
                self.assertEqual("temporal_logits" in arrays, variant != "spectral_only")
                self.assertEqual(len(arrays["spike_counts"]) == 0, variant == "spectral_only")
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                with torch.no_grad():
                    longer = model(torch.randn(2, 9, 250), spectrum)
                self.assertEqual(longer["logits"].shape, (2, 40))

    def test_comparison_pairs_subjects_and_seeds_and_excludes_incomplete_groups(self):
        results = PROJECT_ROOT / "smoke_tests/results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temp:
            cfg = replace(self.cfg, N=2, M=2, seed=42, output_root=Path(temp))
            for window in cfg.windows_ms:
                for variant, values in (("full", [[10, 30], [50, 70]]),
                                        ("equal_fusion", [[20, 40], [80, 100]])):
                    for i, subject in enumerate(cfg.test_subjects):
                        for j, seed in enumerate(cfg.seeds):
                            folder = run_path(cfg.output_root / variant, window, subject, seed)
                            write_json(folder / "details/provenance.json", dict(status="complete", model_variant=variant,
                                       window_ms=window, test_subject=subject, seed=seed))
                            write_json(folder / "metrics.json", dict(acc_percent=values[i][j],
                                       itr_bits_per_minute=values[i][j] * window / 500))
            rows = summarize_ablation(cfg, ("full", "equal_fusion", "spectral_only"))
            rows = [row for row in rows if row["scope"] != "training_time_total"]
            self.assertEqual(len(rows), 24)
            self.assertFalse(any(row["variant"] == "spectral_only" for row in rows))
            paired = [row for row in rows if row["comparison"] == "difference_from_full"]
            for row in paired:
                scale = row["window_ms"] / 500 if row["metric"] == "itr_bits_per_minute" else 1
                self.assertEqual(row["mean"], 20 * scale)
                self.assertEqual(row["variance"], 100 * scale ** 2 if row["scope"] == "across_subjects" else 0)

    def test_legacy_default_and_variant_resume_rejection(self):
        old = self.cfg.to_dict()
        del old["model_variant"]
        self.assertEqual(Config.from_dict(old).model_variant, "full")
        with self.assertRaises(ValueError):
            replace(self.cfg, model_variant="typo").validate()
        hashes = {name: name for name in ("model.py", "data.py", "train.py", "evaluate.py", "ablation_model.py")}
        prepared = PreparedData(np.zeros((2, 9, 250)), np.array([[1, 1, 0], [2, 1, 0]]),
                                np.arange(40), np.arange(40), {"fingerprint": "example"})
        manifest = {"format_version": 4, "config": self.cfg.to_dict(), "purpose": "smoke",
                    "dataset_fingerprint": "example", "selected_trial_count": 2, "code_sha256": hashes}
        for variant in MODEL_VARIANTS[1:]:
            with self.assertRaisesRegex(ValueError, "model_variant"):
                validate_resume(manifest, replace(self.cfg, model_variant=variant), prepared, "smoke", hashes)
        with self.assertRaisesRegex(ValueError, "ablation_model.py"):
            validate_resume(manifest, self.cfg, prepared, "smoke", {**hashes, "ablation_model.py": "changed"})

    def test_v3_upgrade_requires_exact_archived_sources(self):
        results = PROJECT_ROOT / "smoke_tests/results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temp:
            details = Path(temp)
            source = details / "source"
            source.mkdir()
            for name in ABLATION_V3_SOURCES:
                raw = (PROJECT_ROOT / "smoke_tests/fixtures" / name.replace(".py", "_v3.py")).read_bytes()
                for ending in (b"\n", b"\r\n"):
                    content = raw.replace(b"\r\n", b"\n").replace(b"\n", ending)
                    (source / name).write_bytes(content)
                    self.assertTrue(verified_source_upgrade(name, hashlib.sha256(content).hexdigest(), details,
                                                            ABLATION_V3_SOURCES))
                    self.assertFalse(verified_source_upgrade(name, "wrong hash", details, ABLATION_V3_SOURCES))
                (source / name).write_bytes(raw + b"\n# unknown edit\n")
                self.assertFalse(verified_source_upgrade(name, file_sha256(source / name), details, ABLATION_V3_SOURCES))

