"""消融变量控制、输出缺项和旧断点边界；CPU 数值验证。"""

import hashlib
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.ablation import summarize_ablation
from experiments.variants import build_model, joint_loss
from config import CONFIG, PROJECT_ROOT, Config
from data import PreparedData, file_sha256
from evaluate import evaluate_model, run_path, write_json
from model import SpikeTFNet
from train import seed_everything
from utils.provenance import ABLATION_V3_SOURCES, FUSION_V15_SOURCES, SOURCE_NAMES, validate_resume, verified_source_upgrade

MODEL_VARIANTS = ("full", "spectral_only", "temporal_only", "equal_fusion", "equal_supervised")


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = replace(CONFIG, device="cpu", evaluation_device="cpu", windows_ms=(500,))

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

    def test_spectral_control_readout_changes_only_downstream_initialization(self):
        models, states = [], []
        for bins in (1, 32):
            cfg = replace(self.cfg, model_variant="spectral_only", temporal_readout_bins=bins)
            seed_everything(42, cfg)
            models.append(build_model(cfg).state_dict())
            states.append(torch.get_rng_state())
        self.assertEqual(models[0].keys(), models[1].keys())
        for name in models[0]:
            if name.startswith("spectral."):
                torch.testing.assert_close(models[0][name], models[1][name], rtol=0, atol=0)
        self.assertFalse(torch.equal(models[0]["fusion.spectral.0.weight"], models[1]["fusion.spectral.0.weight"]))
        self.assertFalse(torch.equal(models[0]["fusion.classifier.weight"], models[1]["fusion.classifier.weight"]))
        self.assertFalse(torch.equal(states[0], states[1]))

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
                if variant == "equal_supervised":
                    expected = expected + cfg.lambda_fusion_branch * (
                        F.cross_entropy(outputs["fusion_spectral_logits"], labels)
                        + F.cross_entropy(outputs["fusion_temporal_logits"], labels))
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
                for key in ("fusion_spectral_logits", "fusion_temporal_logits"):
                    self.assertEqual(key in arrays, variant == "equal_supervised")
                self.assertEqual(len(arrays["spike_counts"]) == 0, variant == "spectral_only")
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                longer_model = build_model(replace(cfg, windows_ms=(500, 1000)), window_ms=1000).eval()
                with torch.no_grad():
                    longer = longer_model(torch.randn(2, 9, 250), spectrum)
                self.assertEqual(longer["logits"].shape, (2, 40))

    def test_supervision_changes_loss_only_and_zero_weight_recovers_equal_gradients(self):
        base_cfg = replace(self.cfg, model_variant="equal_fusion")
        new_cfg = replace(self.cfg, model_variant="equal_supervised", lambda_fusion_branch=0)
        models = []
        for cfg in (base_cfg, new_cfg):
            seed_everything(42, cfg)
            models.append(build_model(cfg).train())
        time, spectrum, labels = torch.randn(2, 9, 125), torch.randn(2, 9, 560), torch.tensor([2, 6])
        results = []
        for cfg, model in zip((base_cfg, new_cfg), models):
            seed_everything(90, cfg)
            out = model(time, spectrum)
            loss = joint_loss(out, labels, cfg)
            loss.backward()
            results.append((out, loss, torch.get_rng_state()))
        for key in results[0][0]:
            torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)
        for index in (1, 2):
            torch.testing.assert_close(results[0][index], results[1][index], rtol=0, atol=0)
        for (name, parameter), (other_name, other) in zip(models[0].named_parameters(), models[1].named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)

    def test_path_supervision_reaches_projection_and_shared_classifier(self):
        cfg = replace(self.cfg, model_variant="equal_supervised")
        model = build_model(cfg).train()
        out = model(torch.randn(2, 9, 125), torch.randn(2, 9, 560))
        labels = torch.tensor([2, 6])
        loss = sum(F.cross_entropy(out[f"fusion_{branch}_logits"], labels)
                   for branch in ("spectral", "temporal"))
        loss.backward()
        for path in (model.spectral.embedding, model.temporal.embedding,
                     model.fusion.spectral, model.fusion.temporal, model.fusion.classifier):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in path.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in model.spectral.classifier.parameters()))
        torch.testing.assert_close(out["logits"],
                                   .5 * (out["fusion_spectral_logits"] + out["fusion_temporal_logits"]),
                                   atol=1e-6, rtol=1e-5)

    def test_v15_variants_keep_forward_backward_and_buffers(self):
        spec = importlib.util.spec_from_file_location(
            "v15_variants", PROJECT_ROOT / "smoke_tests/fixtures/v15/variants.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for variant in MODEL_VARIANTS[:-1]:
            cfg = replace(self.cfg, model_variant=variant)
            models = []
            for builder in (old.build_model, build_model):
                seed_everything(42, cfg)
                models.append(builder(cfg).train())
            time, spectrum, labels = torch.randn(2, 9, 125), torch.randn(2, 9, 560), torch.tensor([2, 6])
            results = []
            for model, loss_function in zip(models, (old.joint_loss, joint_loss)):
                seed_everything(90, cfg)
                out = model(time, spectrum)
                loss = loss_function(out, labels, cfg)
                loss.backward()
                results.append((out, loss, torch.get_rng_state()))
            self.assertEqual(results[0][0].keys(), results[1][0].keys())
            for key in results[0][0]:
                torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)
            for index in (1, 2):
                torch.testing.assert_close(results[0][index], results[1][index], rtol=0, atol=0)
            for (name, parameter), (_, other) in zip(models[0].named_parameters(), models[1].named_parameters()):
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
            for name, value in models[0].state_dict().items():
                torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)

    def test_v15_resume_requires_authentic_variant_and_preserved_loss(self):
        cfg = replace(self.cfg, model_variant="equal_fusion", lambda_fusion_branch=0)
        hashes = {name: file_sha256(PROJECT_ROOT / name) for name in SOURCE_NAMES}
        prepared = PreparedData(np.zeros((2, 9, 125)), np.array([[1, 1, 0], [2, 1, 0]]),
                                np.arange(40), np.arange(40), {"fingerprint": "example"})
        folder = PROJECT_ROOT / "smoke_tests/results"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            details = Path(temp)
            source = details / "source/experiments/variants.py"
            source.parent.mkdir(parents=True)
            source.write_bytes((PROJECT_ROOT / "smoke_tests/fixtures/v15/variants.py").read_bytes())
            legacy = details / "source/ablation_model.py"
            legacy.write_bytes(source.read_bytes())
            self.assertTrue(verified_source_upgrade("ablation_model.py", file_sha256(legacy), details, FUSION_V15_SOURCES))
            snapshot = cfg.to_dict()
            del snapshot["lambda_fusion_branch"]
            manifest = {"format_version": 15, "config": snapshot, "purpose": "smoke",
                        "dataset_fingerprint": "example", "selected_trial_count": 2,
                        "code_sha256": {**hashes, "experiments/variants.py": file_sha256(source)}}
            validate_resume(manifest, cfg, prepared, "smoke", hashes, details)
            with self.assertRaisesRegex(ValueError, "lambda_fusion_branch"):
                validate_resume(manifest, replace(cfg, lambda_fusion_branch=.3), prepared, "smoke", hashes, details)
            with self.assertRaisesRegex(ValueError, "model_variant"):
                validate_resume(manifest, replace(cfg, model_variant="equal_supervised"), prepared, "smoke", hashes, details)
            source.write_bytes(source.read_bytes() + b"\n# unverified edit\n")
            manifest["code_sha256"]["experiments/variants.py"] = file_sha256(source)
            with self.assertRaisesRegex(ValueError, "source code differs"):
                validate_resume(manifest, cfg, prepared, "smoke", hashes, details)

    def test_comparison_pairs_subjects_and_seeds_and_excludes_incomplete_groups(self):
        results = PROJECT_ROOT / "smoke_tests/results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temp:
            cfg = replace(self.cfg, N=2, M=2, seed=42, windows_ms=(500, 1000), output_root=Path(temp))
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
                self.assertEqual(row["std"], 10 * scale if row["scope"] == "across_subjects" else 0)

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
        hashes = {("experiments/variants.py" if k == "ablation_model.py" else k): v for k, v in hashes.items()}
        with self.assertRaisesRegex(ValueError, "ablation_model.py"):
            validate_resume(manifest, self.cfg, prepared, "smoke", {**hashes, "experiments/variants.py": "changed"})

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

