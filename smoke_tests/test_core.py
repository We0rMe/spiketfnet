"""关键数值、状态隔离和评估口径检查；不依赖 GPU。"""

import math
import unittest
from dataclasses import replace

import numpy as np
import torch
from scipy import signal
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG as PROJECT_CONFIG, Config
from data import complex_spectrum, loso_indices, preprocess_subject
from evaluate import aggregate_matrix, evaluate_model, itr_bits_per_minute
from model import GatedFusion, SpikeAttention, SpikeTFNet, joint_loss, lif_node
from train import seed_everything, train_epoch

CONFIG = replace(PROJECT_CONFIG, windows_ms=(500, 600, 700, 800, 900, 1000),
                 device="cpu", evaluation_device="cpu")


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_fold_count_does_not_reduce_source_pool(self):
        cfg = replace(CONFIG, N=2, M=3, seed=42).validate()
        ids = np.array([(s, 1, c) for s in cfg.subject_ids for c in range(2)])
        self.assertEqual(cfg.test_subjects, (1, 2))
        self.assertEqual(cfg.seeds, (42, 43, 44))
        for subject in cfg.test_subjects:
            train, test = loso_indices(ids, subject, cfg.subject_ids)
            self.assertEqual(len(set(ids[train, 0])), 34)
            self.assertEqual(set(ids[test, 0]), {subject})
            self.assertFalse(set(train) & set(test))
        for bad in (0, 36):
            with self.assertRaises(ValueError):
                replace(CONFIG, N=bad).validate()

    def test_config_snapshot_roundtrip(self):
        import json
        restored = Config.from_dict(json.loads(json.dumps(CONFIG.to_dict())))
        self.assertEqual(restored, CONFIG)

    def test_filter_trial_independence_and_crop(self):
        cfg = replace(CONFIG, subject_ids=(1, 2), N=1, raw_channels=2, raw_samples=500,
                      num_classes=2, num_blocks=2, channels=("O1", "Oz"), windows_ms=(500, 1000))
        raw = np.zeros((2, 500, 2, 2))
        raw[0, 170, 1, 1] = 1
        result = preprocess_subject(raw, cfg, [0, 1])
        self.assertEqual(result.shape, (4, 2, 250))
        np.testing.assert_array_equal(result[:3], 0)
        sos = signal.butter(4, (8, 64), btype="bandpass", fs=250, output="sos")
        reference = signal.sosfiltfilt(sos, raw[0, :, 1, 1])[160:410]
        np.testing.assert_allclose(result[3, 0], reference, atol=1e-8)
        self.assertEqual([CONFIG.samples(w) for w in CONFIG.windows_ms], [125, 150, 175, 200, 225, 250])

    def test_complex_fft_phase_scale_and_endpoint(self):
        x = 3 * np.sin(2 * np.pi * 10 * np.arange(125) / 250)
        features = complex_spectrum(x[None, None, :], CONFIG)
        self.assertEqual(features.shape, (1, 1, 560))
        self.assertAlmostEqual(float(features[0, 0, 10]), 0, places=5)
        self.assertAlmostEqual(float(features[0, 0, 280 + 10]), -3, places=5)
        extended = complex_spectrum(x[None, None, :], replace(CONFIG, include_upper_bin=True))
        self.assertEqual(extended.shape[-1], 562)

    def test_lif_matches_paper_recurrence_and_surrogate(self):
        cell = lif_node(CONFIG)
        current = torch.tensor([0.6], requires_grad=True)
        spikes = torch.stack([cell(current) for _ in range(4)])
        torch.testing.assert_close(spikes[:, 0], torch.tensor([0., 0., 1., 0.]))
        spikes.sum().backward()
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertGreater(current.grad.abs().item(), 0)
        cell.reset()
        self.assertEqual(cell(current).item(), 0)

    def test_attention_is_unsoftmaxed_token_coincidence(self):
        cfg = replace(CONFIG, temporal_width=4, attention_heads=2)
        attention = SpikeAttention(cfg)
        attention.qkv = nn.ModuleList([nn.Identity() for _ in range(3)])
        attention.attended = nn.Identity()
        attention.output = nn.Identity()
        x = torch.tensor([[[1., 0., 1.], [0., 1., 1.], [1., 1., 0.], [0., 0., 1.]]])
        expected = torch.zeros_like(x)
        for head in range(2):
            tokens = x[0, head * 2:(head + 1) * 2].T
            for i in range(3):
                for j in range(3):
                    expected[0, head * 2:(head + 1) * 2, i] += cfg.attention_scale * torch.dot(tokens[i], tokens[j]) * tokens[j]
        torch.testing.assert_close(attention(x), x + expected)

    def test_gate_endpoints_and_three_way_loss(self):
        f, t = torch.randn(2, 4), torch.randn(2, 4)
        torch.testing.assert_close(GatedFusion.blend(f, t, torch.zeros_like(f)), f)
        torch.testing.assert_close(GatedFusion.blend(f, t, torch.ones_like(f)), t)
        outputs = {key: torch.randn(2, 40, requires_grad=True) for key in ("logits", "spectral_logits", "temporal_logits")}
        labels = torch.tensor([1, 3])
        loss = joint_loss(outputs, labels, CONFIG)
        expected = sum(weight * nn.functional.cross_entropy(outputs[key], labels) for key, weight in
                       (("logits", 1), ("spectral_logits", .3), ("temporal_logits", .3)))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(all(value.grad.abs().sum() > 0 for value in outputs.values()))

    def test_model_all_windows_and_state_isolation(self):
        seed_everything(42, CONFIG)
        model = SpikeTFNet(CONFIG).eval()
        with torch.no_grad():
            for window in CONFIG.windows_ms:
                time = torch.randn(2, 9, CONFIG.samples(window))
                freq = torch.randn(2, 9, CONFIG.spectral_features)
                output = model(time, freq)
                self.assertEqual(output["logits"].shape, (2, 40))
                self.assertEqual(output["gate"].shape, (2, 128))
                self.assertTrue(((output["gate"] > 0) & (output["gate"] < 1)).all())
            first = model(time[:1], freq[:1])["logits"]
            model(torch.randn(3, 9, 125), torch.randn(3, 9, 560))
            second = model(time[:1], freq[:1])["logits"]
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(first, model(time, freq)["logits"][:1], rtol=1e-4, atol=1e-5)

    def test_real_optimizer_step_and_eval_bn_isolation(self):
        seed_everything(42, CONFIG)
        model = SpikeTFNet(CONFIG)
        loader = DataLoader(TensorDataset(torch.randn(4, 9, 125), torch.randn(4, 9, 560), torch.tensor([0, 1, 2, 3])), batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
        previous = model.fusion.classifier.weight.detach().clone()
        row = train_epoch(model, loader, optimizer, CONFIG, torch.device("cpu"))
        self.assertTrue(math.isfinite(row["loss"]))
        self.assertFalse(torch.equal(previous, model.fusion.classifier.weight))
        for branch in (model.spectral, model.temporal, model.fusion):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in branch.parameters() if p.grad is not None), 0)
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        metrics, evidence = evaluate_model(model, loader, CONFIG, 500, torch.device("cpu"))
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        self.assertEqual(set(metrics), {"acc_percent", "itr_bits_per_minute"})
        self.assertTrue((evidence["spike_counts"] <= evidence["spike_elements"]).all())
        self.assertGreater(int(evidence["spike_elements"].sum()), 0)

    def test_simulation_steps_are_independent_hyperparameter(self):
        cfg_one = replace(CONFIG, simulation_steps=1)
        seed_everything(42, CONFIG)
        model_one = SpikeTFNet(cfg_one)
        model_four = SpikeTFNet(CONFIG)
        model_four.load_state_dict(model_one.state_dict())
        loader = DataLoader(TensorDataset(torch.randn(2, 9, 125), torch.randn(2, 9, 560), torch.tensor([0, 1])), batch_size=2)
        _, one = evaluate_model(model_one, loader, cfg_one, 500, torch.device("cpu"))
        _, four = evaluate_model(model_four, loader, CONFIG, 500, torch.device("cpu"))
        np.testing.assert_array_equal(four["spike_elements"], 4 * one["spike_elements"])
        self.assertEqual(four["logits"].shape, one["logits"].shape)

    def test_itr_limits(self):
        for p in (0, .01, 1 / 40):
            self.assertEqual(itr_bits_per_minute(p, 40, 1), 0)
        self.assertAlmostEqual(itr_bits_per_minute(1, 40, 1), 60 * math.log2(40))
        self.assertAlmostEqual(itr_bits_per_minute(.8, 40, 1.5), itr_bits_per_minute(.8, 40, 1) / 1.5)

    def test_variance_and_hierarchical_aggregation(self):
        stats = aggregate_matrix([[10, 30], [50, 70]], 0)
        self.assertEqual(stats["across_repeats"], {"count": 2, "mean": 40., "variance": 100.})
        self.assertEqual(stats["across_subjects"]["variance"], 400.)
        self.assertEqual(aggregate_matrix([[20]], 0)["across_repeats"]["variance"], 0)
        self.assertIsNone(aggregate_matrix([[20]], 1)["across_repeats"]["variance"])


if __name__ == "__main__":
    unittest.main()
