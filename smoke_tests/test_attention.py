"""按时间位置归一化的注意力：投影轴、BN 统计、状态和窗长边界。"""

import unittest
from dataclasses import replace

import torch
from torch import nn
from spikingjelly.activation_based import functional

from config import CONFIG
from experiments.variants import build_model
from model import SpikeAttention


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_linear_preserves_time_positions_and_bn_uses_batch_and_features(self):
        cfg = replace(CONFIG, windows_ms=(20,), temporal_width=8, attention_heads=2, temporal_readout_bins=1)
        attention = SpikeAttention(cfg).double()
        linear, bn, _ = attention.qkv[0][0]
        self.assertIsInstance(linear, nn.Linear)
        self.assertIsNone(linear.bias)
        z = torch.arange(3 * 5 * 8, dtype=torch.double).reshape(3, 5, 8) / 20
        projected = linear(z)
        torch.testing.assert_close(projected, z @ linear.weight.T)
        changed = z.clone()
        changed[:, 2] += 10
        torch.testing.assert_close(linear(changed)[:, [0, 1, 3, 4]], projected[:, [0, 1, 3, 4]])
        mean = projected.mean((0, 2))
        var = projected.var((0, 2), unbiased=False)
        expected = (projected - mean[None, :, None]) / (var[None, :, None] + bn.eps).sqrt()
        torch.testing.assert_close(bn(projected), expected)
        torch.testing.assert_close(bn.running_mean, bn.momentum * mean)
        torch.testing.assert_close(bn.running_var,
                                   1 - bn.momentum + bn.momentum * projected.var((0, 2), unbiased=True))

    def test_each_head_normalizes_and_spikes_before_concatenation(self):
        cfg = replace(CONFIG, windows_ms=(500,), temporal_width=8, attention_heads=2)
        net = SpikeAttention(cfg)
        calls, handles = [], []
        for h, branch in enumerate(net.attended):
            def record(module, args, output, head=h):
                calls.append((head, tuple(output.shape)))
            handles.append(branch.register_forward_hook(record))
        handles.append(net.output[0].register_forward_pre_hook(
            lambda module, args: calls.append(("concat", tuple(args[0].shape)))))
        try:
            x = torch.randn(3, 8, 125, requires_grad=True)
            outputs = torch.stack([net(x) for _ in range(cfg.simulation_steps)])
            outputs.square().mean().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, [(0, (3, 125, 4)), (1, (3, 125, 4)),
                                 ("concat", (3, 125, 8))] * cfg.simulation_steps)
        self.assertEqual(outputs.shape, (4, 3, 125, 8))
        for name, parameter in net.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        neurons = [layer[-1] for head in net.qkv for layer in head] + [layer[-1] for layer in net.attended]
        self.assertEqual(len({cell.v.data_ptr() for cell in neurons}), len(neurons))
        for bn in (m for m in net.modules() if isinstance(m, nn.BatchNorm1d)):
            self.assertEqual(bn.num_features, 125)
            self.assertEqual(bn.num_batches_tracked.item(), cfg.simulation_steps)
        functional.reset_net(net)
        self.assertTrue(all(isinstance(cell.v, float) for cell in neurons))

    def test_window_is_explicit_and_wrong_window_cannot_share_bn(self):
        cfg = replace(CONFIG, windows_ms=(500, 1000), device="cpu")
        with self.assertRaisesRegex(ValueError, "window_ms"):
            build_model(cfg)
        for window in cfg.windows_ms:
            model = build_model(cfg, window_ms=window).eval()
            self.assertEqual(model.temporal.attention.length, cfg.samples(window))
            with torch.no_grad():
                out = model(torch.randn(2, 9, cfg.samples(window)), torch.randn(2, 9, cfg.spectral_features))
            self.assertEqual(out["logits"].shape, (2, 40))
            with self.assertRaisesRegex(ValueError, "EEG samples"):
                model.temporal.attention(torch.randn(2, cfg.temporal_width, cfg.samples(window) + 1))

    def test_eval_does_not_update_time_position_statistics(self):
        net = SpikeAttention(replace(CONFIG, windows_ms=(500,))).eval()
        before = {name: value.clone() for name, value in net.named_buffers()}
        x = torch.randn(2, CONFIG.temporal_width, 125)
        with torch.no_grad():
            first = net(x)
            functional.reset_net(net)
            second = net(x)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name, value in net.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
