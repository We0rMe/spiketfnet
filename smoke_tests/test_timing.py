"""计时不改变训练，分支反向归属、缺失历史和配置选择的验证。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from spikingjelly.activation_based import neuron
from torch.utils.data import DataLoader, TensorDataset

from experiments.ablation import run_ablation
from experiments.variants import build_model
from config import CONFIG, PROJECT_ROOT
from data import PreparedData
from train import seed_everything, train_epoch
from utils.timing import BranchTimer, PARTS, TIME_KEYS, normalize_history, timing_log, timing_totals


class Tick(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clock, forward_cost, backward_cost):
        ctx.clock, ctx.cost = clock, backward_cost
        clock[0] += forward_cost
        return x.clone()

    @staticmethod
    def backward(ctx, gradient):
        ctx.clock[0] += ctx.cost
        return gradient, None, None, None


class TimedConv(nn.Conv1d):
    def __init__(self, clock):
        super().__init__(1, 1, 1)
        self.clock = clock

    def forward(self, x):
        return Tick.apply(x * self.weight.flatten()[0], self.clock, 2, 3)


class TimedLIF(neuron.LIFNode):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def forward(self, x):
        spike = Tick.apply(x + self.v, self.clock, 3, 4)
        self.v = Tick.apply(spike * 0.5, self.clock, 5, 7)
        return spike


class Untimed(nn.Module):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def forward(self, x):
        return Tick.apply(x, self.clock, 100, 200)


class TimedTemporal(nn.Module):
    def __init__(self, clock, steps=4):
        super().__init__()
        self.steps = steps
        self.current = TimedConv(clock)  # 第二模块外的卷积不能计入。
        self.encoder = TimedLIF(clock)
        self.blocks = nn.Sequential(*(nn.Sequential(TimedConv(clock), Untimed(clock), TimedLIF(clock),
                                                    TimedConv(clock), Untimed(clock), TimedLIF(clock))
                                      for _ in range(3)))
        self.attention = TimedConv(clock)

    def forward(self, x):
        current = self.encoder(self.current(x))
        return torch.stack([self.attention(self.blocks(current)) for _ in range(self.steps)]).mean(0)


class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_instrumentation_preserves_gradients_weights_rng_for_all_groups(self):
        for variant in ("full", "spectral_only", "temporal_only", "equal_fusion"):
            cfg = replace(CONFIG, model_variant=variant, device="cpu", windows_ms=(500,), batch_size=2)
            seed_everything(42, cfg)
            data = TensorDataset(torch.randn(3, 9, 125), torch.randn(3, 9, 560), torch.tensor([0, 1, 2]))
            states, gradients, rngs, rows = [], [], [], []
            for enabled in (False, True):
                seed_everything(15, cfg)
                model = build_model(cfg)
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                loader = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(42))
                rows.append(train_epoch(model, loader, optimizer, replace(cfg, time_branches=enabled), torch.device("cpu")))
                states.append({key: value.clone() for key, value in model.state_dict().items()})
                gradients.append({key: value.grad.clone() for key, value in model.named_parameters()})
                rngs.append(torch.get_rng_state())
                self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
            for a, b in ((states[0], states[1]), (gradients[0], gradients[1])):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            torch.testing.assert_close(rngs[0], rngs[1], rtol=0, atol=0)
            for key in ("loss", "train_acc_percent"):
                self.assertEqual(rows[0][key], rows[1][key])
            self.assertEqual(rows[1]["train_trials"], 3)  # 包括末尾不足一个 batch 的样本。
            for key in TIME_KEYS:
                absent = variant == "spectral_only"
                self.assertEqual(rows[0][key], 0 if absent else None)
                if absent:
                    self.assertEqual(rows[1][key], 0)
                else:
                    self.assertGreater(rows[1][key], 0)

    def check_operator_times(self, device, clock):
        model = nn.Module()
        model.spectral = TimedConv(clock)
        model.temporal = TimedTemporal(clock)
        with BranchTimer(model, device) as timer:
            output = model.temporal(torch.ones(1)) + model.spectral(torch.ones(1))
            loss = Tick.apply(output, clock, 10000, 20000).sum()
            timer.prepare_backward(loss)
            loss.backward()
            timer.end_backward()
            if device.type == "cuda":
                torch.cuda.synchronize.assert_not_called()
            totals = timer.totals()
        self.assertEqual(totals, dict(zip(TIME_KEYS, (48, 72, 192, 222))))
        self.assertGreater(clock[0], sum(totals.values()) + 30000)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()))

    def test_backward_scope_excludes_bn_shared_work_and_includes_lif_state(self):
        clock = [0.0]
        with patch("utils.timing.time.perf_counter", side_effect=lambda: clock[0]):
            self.check_operator_times(torch.device("cpu"), clock)

    def test_cuda_event_path_synchronizes_once_at_epoch_end(self):
        clock = [0.0]

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self, stream):
                self.at = clock[0]

            def elapsed_time(self, other):
                return (other.at - self.at) * 1000

        with patch("torch.cuda.Event", Event), patch("torch.cuda.current_stream"), patch("torch.cuda.synchronize") as sync:
            self.check_operator_times(torch.device("cuda:0"), clock)
            sync.assert_called_once_with(torch.device("cuda:0"))

    def test_history_coverage_missing_old_epochs_and_absent_branch(self):
        key = TIME_KEYS[0]
        history = [{"temporal2_train_seconds": 10}, {key: 2, "train_trials": 4}]
        totals = timing_totals([("full", history)], 2)
        self.assertIsNone(totals[key]["total_seconds"])
        self.assertEqual(totals[key]["recorded_seconds"], 2)
        self.assertEqual(totals[key]["recorded_epochs"], 1)
        self.assertEqual(totals[key]["seconds_per_trial"], 0.5)
        self.assertEqual(timing_totals([("spectral_only", history)], 2)[key]["total_seconds"], 0)
        combined = timing_totals([("full", history), ("temporal_only", history)], 2)
        self.assertEqual(combined[key]["recorded_seconds"], 4)
        self.assertEqual(combined[key]["expected_epochs"], 4)
        self.assertIsNone(combined[TIME_KEYS[1]]["recorded_seconds"])

    def test_trial_means_use_matching_sample_counts_and_keep_unknown_history(self):
        key = TIME_KEYS[0]
        history = normalize_history([{key: 8, "temporal_train_seconds": 6, "monitor_seconds": 2, "train_trials": 4},
                                     {key: 18, "train_trials": 6}])
        self.assertNotIn("temporal_train_seconds", history[0])
        self.assertNotIn("monitor_seconds", history[0])
        self.assertEqual(history[0][key + "_per_trial"], 2)
        totals = timing_totals([("full", history)], 2)
        self.assertEqual(totals[key]["seconds_per_trial"], 2.6)
        self.assertEqual(totals[key]["recorded_trials"], 10)
        self.assertEqual(set(totals), set(TIME_KEYS))
        line = timing_log(history[1])
        self.assertNotIn("/trial", line)
        self.assertIn("T2 Conv Forward=18.000s", line)
        self.assertNotIn("monitor=", line)
        self.assertNotIn("Total Time", line)

    def test_single_group_selection_names_folder_without_mutating_main_config(self):
        results = PROJECT_ROOT / "smoke_tests/results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temp:
            cfg = replace(CONFIG, N=1, M=1, subject_ids=(1, 2), windows_ms=(500,), output_root=Path(temp))
            data = PreparedData(np.zeros((2, 9, 125)), np.array([[1, 1, 0], [2, 1, 0]]),
                                np.arange(40), np.arange(40), {"fingerprint": "test"})
            with patch("experiments.ablation.run_experiment") as run:
                run_ablation(cfg, ("spectral_only",), data, purpose="smoke")
                run.assert_called_once()
                actual = run.call_args.args[0]
                self.assertEqual(actual.model_variant, "spectral_only")
                self.assertEqual(actual.output_root / actual.experiment_name, Path(temp) / "spectral_only")
                self.assertEqual(cfg.model_variant, "full")
            with self.assertRaises(ValueError):
                run_ablation(cfg, "spectral_only", data, purpose="smoke")
