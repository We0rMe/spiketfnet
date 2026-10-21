"""计时不改变训练，分支反向归属、缺失历史和配置选择的验证。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
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


class TimedBranch(nn.Module):
    def __init__(self, clock, forward_cost, backward_cost):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.clock, self.forward_cost, self.backward_cost = clock, forward_cost, backward_cost

    def forward(self, x):
        out = Tick.apply(x * self.weight, self.clock, self.forward_cost, self.backward_cost)
        return out, out


class TimedLayer(TimedBranch):
    def forward(self, x):
        return super().forward(x)[0]


class TimedTemporal(nn.Module):
    def __init__(self, clock, steps):
        super().__init__()
        self.clock, self.steps = clock, steps
        self.current = TimedLayer(clock, 1, 2)
        self.encoder = TimedLayer(clock, 3, 4)
        self.dropout = TimedLayer(clock, 5, 6)
        self.blocks = nn.Sequential(*(TimedLayer(clock, 7, 8) for _ in range(3)))
        self.attention = TimedLayer(clock, 9, 10)
        self.embedding = TimedLayer(clock, 11, 12)
        self.classifier = TimedLayer(clock, 13, 14)

    def forward(self, x):
        current = self.current(x)
        outputs = []
        for _ in range(self.steps):
            spikes = self.dropout(self.encoder(current))
            features = self.attention(self.blocks(spikes))
            pooled = Tick.apply(features, self.clock, 15, 16)
            outputs.append(self.classifier(self.embedding(pooled)))
        output = Tick.apply(torch.stack(outputs).mean(0), self.clock, 17, 18)
        return output, output


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
            for branch in PARTS:
                absent = variant == ("temporal_only" if branch == "spectral" else "spectral_only")
                key = branch + "_train_seconds"
                self.assertEqual(rows[0][key], 0 if absent else None)
                if absent:
                    self.assertEqual(rows[1][key], 0)
                else:
                    self.assertGreater(rows[1][key], 0)

    def test_backward_time_excludes_shared_work_even_without_input_gradients(self):
        clock = [0.0]
        model = nn.Module()
        model.spectral = TimedBranch(clock, 2, 3)
        model.temporal = TimedTemporal(clock, steps=4)
        with patch("utils.timing.time.perf_counter", side_effect=lambda: clock[0]):
            with BranchTimer(model, torch.device("cpu")) as timer:
                f, _ = model.spectral(torch.ones(1))
                t, _ = model.temporal(torch.ones(1))
                loss = Tick.apply(f + t, clock, 100, 200).sum()
                timer.prepare_backward(loss)
                loss.backward()
                timer.end_backward()
                self.assertEqual(timer.totals(), {"spectral_train_seconds": 5, "temporal1_train_seconds": 75,
                                                 "temporal2_train_seconds": 180, "temporal3_train_seconds": 435})
        self.assertEqual(clock[0], 995)

    def test_cuda_event_path_synchronizes_once_at_epoch_end(self):
        clock = [0.0]

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self, stream):
                self.at = clock[0]

            def elapsed_time(self, other):
                return (other.at - self.at) * 1000

        model = nn.Module()
        model.spectral = TimedBranch(clock, 2, 3)
        with patch("torch.cuda.Event", Event), patch("torch.cuda.current_stream"), patch("torch.cuda.synchronize") as sync:
            with BranchTimer(model, torch.device("cuda:0")) as timer:
                out, _ = model.spectral(torch.ones(1))
                loss = out.sum()
                timer.prepare_backward(loss)
                loss.backward()
                timer.end_backward()
                sync.assert_not_called()
                self.assertEqual(timer.totals(), {"spectral_train_seconds": 5, "temporal1_train_seconds": 0,
                                                 "temporal2_train_seconds": 0, "temporal3_train_seconds": 0})
                sync.assert_called_once_with(torch.device("cuda:0"))

    def test_history_coverage_missing_old_epochs_and_absent_branch(self):
        history = [{"train_seconds": 10}, {"train_seconds": 12, "spectral_train_seconds": 2, "temporal_train_seconds": 3}]
        totals = timing_totals([("full", history)], 2)
        self.assertEqual(totals["train_seconds"]["total_seconds"], 22)
        self.assertIsNone(totals["spectral_train_seconds"]["total_seconds"])
        self.assertEqual(totals["spectral_train_seconds"]["recorded_seconds"], 2)
        self.assertEqual(totals["spectral_train_seconds"]["recorded_epochs"], 1)
        self.assertEqual(timing_totals([("spectral_only", history)], 2)["temporal1_train_seconds"]["total_seconds"], 0)
        combined = timing_totals([("full", history), ("full", history)], 2)
        self.assertEqual(combined["train_seconds"]["total_seconds"], 44)
        self.assertEqual(combined["temporal1_train_seconds"]["expected_epochs"], 4)
        self.assertIsNone(combined["temporal1_train_seconds"]["recorded_seconds"])

    def test_trial_means_use_matching_sample_counts_and_keep_unknown_history(self):
        history = normalize_history([{"train_seconds": 8, "temporal_train_seconds": 6,
                                      "monitor_seconds": 2, "train_trials": 4},
                                     {"train_seconds": 18, "temporal1_train_seconds": 6, "train_trials": 6}])
        self.assertNotIn("temporal_train_seconds", history[0])
        self.assertNotIn("monitor_seconds", history[0])
        self.assertEqual(history[0]["train_seconds_per_trial"], 2)
        totals = timing_totals([("full", history)], 2)
        self.assertEqual(totals["train_seconds"]["seconds_per_trial"], 2.6)
        self.assertEqual(totals["temporal1_train_seconds"]["seconds_per_trial"], 1)
        self.assertIsNone(totals["temporal1_train_seconds"]["total_seconds"])
        self.assertEqual(totals["temporal1_train_seconds"]["recorded_trials"], 6)
        self.assertEqual(set(totals), set(TIME_KEYS))
        line = timing_log(history[1])
        self.assertEqual(line.count("ms/trial"), 2)
        self.assertIn("Temporal Times1=6.000s (1000.000ms/trial)", line)
        self.assertNotIn("monitor=", line)

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
