"""简洁分支计时不改变训练；历史缺项、消融缺失分支和汇总口径。"""

import copy
import unittest
from dataclasses import replace

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG
from experiments.variants import build_model
from train import seed_everything, train_epoch
from utils.branch_timing import timing_log, timing_totals, timing_markdown
from utils.timing import SUMMARY_KEYS, normalize_history


class BranchSummaryTests(unittest.TestCase):
    def test_branch_only_timer_preserves_both_modes_all_groups(self):
        torch.set_num_threads(2)
        for mode in ("native", "split_bias"):
            for variant in ("full", "spectral_only", "temporal_only", "equal_fusion"):
                cfg = replace(CONFIG, device="cpu", windows_ms=(500,), t2_conv_execution=mode,
                              model_variant=variant, batch_size=2, t2_profile_batch=0)
                states, gradients, rngs = [], [], []
                for enabled in (False, True):
                    seed_everything(21, cfg)
                    model = build_model(cfg)
                    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                    data = TensorDataset(torch.randn(3, 9, 125), torch.randn(3, 9, 560), torch.tensor([0, 1, 2]))
                    row = train_epoch(model, DataLoader(data, batch_size=2), optimizer,
                                      replace(cfg, time_branches=enabled), torch.device("cpu"))
                    self.assertIsNone(row["_t2_diagnostics"])
                    self.assertGreater(row["train_seconds"], 0)
                    self.assertNotIn("T2", timing_log(row))
                    self.assertNotIn("/trial", timing_log(row))
                    for key, absent in ((SUMMARY_KEYS[0], variant == "temporal_only"),
                                        (SUMMARY_KEYS[1], variant == "spectral_only")):
                        if absent or not enabled:
                            self.assertEqual(row[key], 0 if absent else None)
                        else:
                            self.assertGreater(row[key], 0)
                    states.append(copy.deepcopy(model.state_dict()))
                    gradients.append({n: p.grad.clone() for n, p in model.named_parameters()})
                    rngs.append(torch.get_rng_state())
                    self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
                for a, b in ((states[0], states[1]), (gradients[0], gradients[1])):
                    for key in a:
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                torch.testing.assert_close(*rngs, rtol=0, atol=0)

    def test_old_partial_times_are_not_fabricated_as_branch_totals(self):
        rows = normalize_history([{"epoch": 1, "train_trials": 4, "temporal2_conv_forward_seconds": 10},
                                  {"epoch": 2, "train_trials": 4, "spectral_train_seconds": 1,
                                   "temporal_train_seconds": 2, "train_seconds": 5}])
        totals = timing_totals([("full", rows)], 2)
        self.assertEqual(tuple(totals), SUMMARY_KEYS)
        for key, expected in zip(SUMMARY_KEYS, (1, 2, 5)):
            self.assertIsNone(rows[0][key])
            self.assertIsNone(totals[key]["total_seconds"])
            self.assertEqual(totals[key]["recorded_seconds"], expected)
            self.assertEqual(totals[key]["seconds_per_trial"], expected / 4)
        self.assertEqual(timing_totals([("spectral_only", rows)], 2)["temporal_train_seconds"]["total_seconds"], 0)
        self.assertEqual(timing_totals([("temporal_only", rows)], 2)["spectral_train_seconds"]["total_seconds"], 0)
        self.assertIn("Spectral Times", "\n".join(timing_markdown(totals)))
