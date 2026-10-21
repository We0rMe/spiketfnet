"""真实 v4 断点升级到分支计时版本；四组训练轨迹与缺失历史验证。"""

import csv
import io
import json
import os
import shutil
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from experiments.ablation import run_ablation
from config import CONFIG
from data import prepare_data
from evaluate import read_history, run_path
from smoke_tests.run_ablation_smoke import MODEL_VARIANTS, compare_runs, load_module
from utils.timing import PARTS, timing_totals
from utils.provenance import LEGACY_SOURCE_PATHS
import train


def main(output_name="timing_check"):
    base = ROOT / "smoke_tests/results" / output_name
    base.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(base / "plot_cache")
    stream = io.StringIO()
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.discover(str(ROOT / "smoke_tests"), pattern="test_*.py"))
    (base / "checks.txt").write_text(stream.getvalue(), encoding="utf-8")
    if not outcome.wasSuccessful():
        print(stream.getvalue())
        raise AssertionError("Unit checks failed")
    source = base / "v4_source"
    source.mkdir()
    for name in ("config.py", "model.py", "ablation_model.py", "ablation.py", "data.py", "requirements.txt"):
        shutil.copy2(ROOT / LEGACY_SOURCE_PATHS.get(name, name), source / name)
    for name in ("train.py", "evaluate.py"):
        shutil.copy2(ROOT / "smoke_tests/fixtures" / name.replace(".py", "_v4.py"), source / name)
    old_eval = load_module("timing_evaluate_v4", source / "evaluate.py")
    with patch.dict(sys.modules, {"evaluate": old_eval}):
        old = load_module("timing_train_v4", source / "train.py")
    old.PROJECT_ROOT = source
    cfg = replace(CONFIG, data_root=Path(r"G:\SSVEP\data\benchmark"), subject_ids=(1, 2, 3), N=1, M=1,
                  seed=42, windows_ms=(500,), epochs=2, batch_size=8, device="cpu", evaluation_device="cpu",
                  num_workers=0, cpu_threads=2, monitor_every=1, plot_every=1, resume=True, time_branches=True,
                  output_root=base / "upgraded", cache_root=base.parent / "cache")
    data = prepare_data(cfg)
    selected = data.subset(np.flatnonzero((data.ids[:, 1] == 1) & (data.ids[:, 2] % 5 == 0)))
    old_save = old.save_resume

    def interrupt(*args):
        old_save(*args)
        if len(args[5]) == 1:
            raise KeyboardInterrupt("Injected v4 epoch-1 interruption")

    runs = []
    for variant in MODEL_VARIANTS:
        group = replace(cfg, model_variant=variant, experiment_name=variant)
        old.run_experiment(replace(group, output_root=base / "reference"), selected, purpose="smoke")
        with patch.object(old, "save_resume", side_effect=interrupt):
            try:
                old.run_experiment(group, selected, purpose="smoke")
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Missing injected interruption")
        root = cfg.output_root / variant
        path = root / "details/experiment.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        del manifest["config"]["time_branches"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        train.run_experiment(group, selected, purpose="smoke")
        compare_runs(base / "reference" / variant, root, cfg)
        folder = run_path(root, 500, 1, cfg.seed)
        history = read_history(folder / "details/history.csv")
        self_times = timing_totals([(variant, history)], cfg.epochs)
        assert history[0]["spectral_train_seconds"] is None and history[0]["temporal1_train_seconds"] is None
        for branch in PARTS:
            absent = variant == ("temporal_only" if branch == "spectral" else "spectral_only")
            key = branch + "_train_seconds"
            assert history[1][key] == 0 if absent else history[1][key] > 0
            assert self_times[key]["total_seconds"] == 0 if absent else self_times[key]["total_seconds"] is None
            if not absent:
                assert self_times[key]["recorded_epochs"] == 1 and self_times[key]["expected_epochs"] == 2
        assert abs(self_times["train_seconds"]["total_seconds"] - sum(row["train_seconds"] for row in history)) < 1e-9
        assert "Spectral Times=" in (folder / "training.log").read_text(encoding="utf-8")
        assert "完整总时长未知" in (folder / "report.md").read_text(encoding="utf-8")
        runs.append((variant, history))
    with patch("train.train_epoch", side_effect=AssertionError("Completed task trained again")):
        run_ablation(cfg, MODEL_VARIANTS, selected, purpose="smoke")
    with (cfg.output_root / "comparison.csv").open(encoding="utf-8-sig") as handle:
        totals = {row["metric"]: row for row in csv.DictReader(handle)
                  if row["scope"] == "training_time_total" and row["variant"] == "all_selected"}
    expected = timing_totals(runs, cfg.epochs)
    for key, item in expected.items():
        assert float(totals[key]["recorded_seconds"]) == item["recorded_seconds"]
        assert int(totals[key]["expected_epochs"]) == item["expected_epochs"]
        assert (totals[key]["total_seconds"] == "") == (item["total_seconds"] is None)
    report = (f"# Configuration and branch timing: PASS\n\n{outcome.testsRun} checks passed.\n\n"
              "Real Benchmark S1-S3, 8 trials/subject, S1 held out, seed 42, 500 ms, 2 epochs.\n"
              "All four genuine v4 checkpoints resumed after epoch 1 with timing enabled.\n"
              "Final weights, gradients (unit checks), training/monitor metrics and predictions match untimed training.\n"
              "Old epoch branch times remain missing. New times appear in log/history/report; aggregate coverage and totals verified.\n"
              "Disabled branches contribute zero; disabled instrumentation contributes unknown for present branches.\n"
              "Synthetic-clock tests verify backward attribution excludes shared work; mocked CUDA-event path syncs only at epoch end.\n"
              "Actual hardware verification is CPU only; CUDA timing still requires the server.\n")
    (base / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
