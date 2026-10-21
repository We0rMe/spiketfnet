"""真实 v5/v6/v7 四组断点迁移；检查第二模块算子计时与训练轨迹。"""

import io
import csv
import json
import os
import shutil
import subprocess
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
import torch

from config import CONFIG
from data import prepare_data
from evaluate import read_history, reevaluate_experiment, run_path
from experiments.ablation import run_ablation
from smoke_tests.run_ablation_smoke import MODEL_VARIANTS, compare_runs, load_module
from utils.provenance import LEGACY_SOURCE_PATHS, SOURCE_NAMES
from utils.timing import PARTS, TIME_KEYS, timing_totals
import train


def main(output_name="t2_operator_check", version=7):
    if version not in (5, 6, 7):
        raise ValueError("This smoke supports v5/v6/v7 fixtures")
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

    source = base / f"v{version}_source"
    source.mkdir()
    fixtures = ROOT / "smoke_tests/fixtures"
    names = SOURCE_NAMES if version >= 6 else (
        "train.py", "evaluate.py", "config.py", "model.py", "data.py", "requirements.txt", *LEGACY_SOURCE_PATHS)
    for name in names:
        original = ROOT / LEGACY_SOURCE_PATHS.get(name, name)
        if version >= 6 and (fixtures / f"v{version}" / name).is_file():
            original = fixtures / f"v{version}" / name
        elif version == 5 and name in ("train.py", "evaluate.py"):
            original = fixtures / name.replace(".py", "_v5.py")
        elif version == 5 and name == "training_timing.py":
            original = fixtures / "v6/utils/timing.py"
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, source / name)
    helpers = fixtures / ("v7" if version == 7 else "v6")
    old_timing = load_module("frozen_timing", helpers / "utils/timing.py")
    old_provenance = load_module("frozen_provenance", helpers / "utils/provenance.py")
    with patch.dict(sys.modules, {"training_timing": old_timing, "utils.timing": old_timing,
                                 "utils.provenance": old_provenance}):
        old_eval = load_module("frozen_evaluate", source / "evaluate.py")
        with patch.dict(sys.modules, {"evaluate": old_eval}):
            old = load_module("frozen_train", source / "train.py")
    old.PROJECT_ROOT = source
    cfg = replace(CONFIG, data_root=Path(r"G:\SSVEP\data\benchmark"), subject_ids=(1, 2, 3), N=1, M=1,
                  seed=42, windows_ms=(500,), epochs=2, batch_size=8, device="cpu", evaluation_device="cpu",
                  num_workers=0, cpu_threads=2, monitor_every=1, plot_every=1, resume=True,
                  output_root=base / "upgraded", cache_root=base.parent / "cache")
    data = prepare_data(cfg)
    selected = data.subset(np.flatnonzero((data.ids[:, 1] == 1) & (data.ids[:, 2] % 5 == 0)))
    save = old.save_resume

    def interrupt(*args):
        save(*args)
        if len(args[5]) == 1:
            raise KeyboardInterrupt(f"Injected v{version} epoch-1 interruption")

    for variant in MODEL_VARIANTS:
        group = replace(cfg, model_variant=variant, experiment_name=variant)
        reference = old.run_experiment(replace(group, output_root=base / "reference"), selected, purpose="smoke")
        # v5 的已完成实验可直接用新代码评估，无需先迁移清单。
        reevaluate_experiment(reference, selected, device_name="cpu")
        with patch.object(old, "save_resume", side_effect=interrupt):
            try:
                old.run_experiment(group, selected, purpose="smoke")
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Missing injected interruption")
        root = cfg.output_root / variant
        folder = run_path(root, 500, 1, cfg.seed)
        before = torch.load(folder / "details/resume.pt", weights_only=True, map_location="cpu")["history"]
        train.run_experiment(group, selected, purpose="smoke")
        after = read_history(folder / "details/history.csv")
        for key in ("epoch", "loss", "train_acc_percent"):
            assert before[0][key] == after[0][key], "Retained history changed"
        assert after[0]["train_trials"] == 16
        for key in TIME_KEYS:
            absent = variant == "spectral_only"
            assert after[0][key] is None
            assert after[1][key] == 0 if absent else after[1][key] > 0
            assert after[1][key + "_per_trial"] == after[1][key] / 16
        totals = timing_totals([(variant, after)], 2)
        for key in TIME_KEYS:
            item = totals[key]
            if variant == "spectral_only":
                assert item["total_seconds"] == 0
            else:
                assert item["total_seconds"] is None and item["recorded_epochs"] == 1
                assert item["recorded_trials"] == 16
        header = (folder / "details/history.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "temporal_train_seconds" not in header and "monitor_seconds" not in header
        log = (folder / "training.log").read_text(encoding="utf-8")
        new_line = next(line for line in log.splitlines() if "T2 Conv Forward=" in line and "epoch=2/2" in line)
        assert "/trial" not in new_line
        assert not any(token in new_line for token in ("monitor=", "save=", "plot=", "temporal="))
        compare_runs(reference, root, cfg)
        reevaluate_experiment(root, selected, device_name="cpu")
        compare_runs(reference, root, cfg)
        manifest = json.loads((root / "details/experiment.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == 8
        archive = root / "details/source_resume"
        for name in SOURCE_NAMES:
            assert (archive / name).read_bytes() == (ROOT / name).read_bytes(), name
        result = subprocess.run([sys.executable, "-c",
                                 "import train, evaluate, experiments.ablation, utils.provenance; "
                                 "from pathlib import Path; "
                                 "assert train.PROJECT_ROOT == Path.cwd()"],
                                cwd=archive, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    with patch("train.train_epoch", side_effect=AssertionError("Completed task trained again")):
        run_ablation(cfg, MODEL_VARIANTS, selected, purpose="smoke")
    with (cfg.output_root / "comparison.csv").open(encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scope"] == "training_time_total"]
    assert {row["metric"] for row in rows} == set(TIME_KEYS)
    all_runs = [(variant, read_history(run_path(cfg.output_root / variant, 500, 1, cfg.seed) / "details/history.csv"))
                for variant in MODEL_VARIANTS]
    totals = timing_totals(all_runs, cfg.epochs)
    for row in rows:
        if row["variant"] == "all_selected":
            item = totals[row["metric"]]
            assert float(row["recorded_seconds"]) == item["recorded_seconds"]
            assert float(row["seconds_per_trial"]) == item["seconds_per_trial"]
    report = (f"# Module timing and resume: PASS\n\n{outcome.testsRun} unit checks passed.\n\n"
              "Real Benchmark S1-S3, 8 trials/subject, S1 held out, seed 42, 500 ms, 2 epochs.\n"
              f"All four genuine v{version} epoch-1 checkpoints resumed with T2 Conv/LIF forward/backward timing enabled.\n"
              f"Final weights, loss/ACC, monitor metrics and evaluation arrays match uninterrupted v{version} exactly.\n"
              "Old per-operator times remain unknown; completed tasks skipped, numerical history preserved.\n"
              "Epoch log and CSV show only four diagnostic timings; per-trial means, coverage and cross-group totals verified.\n"
              "Direct legacy and migrated v8 reevaluation passed; source archives import in a separate process.\n"
              "Unknown code changes and modified new helper modules are rejected by resume checks.\n"
              "CPU smoke only; no formal 35-fold or GPU training was run.\n")
    (base / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
