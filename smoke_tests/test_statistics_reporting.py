"""统计轴、单观测缺项，以及只重建报告时的数据保护。"""

import csv
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import CONFIG, PROJECT_ROOT
from data import file_sha256
from evaluate import aggregate_matrix, refresh_saved_reports, run_path, write_json
from utils.provenance import verified_line_endings


class ReportingTests(unittest.TestCase):
    def test_subject_variation_is_distinct_from_seed_variation(self):
        one_seed = aggregate_matrix([[20], [60]], 0)
        self.assertEqual(one_seed["across_subjects"], {"count": 2, "mean": 40., "std": 20.})
        self.assertIsNone(one_seed["across_repeats"]["std"])
        repeats = aggregate_matrix([[20, 40], [60, 80]], 0)
        self.assertEqual(repeats["across_subjects"]["std"], 20.)
        self.assertEqual(repeats["across_repeats"]["std"], 10.)
        self.assertEqual(aggregate_matrix([[20], [20]], 0)["across_subjects"]["std"], 0.)
        self.assertAlmostEqual(aggregate_matrix([[20], [60]], 1)["across_subjects"]["std"], 800 ** 0.5)

    def test_refresh_saved_reports_uses_metrics_without_training_or_inference(self):
        output = PROJECT_ROOT / "smoke_tests/results"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            root = Path(temporary)
            cfg = replace(CONFIG, N=2, M=1, windows_ms=(500,), epochs=2, variance_ddof=0)
            write_json(root / "details/experiment.json", {"config": cfg.to_dict()})
            protected = []
            for subject, acc in ((1, 20.), (2, 60.)):
                folder = run_path(root, 500, subject, cfg.seed)
                write_json(folder / "details/provenance.json", dict(status="complete", test_subject=subject,
                           seed=cfg.seed, window_ms=500, training_subjects=[3, 4], training_trials=8, test_trials=10))
                write_json(folder / "metrics.json", dict(acc_percent=acc, itr_bits_per_minute=acc * 2))
                (folder / "training.log").write_text("original log\n", encoding="utf-8")
                (folder / "details/resume.pt").write_bytes(b"not a checkpoint: must not load")
                protected.extend(folder.rglob("*.*"))
            protected.append(root / "details/experiment.json")
            original = {p: p.read_bytes() for p in protected}
            console = io.StringIO()
            with redirect_stdout(console), patch("evaluate.evaluate_model", side_effect=AssertionError("inference")), \
                    patch("evaluate.prepare_data", side_effect=AssertionError("EEG access")):
                refresh_saved_reports(root)
            self.assertIn("40.000000 ± 20.000000", console.getvalue())
            report = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("40.00 ± 20.00", report)
            self.assertIn("80.00 ± 40.00", report)
            self.assertIn("种子间标准差为 NA", report)
            self.assertNotIn("T2 Conv", report)
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)
            fold = (run_path(root, 500, 1, cfg.seed) / "report.md").read_text(encoding="utf-8")
            self.assertIn("测试 ACC | 20.0000%", fold)
            self.assertIn("测试 ITR | 40.0000", fold)
            with (root / "summary.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("mean_plus_std", rows[0])
            self.assertNotIn("variance", rows[0])
            self.assertNotIn("mean_plus_variance", rows[0])
            self.assertTrue(all(row["std"] == "" for row in rows if row["scope"] == "across_repeats"))

    def test_line_ending_compatibility_requires_authentic_identical_source(self):
        output = PROJECT_ROOT / "smoke_tests/results"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            details = Path(temporary)
            source = details / "source/model.py"
            source.parent.mkdir()
            content = (PROJECT_ROOT / "model.py").read_bytes().replace(b"\r\n", b"\n")
            source.write_bytes(content)
            old_hash = hashlib.sha256(content).hexdigest()
            current_hash = file_sha256(PROJECT_ROOT / "model.py")
            self.assertTrue(verified_line_endings("model.py", "model.py", old_hash, current_hash, details))
            self.assertFalse(verified_line_endings("model.py", "model.py", "bad", current_hash, details))
            self.assertFalse(verified_line_endings("model.py", "model.py", old_hash, "bad", details))
            source.write_bytes(content + b"\n# changed\n")
            self.assertFalse(verified_line_endings("model.py", "model.py", file_sha256(source), current_hash, details))
