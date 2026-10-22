"""目录迁移的源码映射、可信升级和新辅助模块校验。"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import CONFIG, PROJECT_ROOT
from data import PreparedData, file_sha256
from utils.provenance import (LAYOUT_V5_SOURCES, LEGACY_SOURCE_PATHS, MODULE_TIMING_V6_SOURCES, SOURCE_NAMES,
                              DETAILED_TIMING_V8_SOURCES, validate_resume, verified_source_upgrade)


class LayoutTests(unittest.TestCase):
    def test_source_list_contains_real_files_and_preserves_moved_code(self):
        self.assertEqual(len(SOURCE_NAMES), len(set(SOURCE_NAMES)))
        for name in SOURCE_NAMES:
            self.assertTrue((PROJECT_ROOT / name).is_file(), name)
        expected = {
            "experiments/variants.py": "d85a5f6c074f4579f0b949b1920a3bb20b8dd5c0d1542fbb79df3c4b3ef7f80e",
        }
        for name, digest in expected.items():
            content = (PROJECT_ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_v5_upgrade_requires_authenticated_archive(self):
        results = PROJECT_ROOT / "smoke_tests/results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temp:
            details = Path(temp)
            source = details / "source"
            source.mkdir()
            for name in LAYOUT_V5_SOURCES:
                fixture = ("v6/utils/timing.py" if name == "training_timing.py" else name.replace(".py", "_v5.py"))
                raw = (PROJECT_ROOT / "smoke_tests/fixtures" / fixture).read_bytes()
                for ending in (b"\n", b"\r\n"):
                    (source / name).write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", ending))
                    self.assertTrue(verified_source_upgrade(name, file_sha256(source / name), details,
                                                            LAYOUT_V5_SOURCES))
                    self.assertFalse(verified_source_upgrade(name, "incorrect", details, LAYOUT_V5_SOURCES))
                (source / name).write_bytes(raw + b"\n# unknown change\n")
                self.assertFalse(verified_source_upgrade(name, file_sha256(source / name), details,
                                                         LAYOUT_V5_SOURCES))

    def test_legacy_mapping_and_new_helpers_are_checked(self):
        prepared = PreparedData(np.zeros((2, 9, 250)), np.array([[1, 1, 0], [2, 1, 0]]),
                                np.arange(40), np.arange(40), {"fingerprint": "example"})
        hashes = {name: name for name in SOURCE_NAMES}
        manifest = {"format_version": 9, "config": CONFIG.to_dict(), "purpose": "smoke",
                    "dataset_fingerprint": "example", "selected_trial_count": 2, "code_sha256": hashes}
        validate_resume(manifest, CONFIG, prepared, "smoke", hashes)
        for name in ("utils/provenance.py", "utils/timing.py", "utils/operator_profile.py", "experiments/variants.py",
                     "experiments/__init__.py", "utils/__init__.py"):
            with self.assertRaisesRegex(ValueError, "Resume source code differs"):
                validate_resume(manifest, CONFIG, prepared, "smoke", {**hashes, name: "changed"})
        legacy_hashes = {old: hashes[new] for old, new in LEGACY_SOURCE_PATHS.items()}
        manifest = {**manifest, "format_version": 5, "code_sha256": {**hashes, **legacy_hashes}}
        validate_resume(manifest, CONFIG, prepared, "smoke", hashes)
        with self.assertRaisesRegex(ValueError, "training_timing.py"):
            validate_resume(manifest, CONFIG, prepared, "smoke", {**hashes, "utils/timing.py": "changed"})

    def test_v6_authentication_covers_changed_helpers(self):
        results = PROJECT_ROOT / "smoke_tests/results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temp:
            details = Path(temp)
            for name in MODULE_TIMING_V6_SOURCES:
                path = details / "source" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = (PROJECT_ROOT / "smoke_tests/fixtures/v6" / name).read_bytes()
                for ending in (b"\n", b"\r\n"):
                    path.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", ending))
                    self.assertTrue(verified_source_upgrade(name, file_sha256(path), details, MODULE_TIMING_V6_SOURCES))
                path.write_bytes(raw + b"\n# unverified edit\n")
                self.assertFalse(verified_source_upgrade(name, file_sha256(path), details, MODULE_TIMING_V6_SOURCES))

    def test_v8_authentication_covers_changed_files(self):
        base = PROJECT_ROOT / "smoke_tests/results"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            details = Path(temp)
            for name in DETAILED_TIMING_V8_SOURCES:
                path = details / "source" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = (PROJECT_ROOT / "smoke_tests/fixtures/v8" / name).read_bytes()
                for ending in (b"\n", b"\r\n"):
                    path.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", ending))
                    self.assertTrue(verified_source_upgrade(name, file_sha256(path), details, DETAILED_TIMING_V8_SOURCES))
                path.write_bytes(raw + b"\n# unknown edit\n")
                self.assertFalse(verified_source_upgrade(name, file_sha256(path), details, DETAILED_TIMING_V8_SOURCES))
