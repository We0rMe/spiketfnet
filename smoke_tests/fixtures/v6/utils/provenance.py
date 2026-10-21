"""源码归档清单与断点兼容校验；仅允许已验证的历史实现升级。"""

import hashlib
from pathlib import Path

from config import Config
from data import file_sha256

SOURCE_NAMES = (
    "config.py", "data.py", "model.py", "train.py", "evaluate.py", "requirements.txt",
    "experiments/__init__.py", "experiments/ablation.py", "experiments/variants.py",
    "utils/__init__.py", "utils/timing.py", "utils/provenance.py",
)

LEGACY_SOURCE_PATHS = {
    "ablation_model.py": "experiments/variants.py",
    "training_timing.py": "utils/timing.py",
    "ablation.py": "experiments/ablation.py",
}

LAYOUT_V5_SOURCES = {
    "train.py": "46b71ea474016bd476ace46127843d2e758789f320cdfecbc669cec8d4ffc061",
    "evaluate.py": "fad5da7cb6c004e695a9441f3b4c2792ab59adc7b23f8453d64116020e2cd8ce",
}


REPORTING_V2_SOURCES = {
    "train.py": "67e00a4b9018a5be210cd3967760c6203cee4055668bd4569ad98ef65fd40704",
    "evaluate.py": "1c800e8cc51dfae59ce18ef64af81150011805afceaae5d1e5ed318409cb36dd",
}

ABLATION_V3_SOURCES = {
    "train.py": "4fff6ebc49e0b31e23dbccbe598e4a2bac1fb5390b75a361f042431646144edc",
    "evaluate.py": "dca51e7f17af5f882844a8f9156aca5165d666f311e12b7feae9b347ff9423c3",
}

TIMING_V4_SOURCES = {
    "train.py": "8f9a556c95282f7d1b72fa7097cce9cd8f4d4c67be245e96776df9aca5e0bfc5",
    "evaluate.py": "aeb04d511ac52fb92c5b76ed4869c0ae836e7ca40379bf66b85ef6ccec434f15",
}


def verified_reporting_upgrade(name, old_hash, details):
    return verified_source_upgrade(name, old_hash, details, REPORTING_V2_SOURCES)


def verified_source_upgrade(name, old_hash, details, approved):
    if details is None or name not in approved:
        return False
    for directory in ("source_resume", "source"):
        source = Path(details) / directory / name
        if source.is_file() and file_sha256(source) == old_hash:
            normalized = source.read_bytes().replace(b"\r\n", b"\n")
            return hashlib.sha256(normalized).hexdigest() == approved[name]
    return False


def validate_resume(manifest, cfg, prepared, purpose, code_hashes, details=None):
    previous = Config.from_dict(manifest["config"]).to_dict()
    allowed = {"N", "M", "seed", "windows_ms", "device", "gpu_ids", "evaluation_device", "num_workers",
               "cpu_threads", "data_root", "cache_root", "output_root", "experiment_name",
               "resume", "check_gradients", "variance_ddof", "monitor_every", "plot_every", "time_branches"}
    differences = [key for key, value in cfg.to_dict().items() if key not in allowed and previous[key] != value]
    if differences:
        raise ValueError(f"Resume would change training/model settings: {differences}; use a new experiment_name")
    if manifest["purpose"] != purpose or manifest["dataset_fingerprint"] != prepared.metadata["fingerprint"]:
        raise ValueError("Resume dataset fingerprint or experiment purpose differs")
    selection_hash = hashlib.sha256(prepared.ids.tobytes()).hexdigest()
    if manifest["selected_trial_count"] != len(prepared.ids) or manifest.get("selection_sha256", selection_hash) != selection_hash:
        raise ValueError("Resume trial selection differs")
    # 旧版可升级训练/评估基础设施；网络和预处理代码必须完全相同。
    checked_sources = ("model.py", "data.py") if manifest.get("format_version", 1) == 1 else (
        "model.py", "data.py", "train.py", "evaluate.py")
    version = manifest.get("format_version", 1)
    if version >= 6:
        checked_sources += ("experiments/variants.py", "experiments/__init__.py",
                            "utils/timing.py", "utils/provenance.py", "utils/__init__.py")
    elif version >= 4:
        checked_sources += ("ablation_model.py",)
    elif cfg.model_variant != "full":
        raise ValueError("Legacy experiments can only resume as full models")
    if version == 5:
        checked_sources += ("training_timing.py",)
    for name in checked_sources:
        previous_hash = manifest["code_sha256"][name]
        current_name = LEGACY_SOURCE_PATHS.get(name, name) if version < 6 else name
        if previous_hash == code_hashes[current_name]:
            continue
        if manifest.get("format_version") == 2 and verified_reporting_upgrade(name, previous_hash, details):
            print(f"Verified reporting-only v2 upgrade: {name}; training semantics preserved.", flush=True)
            continue
        if manifest.get("format_version") == 3 and verified_source_upgrade(
                name, previous_hash, details, ABLATION_V3_SOURCES):
            print(f"Verified v3 full-model upgrade: {name}; original model and training preserved.", flush=True)
            continue
        if manifest.get("format_version") == 4 and verified_source_upgrade(
                name, previous_hash, details, TIMING_V4_SOURCES):
            print(f"Verified v4 timing upgrade: {name}; old branch times remain unknown.", flush=True)
            continue
        if version == 5 and verified_source_upgrade(name, previous_hash, details, LAYOUT_V5_SOURCES):
            print(f"Verified v5 directory-layout upgrade: {name}; training semantics preserved.", flush=True)
            continue
        raise ValueError(f"Resume source code differs: {name}; unverified changes cannot resume this experiment")
