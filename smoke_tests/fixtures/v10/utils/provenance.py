"""源码归档清单与断点兼容校验；仅允许已验证的历史实现升级。"""

import hashlib
from pathlib import Path

from config import Config
from data import file_sha256

SOURCE_NAMES = (
    "config.py", "data.py", "model.py", "train.py", "evaluate.py", "requirements.txt",
    "experiments/__init__.py", "experiments/ablation.py", "experiments/variants.py",
    "utils/__init__.py", "utils/timing.py", "utils/provenance.py", "utils/operator_profile.py",
)

LEGACY_SOURCE_PATHS = {
    "ablation_model.py": "experiments/variants.py",
    "training_timing.py": "utils/timing.py",
    "ablation.py": "experiments/ablation.py",
}

LAYOUT_V5_SOURCES = {
    "train.py": "46b71ea474016bd476ace46127843d2e758789f320cdfecbc669cec8d4ffc061",
    "evaluate.py": "fad5da7cb6c004e695a9441f3b4c2792ab59adc7b23f8453d64116020e2cd8ce",
    "training_timing.py": "2469b7d55e27b1fc2c3bd86d041fabef5e519ccedf2ae76d1d766c504a6a9aa1",
}

LOG_FORMAT_V7_SOURCES = {
    "utils/timing.py": "87b5a7e948e2bf62669f9d5b25eba4573017030a674ffb82bc1ef827575acde1",
    "utils/provenance.py": "fda11333bed2fc636bafb399a7b7a4af8a0bfdf77aece478ee995cc5005abcd7",
}

DIAGNOSTIC_V7_SOURCES = {
    "train.py": "2324625648995eec73635e72d2effab1b6e224c5f1da23169ae962882c0a37bc",
    "utils/timing.py": "41b340edd8f7f70091794ee816936f1ac7aed38093f910a8ac9dec676b2b6622",
    "utils/provenance.py": "04024bb5e03db3d48c265049f01f652d9ebda8a5fb0886a0c085385f7b2e67bd",
}

DETAILED_TIMING_V8_SOURCES = {
    "train.py": "38a412ff2bd52d6435c043d88f50575041c5d7e0e22091ee38a3795493b44348",
    "evaluate.py": "48ed758d879bfd34999f3da8326599c80248117809d6b8de17a1f6a8e55c450f",
    "utils/timing.py": "67d42dba38ae28db22e11a1b0bdaed74e6c30fd57f24a7654a3d04622f4a655d",
    "utils/provenance.py": "48242eab8116e5c0774182e4cf060c142f18a91a71977883416087ef99ccdf09",
}

CONV_EXECUTION_V9_SOURCES = {
    "model.py": "4d93fe4b52ddf33ffbf8127dff145c81351dddb47a51e7e4a04d86eb402e430c",
    "train.py": "d43318d26a22b292569c09313be1e39abd1cdf08793381c82d73ddba14da7318",
    "evaluate.py": "f1c83eac7c55514f5b2f0ffc20f085c47a1975457268d34912ca23112d8b77ab",
    "utils/operator_profile.py": "6008602d2e561c40cfda59c5681368154c708dc9ea4ce366d086ebf90e0af66a",
    "utils/provenance.py": "50d3fa49dee2a620c484e5917407492a2d68f4ba459ceab4149066442af611d8",
}

MODULE_TIMING_V6_SOURCES = {
    "train.py": "35ed6c3c74f353c48e84148289d042680278671bdd29adf3f8379e3e1f619ac6",
    "evaluate.py": "515ceb814c0f8f73c623de0b02be728a27a4f06ac71846c1ca563cafb75ec300",
    "utils/timing.py": "2469b7d55e27b1fc2c3bd86d041fabef5e519ccedf2ae76d1d766c504a6a9aa1",
    "utils/provenance.py": "b3c5d595ed225a9912665f38753d35184ab5faffc16ce3749067a620fdfc82a5",
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
               "resume", "check_gradients", "variance_ddof", "monitor_every", "plot_every", "time_branches",
               "t2_profile_batch"}
    differences = [key for key, value in cfg.to_dict().items() if key not in allowed and previous[key] != value]
    if differences:
        raise ValueError(f"Resume would change training/model settings: {differences}; use a new experiment_name")
    if manifest["purpose"] != purpose or manifest["dataset_fingerprint"] != prepared.metadata["fingerprint"]:
        raise ValueError("Resume dataset fingerprint or experiment purpose differs")
    selection_hash = hashlib.sha256(prepared.ids.tobytes()).hexdigest()
    if manifest["selected_trial_count"] != len(prepared.ids) or manifest.get("selection_sha256", selection_hash) != selection_hash:
        raise ValueError("Resume trial selection differs")
    # 历史升级需认证源码；旧模型仅允许进入逐位保持原路径的 native 模式。
    checked_sources = ("model.py", "data.py") if manifest.get("format_version", 1) == 1 else (
        "model.py", "data.py", "train.py", "evaluate.py")
    version = manifest.get("format_version", 1)
    if version >= 9:
        checked_sources += ("utils/operator_profile.py",)
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
        if version <= 9 and cfg.t2_conv_execution == "native" and (version == 9 or name == "model.py") and verified_source_upgrade(
                name, previous_hash, details, CONV_EXECUTION_V9_SOURCES):
            print(f"Verified v{version} native-path upgrade: {name}; original convolution execution preserved.", flush=True)
            continue
        # evaluate.py 在 v7/v8 相同，其他文件只接受各自版本的已冻结内容。
        if version in (7, 8) and (version == 8 or name == "evaluate.py") and verified_source_upgrade(
                name, previous_hash, details, DETAILED_TIMING_V8_SOURCES):
            print(f"Verified v{version} fine-grained timing upgrade: {name}; training semantics preserved.", flush=True)
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
        if version == 6 and verified_source_upgrade(name, previous_hash, details, MODULE_TIMING_V6_SOURCES):
            print(f"Verified v6 module-timing upgrade: {name}; old module times remain unknown.", flush=True)
            continue
        if version == 7 and verified_source_upgrade(name, previous_hash, details, LOG_FORMAT_V7_SOURCES):
            print(f"Verified v7 log-format upgrade: {name}; training and timing calculations preserved.", flush=True)
            continue
        if version == 7 and verified_source_upgrade(name, previous_hash, details, DIAGNOSTIC_V7_SOURCES):
            print(f"Verified v7 T2 diagnostic upgrade: {name}; old operator times remain unknown.", flush=True)
            continue
        raise ValueError(f"Resume source code differs: {name}; unverified changes cannot resume this experiment")
