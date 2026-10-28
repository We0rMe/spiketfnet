"""读取五组服务器证据；固定完整模型权重检查融合，不训练、不覆盖原结果。"""

import csv
import hashlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from data import channel_indices, complex_spectrum, file_sha256, preprocess_subject
from evaluate import itr_bits_per_minute, run_path

GROUPS = {
    "temporal1": "temporal_v1_500ms/temporal_only",
    "temporal32": "temporal_v1_GAP32_500ms/temporal_only",
    "full1": "full-spectral_v1_500ms/full",
    "full32": "full_v1_GAP32_500ms/full",
    "spectral": "full-spectral_v1_500ms/spectral_only",
}
JOBS = [(s, seed) for s in (1, 2, 3) for seed in (42, 43)]
MODES = ("learned", "spectral_component", "temporal_component", "gate_zero", "equal", "gate_one", "source_constant")
OUTPUT = HERE / "results"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    matrix = np.asarray(values).reshape(3, 2)
    return {"mean": float(matrix.mean()), "subject_std": float(matrix.mean(1).std()),
            "subject_means": matrix.mean(1).tolist(), "seed_means": matrix.mean(0).tolist()}


def acc(logits, labels):
    return float(100 * (logits.argmax(1) == labels).mean())


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()


def authenticated_source(root, manifest, name):
    for directory in ("source_resume", "source"):
        path = root / "details" / directory / name
        if path.is_file() and file_sha256(path) == manifest["code_sha256"][name]:
            return path.read_bytes().replace(b"\r\n", b"\n")
    raise AssertionError(f"Source archive not authenticated: {root}/{name}")


def archived_builder(root, manifest):
    for name in ("model.py", "data.py"):
        assert authenticated_source(root, manifest, name) == (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    module = types.ModuleType("archived_audit_variants")
    exec(compile(authenticated_source(root, manifest, "experiments/variants.py"),
                 str(root / "details/source/experiments/variants.py"), "exec"), module.__dict__)
    return module.build_model


def saved_evidence():
    evidence = {"scope": "500 ms; S1-S3; seeds 42/43; final epoch 100", "groups": {}}
    predictions, identities, manifests = {}, {}, {}
    reference = None
    for name, relative in GROUPS.items():
        root = ROOT / "results_server" / relative
        manifest = read_json(root / "details/experiment.json")
        manifests[name] = manifest
        cfg = Config.from_dict(manifest["config"])
        for source in manifest["code_sha256"]:
            authenticated_source(root, manifest, source)
        fingerprint = (manifest["dataset_fingerprint"], manifest["selection_sha256"])
        reference = fingerprint if reference is None else reference
        assert fingerprint == reference
        rows = []
        for subject, seed in JOBS:
            folder = run_path(root, 500, subject, seed)
            private = folder / "details"
            history = list(csv.DictReader((private / "history.csv").open(encoding="utf-8-sig")))
            assert [int(row["epoch"]) for row in history] == list(range(1, 101))
            provenance = read_json(private / "provenance.json")
            assert provenance["status"] == "complete" and provenance["resumed_after_epoch"] == 0
            metrics = read_json(folder / "metrics.json")
            with np.load(private / "evaluation.npz") as saved:
                y = saved["labels"]
                np.testing.assert_array_equal(y, saved["test_ids"][:, 2])
                assert len(y) == 240 and len(saved["training_ids"]) == 8160
                assert set(saved["training_ids"][:, 0]) == set(range(1, 36)) - {subject}
                assert set(saved["test_ids"][:, 0]) == {subject}
                key = (subject, seed)
                for field in ("training_ids", "test_ids", "probe_time", "probe_spectrum"):
                    identity_key = (key, field)
                    if identity_key in identities:
                        np.testing.assert_array_equal(identities[identity_key], saved[field])
                    else:
                        identities[identity_key] = saved[field].copy()
                score = acc(saved["logits"], y)
                assert np.isclose(score, metrics["acc_percent"])
                assert np.isclose(itr_bits_per_minute(score / 100, 40, 1), metrics["itr_bits_per_minute"])
                predictions[name, key] = saved["logits"].argmax(1) == y
                row = {"subject": subject, "seed": seed, **metrics,
                       "online_train_acc": float(history[-1]["train_acc_percent"]),
                       "source_monitor_acc": float(history[-1]["monitor_acc_percent"]),
                       "evaluation_sha256": file_sha256(private / "evaluation.npz"),
                       "weight_sha256": file_sha256(private / "model.pt"),
                       "environment": provenance["timing_environment"],
                       "parameters": provenance["model_parameters"]["trainable"]}
                for branch in ("spectral", "temporal"):
                    if branch + "_logits" in saved:
                        row[branch + "_aux_acc"] = acc(saved[branch + "_logits"], y)
                if "gate" in saved:
                    g = saved["gate"]
                    sf = saved["spectral_logits"].argmax(1) == y
                    st = saved["temporal_logits"].argmax(1) == y
                    combined = saved["logits"].argmax(1) == y
                    decisions = (g > .5).sum(0)
                    row.update(gate_mean=float(g.mean()), gate_near_endpoints=float(((g < .1) | (g > .9)).mean()),
                               gate_mean_feature_std=float(g.std(0).mean()),
                               gate_fixed_selection_features=int(((decisions == 0) | (decisions == len(g))).sum()),
                               aux_oracle_acc=float(100 * (sf | st).mean()),
                               time_correct_spectral_wrong=int((~sf & st).sum()),
                               spectral_correct_time_wrong=int((sf & ~st).sum()),
                               fusion_loses_spectral_correct=int((sf & ~combined).sum()),
                               fusion_recovers_spectral_wrong=int((~sf & combined).sum()))
                rows.append(row)
        evidence["groups"][name] = {"root": str(root), "config": cfg.to_dict(), "rows": rows,
                                    "acc": stats([r["acc_percent"] for r in rows]),
                                    "itr": stats([r["itr_bits_per_minute"] for r in rows])}
    evidence["independent_complementarity"] = []
    for key in JOBS:
        sf, st = predictions["spectral", key], predictions["temporal32", key]
        evidence["independent_complementarity"].append({"subject": key[0], "seed": key[1],
            "spectral_correct": int(sf.sum()), "temporal_correct": int(st.sum()),
            "oracle_correct": int((sf | st).sum()), "time_correct_spectral_wrong": int((~sf & st).sum())})
    return evidence, manifests


def replay(evidence, manifests):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root32 = ROOT / "results_server" / GROUPS["full32"]
    cfg = replace(Config.from_dict(manifests["full32"]["config"]), data_root=Path(r"G:\SSVEP\data\benchmark"), device="cpu")
    dataset = read_json(root32 / "details/dataset.json")
    inputs = {}
    source_hashes = {}
    # S35 block1仅提供不依赖测试输入的固定门控参考；不更新任何BN或权重。
    for subject in (1, 2, 3, 35):
        path = cfg.data_root / f"S{subject}.mat"
        digest = file_sha256(path)
        assert digest == dataset["sources"][path.name]
        source_hashes[path.name] = digest
        raw = loadmat(path, variable_names=["data"])["data"]
        time = preprocess_subject(raw, cfg, channel_indices(cfg))
        del raw
        if subject == 35:
            time = time[:40]
        inputs[subject] = (torch.from_numpy(time), torch.from_numpy(complex_spectrum(time, cfg)))
        print(f"Prepared S{subject}", flush=True)
    evidence["local_replay"] = {"device": "cpu", "torch": torch.__version__, "source_reference": "S35 block1, 40 source trials; no labels",
                                "source_hashes": source_hashes, "modes": MODES, "rows": []}
    arrays = {}
    for name in ("full1", "full32"):
        root = ROOT / "results_server" / GROUPS[name]
        builder = archived_builder(root, manifests[name])
        cfg = replace(Config.from_dict(manifests[name]["config"]), device="cpu")
        for subject, seed in JOBS:
            private = run_path(root, 500, subject, seed) / "details"
            model = builder(cfg).eval()
            model.load_state_dict(torch.load(private / "model.pt", map_location="cpu", weights_only=True))
            before = state_digest(model)
            captured = {}
            handles = [getattr(model.fusion, branch).register_forward_hook(
                lambda module, arguments, output, branch=branch: captured.__setitem__(branch, output))
                for branch in ("spectral", "temporal")]
            values = {mode: [] for mode in MODES}
            with np.load(private / "evaluation.npz") as saved:
                labels, server_predictions = saved["labels"].copy(), saved["logits"].argmax(1)
                np.testing.assert_array_equal(inputs[subject][0][:4].numpy(), saved["probe_time"])
                np.testing.assert_array_equal(inputs[subject][1][:4].numpy(), saved["probe_spectrum"])
            try:
                with torch.inference_mode():
                    source_gate = model(*inputs[35])["gate"].mean(0, keepdim=True)
                    for start in range(0, 240, 128):
                        out = model(*(x[start:start + 128] for x in inputs[subject]))
                        sf, st, g = captured["spectral"], captured["temporal"], out["gate"]
                        features = {"spectral_component": (1 - g) * sf, "temporal_component": g * st,
                                    "gate_zero": sf, "equal": .5 * (sf + st), "gate_one": st,
                                    "source_constant": (1 - source_gate) * sf + source_gate * st}
                        values["learned"].append(out["logits"].numpy())
                        for mode, features_mode in features.items():
                            values[mode].append(model.fusion.classifier(features_mode).numpy())
            finally:
                for handle in handles:
                    handle.remove()
            assert before == state_digest(model)
            joined = {mode: np.concatenate(parts) for mode, parts in values.items()}
            scores = {mode: acc(v, labels) for mode, v in joined.items()}
            row = {"group": name, "subject": subject, "seed": seed, "acc": scores,
                   "cpu_gpu_disagreement_count": int((joined["learned"].argmax(1) != server_predictions).sum()),
                   "state_unchanged": True}
            evidence["local_replay"]["rows"].append(row)
            prefix = f"{name}_s{subject:02d}_seed{seed}"
            arrays[prefix + "_labels"] = labels
            arrays[prefix + "_logits"] = np.stack([joined[mode] for mode in MODES])
            arrays[prefix + "_source_gate"] = source_gate.numpy()
            print(json.dumps(row), flush=True)
            (OUTPUT / "details/evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUTPUT / "details/counterfactuals.npz", **arrays)


def main():
    (OUTPUT / "details").mkdir(parents=True, exist_ok=False)
    evidence, manifests = saved_evidence()
    (OUTPUT / "details/evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    replay(evidence, manifests)
    evidence["script_sha256"] = file_sha256(Path(__file__))
    (OUTPUT / "details/evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("full1", "full32"):
        rows = [row for row in evidence["local_replay"]["rows"] if row["group"] == name]
        print(name, {mode: stats([r["acc"][mode] for r in rows]) for mode in MODES}, flush=True)


if __name__ == "__main__":
    main()
