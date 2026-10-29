"""核对等权/单频域对照，并固定等权模型权重检查两条最终分类路径。"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

import run as audit


MODES = ("final", "spectral_aux", "temporal_aux", "spectral_path", "temporal_path", "aux_equal")


def main():
    audit.GROUPS.update(equal32="equal-spectral_v1_GAP32_500ms/equal_fusion",
                        spectral32="equal-spectral_v1_GAP32_500ms/spectral_only")
    evidence, manifests = audit.saved_evidence()
    evidence["controls"] = {"new_groups": ["equal32", "spectral32"], "replay": []}
    base = manifests["full32"]
    for group in ("equal32", "spectral32"):
        manifest = manifests[group]
        differences = {key for key, value in manifest["config"].items() if value != base["config"][key]}
        assert differences == {"model_variant", "experiment_name", "output_root"}
        assert manifest["environment"] == base["environment"]
        changed = [name for name in manifest["code_sha256"] if audit.authenticated_source(
            Path(evidence["groups"][group]["root"]), manifest, name) != audit.authenticated_source(
                Path(evidence["groups"]["full32"]["root"]), base, name)]
        assert changed == ["config.py"]
    evidence["controls"].update(validated_models=42, only_source_change_from_full32="config.py",
                                dataset_fingerprint=base["dataset_fingerprint"],
                                selection_sha256=base["selection_sha256"])
    for name, group in evidence["groups"].items():
        root = Path(group["root"])
        for row in group["rows"]:
            path = audit.run_path(root, 500, row["subject"], row["seed"]) / "details/evaluation.npz"
            with np.load(path) as saved:
                labels = saved["labels"]
                final = saved["logits"].argmax(1) == labels
                if "spectral_logits" in saved:
                    spectral = saved["spectral_logits"].argmax(1) == labels
                    row["fusion_loses_spectral_correct"] = int((spectral & ~final).sum())
                    row["fusion_recovers_spectral_wrong"] = int((~spectral & final).sum())
                if "spectral_logits" in saved and "temporal_logits" in saved:
                    temporal = saved["temporal_logits"].argmax(1) == labels
                    row["aux_oracle_acc"] = float(100 * (spectral | temporal).mean())
                    row["aux_equal_acc"] = audit.acc(
                        .5 * (saved["spectral_logits"] + saved["temporal_logits"]), labels)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path(evidence["groups"]["equal32"]["root"])
    manifest = manifests["equal32"]
    builder = audit.archived_builder(root, manifest)
    cfg = replace(audit.Config.from_dict(manifest["config"]),
                  data_root=Path(r"G:\SSVEP\data\benchmark"), device="cpu")
    dataset = audit.read_json(root / "details/dataset.json")
    inputs = {}
    for subject in (1, 2, 3, 35):
        path = cfg.data_root / f"S{subject}.mat"
        assert audit.file_sha256(path) == dataset["sources"][path.name]
        raw = loadmat(path, variable_names=["data"])["data"]
        time = audit.preprocess_subject(raw, cfg, audit.channel_indices(cfg))
        inputs[subject] = (torch.from_numpy(time), torch.from_numpy(audit.complex_spectrum(time, cfg)))
        print(f"Prepared S{subject}", flush=True)

    arrays = {}
    for subject, seed in audit.JOBS:
        private = audit.run_path(root, 500, subject, seed) / "details"
        model = builder(cfg).eval()
        model.load_state_dict(torch.load(private / "model.pt", map_location="cpu", weights_only=True))
        before = audit.state_digest(model)
        captured = {}
        handles = [getattr(model.fusion, branch).register_forward_hook(
            lambda module, args, output, branch=branch: captured.__setitem__(branch, output))
            for branch in ("spectral", "temporal")]
        with np.load(private / "evaluation.npz") as saved:
            labels = saved["labels"].copy()
            server_predictions = saved["logits"].argmax(1)
            source_labels = saved["training_ids"][saved["training_ids"][:, 0] == 35, 2].copy()
            np.testing.assert_array_equal(inputs[subject][0][:4].numpy(), saved["probe_time"])
            np.testing.assert_array_equal(inputs[subject][1][:4].numpy(), saved["probe_spectrum"])
        row = {"subject": subject, "seed": seed}
        try:
            with torch.inference_mode():
                for split, input_subject, y in (("target", subject, labels), ("source_S35", 35, source_labels)):
                    pieces = {mode: [] for mode in MODES}
                    scales = {branch: [] for branch in ("spectral", "temporal")}
                    for start in range(0, len(y), 128):
                        outputs = model(*(x[start:start + 128] for x in inputs[input_subject]))
                        spectral = model.fusion.classifier(captured["spectral"])
                        temporal = model.fusion.classifier(captured["temporal"])
                        torch.testing.assert_close(.5 * (spectral + temporal), outputs["logits"],
                                                   atol=1e-5, rtol=1e-5)
                        values = (outputs["logits"], outputs["spectral_logits"], outputs["temporal_logits"],
                                  spectral, temporal, .5 * (outputs["spectral_logits"] + outputs["temporal_logits"]))
                        for mode, value in zip(MODES, values):
                            pieces[mode].append(value.numpy())
                        for branch, value in (("spectral", spectral), ("temporal", temporal)):
                            scales[branch].append(value.std(dim=1, unbiased=False).numpy())
                    joined = {key: np.concatenate(value) for key, value in pieces.items()}
                    row[split] = {"acc": {key: audit.acc(value, y) for key, value in joined.items()},
                                  "mean_logit_std": {k: float(np.concatenate(v).mean()) for k, v in scales.items()}}
                    prefix = f"s{subject:02d}_seed{seed}_{split}"
                    arrays[prefix + "_labels"] = y
                    arrays[prefix + "_logits"] = np.stack([joined[mode] for mode in MODES])
                    if split == "target":
                        row["cpu_gpu_disagreement_count"] = int(
                            (joined["final"].argmax(1) != server_predictions).sum())
        finally:
            for handle in handles:
                handle.remove()
        assert before == audit.state_digest(model)
        row["state_unchanged"] = True
        evidence["controls"]["replay"].append(row)
        print(json.dumps(row), flush=True)

    evidence["controls"].update(
        device="cpu", torch=torch.__version__, modes=MODES, source_subject=35,
        scope="Fixed weights and BN; source S35 is seen training data, not validation; no fitting or coefficient search",
        raw_sha256={f"S{s}.mat": dataset["sources"][f"S{s}.mat"] for s in inputs},
        script_sha256=audit.file_sha256(Path(__file__)))
    evidence["reference_groups"] = {k: v["root"] for k, v in evidence["groups"].items()
                                    if k not in ("equal32", "spectral32")}
    evidence["groups"] = {k: v for k, v in evidence["groups"].items() if k in ("equal32", "spectral32")}
    del evidence["independent_complementarity"]  # 已保存在上一轮证据中，不重复存储。
    output = audit.OUTPUT / "details"
    np.savez_compressed(output / "control_replay.npz", **arrays)
    (output / "control_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    for split in ("target", "source_S35"):
        print(split, {mode: audit.stats([r[split]["acc"][mode]
              for r in evidence["controls"]["replay"]]) for mode in MODES}, flush=True)


if __name__ == "__main__":
    main()
