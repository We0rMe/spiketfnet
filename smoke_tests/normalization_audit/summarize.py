"""独立从逐试次预测复算指标，核对权重/源码，再生成唯一可读报告。"""

import json
from pathlib import Path

import numpy as np
import torch

from run import (CORE, EXPERIMENT, MODES, OUTPUT, ROOT, SEEDS, SUBJECTS, Config, Moments,
                 apply_statistics, build_model, digest_tensors, file_sha256, forward,
                 itr_bits_per_minute, score)


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    details = OUTPUT / "details"
    evidence = json.loads((details / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["checks"]["all_completed"] and len(evidence["runs"]) == 14
    assert evidence["plan"]["core_sha256"] == {name: file_sha256(ROOT / name) for name in CORE}
    assert evidence["plan"]["runner_sha256"] == file_sha256(Path(__file__).with_name("run.py"))
    statistics = torch.load(details / "statistics.pt", weights_only=True)
    matrices = {mode: {split: np.empty((7, 2)) for split in ("monitor", "test", "itr")} for mode in MODES}
    server_acc, discrepancies, checked = [], [], 0
    with np.load(details / "predictions.npz") as archive:
        for i, subject in enumerate(SUBJECTS):
            for j, seed in enumerate(SEEDS):
                key = f"S{subject:02d}_seed{seed}"
                row = evidence["runs"][key]
                original = EXPERIMENT / f"window_0500ms/test_S{subject:02d}/seed_{seed}/details"
                assert file_sha256(original / "model.pt") == row["checkpoint_sha256"]
                assert row["calibration_subjects"] == sorted(set(range(1, 36)) - {subject})
                for name, stats in statistics[key].items():
                    assert stats["count_per_step"] == 8160 * 125
                    assert stats["step_mean"].shape[0] == (1 if name == "temporal.current.1" else 4)
                    for field in ("mean", "var", "step_mean", "step_var"):
                        assert torch.isfinite(stats[field]).all()
                    assert torch.all(stats["var"] >= 0) and torch.all(stats["step_var"] >= 0)
                for split, size in (("monitor", 1360), ("test", 240)):
                    ids = archive[f"{key}_{split}_ids"]
                    assert len(ids) == size and len(np.unique(ids, axis=0)) == size
                    assert (np.all(ids[:, 0] == subject) if split == "test" else
                            np.all(ids[:, 0] != subject) and np.all(ids[:, 1] == 1))
                    for mode in MODES:
                        logits = archive[f"{key}_{mode}_{split}_logits"]
                        acc = float((logits.argmax(1) == ids[:, 2]).mean())
                        itr = itr_bits_per_minute(acc, 40, 1.)
                        assert abs(acc * 100 - row[mode][split]["acc_percent"]) < 1e-9
                        assert abs(itr - row[mode][split]["itr_bits_per_minute"]) < 1e-9
                        matrices[mode][split][i, j] = 100 * acc
                        if split == "test":
                            matrices[mode]["itr"][i, j] = itr
                        checked += 1
                with np.load(original / "evaluation.npz") as old:
                    assert np.array_equal(archive[f"{key}_test_ids"], old["test_ids"])
                    server_acc.append(float((old["logits"].argmax(1) == old["labels"]).mean() * 100))
                    discrepancies.append(row["cpu_server_replay"]["prediction_disagreements"])

        # 从保存统计和原权重重建第一折三种推理，检查产物可重放。
        cfg = Config.from_dict(json.loads((EXPERIMENT / "details/experiment.json").read_text(encoding="utf-8"))["config"])
        model = build_model(cfg).eval()
        model.load_state_dict(torch.load(EXPERIMENT / "window_0500ms/test_S01/seed_42/details/model.pt",
                                        weights_only=True, map_location="cpu"))
        with np.load(details / "cache/benchmark_prepared.npz") as prepared:
            target = prepared["ids"][:, 0] == 1
            x = torch.from_numpy(prepared["time"][target])
            y = prepared["ids"][target, 2]
        for mode in MODES:
            net = model if mode == "original" else apply_statistics(
                model, statistics["S01_seed42"], mode == "per_step_reestimated")
            _, logits = score(net, x, y, cfg)
            np.testing.assert_array_equal(logits, archive[f"S01_seed42_{mode}_test_logits"])

    # 独立验证合并方差公式，覆盖末批大小不同的情况。
    x = torch.randn(19, 4, 13, generator=torch.Generator().manual_seed(7))
    moment = Moments()
    for part in (x[:8], x[8:16], x[16:]):
        var, mean = torch.var_mean(part, dim=(0, 2), unbiased=False)
        moment.add(len(part) * 13, mean, var)
    mean, var = moment.tensors()
    torch.testing.assert_close(mean, x.mean((0, 2)), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(var, x.var((0, 2), unbiased=False), rtol=1e-6, atol=1e-7)
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for metric, matrix in matrices[mode].items():
            means = matrix.mean(1)
            summary[mode][metric] = {"mean": float(means.mean()), "subject_std_ddof0": float(means.std()),
                                     "seed_means": matrix.mean(0).tolist()}
        delta = matrices[mode]["test"] - matrices["original"]["test"]
        summary[mode]["test_delta"] = {"mean_pp": float(delta.mean()),
                                        "improved_models": int((delta > 1e-9).sum()),
                                        "unchanged_models": int((np.abs(delta) <= 1e-9).sum()),
                                        "improved_subjects_after_seed_average": int((delta.mean(1) > 1e-9).sum())}
    evidence["summary"] = summary
    evidence["checks"].update(metrics_recomputed_from_logits=checked,
                              first_model_all_modes_reloaded_logits_exact=True,
                              weighted_population_moments_verified=True,
                              all_source_ids_exclude_own_target=True,
                              all_checkpoints_and_production_sources_unchanged=True)
    evidence["backend_comparison"] = {"server_mean_acc": float(np.mean(server_acc)),
                                       "cpu_mean_acc": summary["original"]["test"]["mean"],
                                       "prediction_disagreements_per_240_min_max": [min(discrepancies), max(discrepancies)],
                                       "prediction_disagreement_percent": float(np.mean(discrepancies) / 240 * 100)}
    evidence["artifact_sha256"] = {name: file_sha256(details / name) for name in ("predictions.npz", "statistics.pt")}
    evidence["verifier_sha256"] = file_sha256(__file__)
    (details / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    names = dict(original="原统计（本地CPU）", shared_reestimated="重估统计，仍由各步共享",
                 per_step_reestimated="重估统计，各仿真步分开保存")
    lines = ["# 500毫秒模型：固定权重的归一化统计审查", "",
             "范围：S1–S7，每折种子42、43，共14个已训练模型；没有重新训练，没有修改正式代码。",
             "每折仅使用其他34名被试的8160条源训练输入估计统计量，不使用标签；目标被试从未参与估计。",
             "监测使用已经参与训练的源被试block1（1360条），测试使用留出被试全部240条。", "",
             "| 方式 | 训练监测ACC | 测试ACC | 测试ITR | 测试ACC较本地原统计变化 |",
             "|---|---:|---:|---:|---:|"]
    for mode, result in summary.items():
        cells = [f"{result[m]['mean']:.2f} ± {result[m]['subject_std_ddof0']:.2f}" for m in ("monitor", "test", "itr")]
        lines.append(f"| {names[mode]} | " + " | ".join(cells) + f" | {result['test_delta']['mean_pp']:+.2f}个百分点 |")
    lines += ["", "表中先平均每名被试的两个种子，再计算7名被试的均值±总体标准差；不是方差或置信区间。ITR先逐模型计算。",
              "训练监测属于已见数据；与测试差距不能单凭此表归结为过拟合。", "",
              "## 逐模型ACC", "", "| 被试/种子 | 原监测→共享→按步 | 原测试→共享→按步 |", "|---|---:|---:|"]
    for key, row in evidence["runs"].items():
        cells = [" → ".join(f"{row[mode][split]['acc_percent']:.2f}" for mode in MODES) for split in ("monitor", "test")]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    b = evidence["backend_comparison"]
    lines += ["", "## 结论与边界", "",
              "固定权重后，源训练数据的识别明显改善，而测试改善相对有限。原统计量确实压低了已见数据的评估成绩，但本次重估不能解释或消除与论文的主要差距。",
              "按仿真步保存统计在这里属于额外推理诊断，不等于已经验证按步BN重新训练会更好。两种方式均完整报告，不按目标分数挑选配置。",
              "下一步优先验证保留时间分段信息的汇总方式；不因源训练监测提高就直接推广正式配置。", "",
              f"服务器原测试均值{b['server_mean_acc']:.2f}%，本地CPU原统计均值{b['cpu_mean_acc']:.2f}%。两者有{b['prediction_disagreement_percent']:.2f}%的逐试次预测不同，不能将本地校准分数直接替换服务器指标。",
              "输入原文件哈希、预处理参数、归档代码和保存的代表性输入已核对；源码原始哈希差异仅来自LF/CRLF换行。CPU/GPU存在数值后端差异，脉冲阈值计算可能放大差异，具体来源本次未隔离。所有校准收益均采用同一CPU上的配对比较，服务器GPU效果仍需复核。", "",
              "## 方法与检查", "",
              "校准时关闭所有Dropout，仅让BN使用当前源批次的统计进行前向；按实际元素数量合并总体均值/方差，包含末批96条。校准batch128、顺序种子20261026预先固定。深层BN各仿真步分别累计，再合并或分步保存；没有优化器、反向传播或权重更新。",
              "校准结果依赖源批次归一化的前向分布，不是对所有归一化方案的穷举；效果有限只能否定本次重估足以解决问题，不能排除其他归一化训练方案。",
              f"检查通过：{checked}组ACC/ITR从保存预测独立复算；全部权重和正式源码未改；评估期间统计冻结；第一份模型三种推理从保存统计重放得到完全相同logits；不等批量总体方差合并公式与直接计算一致。",
              "details仅保存evidence.json、predictions.npz和statistics.pt，不重复保存模型权重或原始数据。"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "backend": b, "verified_metric_groups": checked}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
