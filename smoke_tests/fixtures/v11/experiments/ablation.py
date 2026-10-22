"""独立消融入口；复用训练流程，按组顺序运行、组内按 GPU 并行 LOSO。"""

import csv
import json
from dataclasses import replace

import numpy as np

from config import ABLATION_CONFIG, ABLATION_GROUPS, PROJECT_ROOT
from data import file_sha256, prepare_data
from evaluate import aggregate_matrix, format_stat, read_history, run_path, write_text
from train import experiment_lock, run_experiment
from utils.provenance import SOURCE_NAMES, validate_resume
from utils.branch_timing import timing_markdown, timing_rows, timing_totals


def summarize_ablation(cfg, variants):
    """只汇总本次选择的完整组；差值逐被试/种子配对后再统计。"""
    rows = []
    time_rows, all_runs = [], []
    group_runs = {variant: [] for variant in variants}
    for window in cfg.windows_ms:
        matrices = {}
        for variant in variants:
            window_runs = []
            metrics = {key: np.empty((cfg.N, cfg.M)) for key in ("acc_percent", "itr_bits_per_minute")}
            complete = True
            for i, subject in enumerate(cfg.test_subjects):
                for j, seed in enumerate(cfg.seeds):
                    folder = run_path(cfg.output_root / variant, window, subject, seed)
                    window_runs.append((variant, read_history(folder / "details/history.csv")))
                    path = folder / "details/provenance.json"
                    provenance = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                    if provenance.get("status") != "complete":
                        complete = False
                        continue
                    if (provenance.get("model_variant"), provenance.get("window_ms"),
                            provenance.get("test_subject"), provenance.get("seed")) != (variant, window, subject, seed):
                        raise ValueError(f"Mismatched ablation result: {folder}")
                    result = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    for metric in metrics:
                        metrics[metric][i, j] = result[metric]
            if complete:
                matrices[variant] = metrics
            time_rows.extend(timing_rows(timing_totals(window_runs, cfg.epochs), window, variant=variant))
            all_runs.extend(window_runs)
            group_runs[variant].extend(window_runs)
        for variant, metrics in matrices.items():
            for metric, matrix in metrics.items():
                comparisons = [("absolute", matrix)]
                if variant != "full" and "full" in matrices:
                    comparisons.append(("difference_from_full", matrix - matrices["full"][metric]))
                for comparison, values in comparisons:
                    statistics = aggregate_matrix(values, cfg.variance_ddof)
                    for scope in ("across_subjects", "across_repeats"):
                        item = statistics[scope]
                        variance = item["variance"]
                        display = (f"{item['mean']:.6f} ± {variance:.6f}" if variance is not None
                                   else f"{item['mean']:.6f} ± NA")
                        rows.append(dict(window_ms=window, variant=variant, metric=metric, comparison=comparison,
                                         scope=scope, **item, ddof=cfg.variance_ddof, mean_plus_variance=display))
    for variant, runs in group_runs.items():
        time_rows.extend(timing_rows(timing_totals(runs, cfg.epochs), "all", variant=variant))
    totals = timing_totals(all_runs, cfg.epochs)
    time_rows.extend(timing_rows(totals, "all", variant="all_selected"))
    rows.extend(time_rows)
    destination = cfg.output_root / "comparison.csv"
    with destination.with_suffix(".tmp").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    destination.with_suffix(".tmp").replace(destination)
    report = ["# 消融批次汇总", "", f"窗口：{cfg.windows_ms} ms；N={cfg.N}，M={cfg.M}。",
              f"完整组的跨被试均值 ± 方差（先平均各被试的种子，ddof={cfg.variance_ddof}）；未完成组不补零。", "",
              "| 窗口 ms | 实验组 | 测试 ACC (%) | 测试 ITR (bits/min) |",
              "| --- | --- | ---: | ---: |"]
    for window in cfg.windows_ms:
        for variant in variants:
            selected = {row["metric"]: row for row in rows if row.get("comparison") == "absolute"
                        and row["scope"] == "across_subjects" and row["window_ms"] == window and row["variant"] == variant}
            displays = [format_stat(selected[key]) if key in selected else "—"
                        for key in ("acc_percent", "itr_bits_per_minute")]
            report.append(f"| {window} | [{variant}]({variant}/report.md) | {displays[0]} | {displays[1]} |")
    report += ["", "[完整统计与配对差值](comparison.csv)"]
    report += timing_markdown(totals)
    write_text(cfg.output_root / "report.md", "\n".join(report) + "\n")
    return rows


def run_ablation(cfg=ABLATION_CONFIG, variants=ABLATION_GROUPS, prepared=None, purpose="formal"):
    cfg.validate()
    if not isinstance(variants, (tuple, list)) or not variants or len(set(variants)) != len(variants):
        raise ValueError('ABLATION_GROUPS must list unique groups, e.g. ("spectral_only",)')
    configs = [replace(cfg, model_variant=variant, experiment_name=variant).validate() for variant in variants]
    print(f"Selected groups (in order): {', '.join(variants)}; automatic folders under {cfg.output_root}", flush=True)
    suite_details = cfg.output_root / "details"
    suite_details.mkdir(parents=True, exist_ok=True)
    with experiment_lock(suite_details):
        prepared = prepare_data(cfg) if prepared is None else prepared.validate(cfg)
        hashes = {name: file_sha256(PROJECT_ROOT / name) for name in SOURCE_NAMES}
        # 先检查所有已选组，再训练，避免设置不一致时先跑完另一组才报错。
        for group in configs:
            details = group.output_root / group.experiment_name / "details"
            manifest_file = details / "experiment.json"
            if manifest_file.exists():
                if not cfg.resume:
                    raise FileExistsError(f"Refusing to overwrite {details.parent}")
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                validate_resume(manifest, group, prepared, purpose, hashes, details)
        (cfg.output_root / "comparison.csv").unlink(missing_ok=True)
        try:
            for group in configs:
                print(f"\nABLATION {group.model_variant}: windows={group.windows_ms}, N={group.N}, M={group.M}", flush=True)
                run_experiment(group, prepared, purpose)
        finally:
            summarize_ablation(cfg, variants)
    return cfg.output_root


if __name__ == "__main__":
    run_ablation()
