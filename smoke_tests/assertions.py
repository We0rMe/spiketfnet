"""训练轨迹比较排除实际运行耗时；耗时另外检查累计和覆盖率。"""

import csv


def metric_rows(path):
    time_fields = {"recorded_seconds", "total_seconds", "recorded_epochs", "expected_epochs",
                   "recorded_trials", "seconds_per_trial"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: value for key, value in row.items() if key not in time_fields}
                for row in csv.DictReader(handle) if row["scope"] != "training_time_total"]
