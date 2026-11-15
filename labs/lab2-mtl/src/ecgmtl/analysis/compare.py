"""Method comparison: per-task table across folds plus paired deltas against a baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..metrics import aggregate

TASK_METRICS: dict[str, list[str]] = {
    "diag": ["macro_auc", "macro_f1"],
    "rhythm": ["macro_f1", "accuracy"],
    "age": ["mae", "rmse", "r2"],
}


def read_fold_metrics(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def collect_runs(
    table_dir: Path, split: str = "test"
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """condition -> task -> metric -> list of per-fold values."""
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for condition_dir in sorted(table_dir.glob("*/")):
        condition = condition_dir.name
        per_fold = sorted(condition_dir.glob("fold*/metrics.json"))
        if not per_fold:
            continue
        tasks = out.setdefault(condition, {})
        for path in per_fold:
            metrics = json.loads(path.read_text(encoding="utf-8"))[split]
            for task, task_metrics in metrics.items():
                if task == "n" or not isinstance(task_metrics, dict):
                    continue
                bucket = tasks.setdefault(task, {})
                for metric, value in task_metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        bucket.setdefault(metric, []).append(float(value))
    return out


def summarize(runs: dict[str, dict[str, dict[str, list[float]]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition, tasks in runs.items():
        summary[condition] = {}
        for task, metrics in tasks.items():
            n_folds = max(len(values) for values in metrics.values())
            # a metric missing in one fold must not shift the pairing of the others
            clean = {name: values for name, values in metrics.items() if len(values) == n_folds}
            if not clean:
                continue
            folds = [dict(zip(clean, values)) for values in zip(*clean.values())]
            summary[condition][task] = aggregate(folds, list(clean))
    return summary


def flat_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for condition, tasks in summary.items():
        for task, metrics in tasks.items():
            for metric, stats_ in metrics.items():
                rows.append(
                    {
                        "method": condition,
                        "task": task,
                        "metric": metric,
                        "mean": stats_["mean"],
                        "std": stats_["std"],
                        "ci_low": stats_["ci_low"],
                        "ci_high": stats_["ci_high"],
                        "n": stats_["n"],
                    }
                )
    return pd.DataFrame(rows)


def paired_deltas(
    runs: dict[str, dict[str, dict[str, list[float]]]],
    baseline: Sequence[tuple[str, str]],
    conditions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per-task deltas against the matching single-task baseline, with a paired t-test."""
    baseline_of = {"diag": baseline[0][0], "rhythm": baseline[1][0], "age": baseline[2][0]}
    rows = []
    for condition in conditions or runs:
        for task, metrics in runs.get(condition, {}).items():
            ref_name = baseline_of.get(task)
            reference: dict[str, list[float]] = runs.get(ref_name or "", {}).get(task, {})
            for metric, values in metrics.items():
                ref_values = reference.get(metric, [])
                if condition == ref_name or len(values) != len(ref_values) or len(values) < 2:
                    continue
                diffs = np.asarray(values) - np.asarray(ref_values)
                test = stats.ttest_rel(values, ref_values)
                rows.append(
                    {
                        "method": condition,
                        "baseline": ref_name,
                        "task": task,
                        "metric": metric,
                        "delta": float(diffs.mean()),
                        "delta_std": float(diffs.std(ddof=1)),
                        "p_value": float(test.pvalue),
                        "n": int(len(diffs)),
                    }
                )
    return pd.DataFrame(rows)


def write_summary(table_dir: Path, split: str = "test", baseline: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    runs = collect_runs(table_dir, split)
    summary = summarize(runs)
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame = flat_table(summary)
    frame.to_csv(table_dir / "summary.csv", index=False)
    if baseline:
        deltas = paired_deltas(runs, baseline)
        if not deltas.empty:
            deltas.to_csv(table_dir / "deltas.csv", index=False)
    return summary
