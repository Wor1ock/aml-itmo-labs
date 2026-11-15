"""Experiment grid: conditions x fold runs, then the aggregated table."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis.compare import write_summary
from .config import TASKS, Config
from .logging_utils import get_logger
from .train import run_training


@dataclass(frozen=True)
class Condition:
    """One row of the results table: which tasks are trained together and how."""

    name: str
    tasks: tuple[str, ...]
    strategy: str
    weights: dict[str, float] | None = None


CONDITIONS: tuple[Condition, ...] = (
    Condition("single_diag", ("diag",), "equal"),
    Condition("single_rhythm", ("rhythm",), "equal"),
    Condition("single_age", ("age",), "equal"),
    Condition("mtl_equal", TASKS, "equal"),
    Condition("mtl_uncertainty", TASKS, "uncertainty"),
    Condition("mtl_pcgrad", TASKS, "pcgrad"),
    Condition("mtl_age_weight_0.1", TASKS, "equal", {"diag": 1.0, "rhythm": 1.0, "age": 0.1}),
    Condition("loto_no_rhythm", ("diag", "age"), "equal"),
    Condition("loto_no_age", ("diag", "rhythm"), "equal"),
)

BASELINES: tuple[tuple[str, str], ...] = (
    ("diag", "single_diag"),
    ("rhythm", "single_rhythm"),
    ("age", "single_age"),
)


def get_condition(name: str) -> Condition:
    for condition in CONDITIONS:
        if condition.name == name:
            return condition
    raise KeyError(f"unknown condition {name!r}")


def run_dir_for(cfg: Config, condition: Condition, run: int) -> Path:
    test_fold = cfg.split.test_folds[run]
    return cfg.output.out_dir / condition.name / f"fold{test_fold}"


def with_condition(cfg: Config, condition: Condition, run: int) -> Config:
    """Derive a config for one condition and one fold run without mutating the original."""
    weights = condition.weights or {task: 1.0 for task in condition.tasks}
    if set(weights) != set(condition.tasks):
        weights = {task: weights.get(task, 1.0) for task in condition.tasks}
    loss = dataclasses.replace(
        cfg.loss, tasks=condition.tasks, strategy=condition.strategy, weights=weights
    )
    split = dataclasses.replace(cfg.split, run=run)
    return dataclasses.replace(cfg, loss=loss, split=split)


def run_condition(cfg: Config, name: str, folds: int, force: bool = False) -> list[dict[str, object]]:
    log = get_logger()
    condition = get_condition(name)
    results = []
    for run in range(min(folds, len(cfg.split.test_folds))):
        run_cfg = with_condition(cfg, condition, run)
        run_dir = run_dir_for(run_cfg, condition, run)
        if (run_dir / "metrics.json").exists() and not force:
            log.info("skip %s (already done)", run_dir)
            results.append(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")))
            continue
        log.info("=== condition %s, test fold %d ===", name, run_cfg.split.test_folds[run])
        results.append(run_training(run_cfg, run_dir))
    return results


def run_experiments(cfg: Config, force: bool = False) -> dict[str, object]:
    """Run the main table and the ablations, then aggregate everything."""
    log = get_logger()
    for name in cfg.experiments.main:
        run_condition(cfg, name, cfg.experiments.folds, force)
    for name in cfg.experiments.ablations:
        run_condition(cfg, name, cfg.experiments.ablation_folds, force)
    summary = write_summary(cfg.output.out_dir, split="test", baseline=BASELINES)
    log.info("summary written to %s", cfg.output.out_dir / "summary.json")
    return summary


def table_path(cfg: Config) -> Path:
    return cfg.output.out_dir / "summary.json"


def best_run_dir(cfg: Config, condition: str) -> Path | None:
    """First completed fold run of a condition, used to draw curves."""
    base = cfg.output.out_dir / condition
    for path in sorted(base.glob("fold*")):
        if (path / "metrics.json").exists():
            return path
    return None


def scores_of(cfg: Config, condition: str, split: str = "test") -> dict[str, float]:
    """Mean test score of one condition, handy for quick checks."""
    path = cfg.output.out_dir / "summary.json"
    if not path.exists():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for task, metrics in summary.get(condition, {}).items():
        for metric, stats in metrics.items():
            out[f"{task}/{metric}"] = float(np.mean(stats["mean"]))
    return out
