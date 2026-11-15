"""One short run end to end: training step, artefacts, EDA and report."""

from __future__ import annotations

import json

from ecgmtl.analysis.compare import write_summary
from ecgmtl.experiments import BASELINES, run_condition
from ecgmtl.report.eda import build_eda
from ecgmtl.report.main import build_report


def test_smoke_pipeline(tiny_config) -> None:
    run_condition(tiny_config, "mtl_equal", folds=1)
    run_dir = tiny_config.output.out_dir / "mtl_equal" / f"fold{tiny_config.split.test_folds[0]}"
    for name in ("model.pt", "metrics.json", "history.json", "predictions.csv", "config.yaml", "run_meta.json"):
        assert (run_dir / name).exists(), name

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "test" in metrics and "diag" in metrics["test"]
    assert metrics["test"]["diag"]["macro_auc"] >= 0.0
    assert metrics["test"]["age"]["mae"] > 0.0

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert history[0]["train"]["losses"]["diag"] > 0.0

    eda = build_eda(tiny_config)
    assert eda.exists() and "PTB-XL" in eda.read_text(encoding="utf-8")

    write_summary(tiny_config.output.out_dir, baseline=BASELINES)
    report = build_report(tiny_config)
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "mtl_equal" in html and "base64" in html


def test_artefacts_are_written_once(tiny_config) -> None:
    run_condition(tiny_config, "mtl_equal", folds=1)
    run_dir = tiny_config.output.out_dir / "mtl_equal" / f"fold{tiny_config.split.test_folds[0]}"
    mtime = (run_dir / "metrics.json").stat().st_mtime_ns
    run_condition(tiny_config, "mtl_equal", folds=1)  # must skip the finished run
    assert (run_dir / "metrics.json").stat().st_mtime_ns == mtime
