"""Command line entry point: download, prepare, eda, train, experiments, report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .analysis.compare import write_summary
from .data.download import ensure_raw
from .data.prepare import prepare
from .experiments import BASELINES, run_condition, run_experiments
from .logging_utils import setup_logging
from .report.eda import build_eda
from .report.main import build_report


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="path to a YAML config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=3",
    )


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def _load(args: argparse.Namespace) -> Config:
    return load_config(Path(args.config), _overrides(args))


def _setup(cfg: Config, name: str) -> None:
    setup_logging(Path(cfg.output.out_dir) / f"{name}.log")


def _run(args: argparse.Namespace) -> int:
    cfg = _load(args)
    _setup(cfg, args.command or "ecgmtl")
    if args.command == "download":
        ensure_raw(cfg.data)
    elif args.command == "prepare":
        prepare(cfg.data)
    elif args.command == "eda":
        build_eda(cfg)
    elif args.command == "train":
        run_condition(cfg, args.condition, args.folds, force=args.force)
    elif args.command == "experiments":
        run_experiments(cfg, force=args.force)
    elif args.command == "summarize":
        write_summary(cfg.output.out_dir, split="test", baseline=BASELINES)
    elif args.command == "report":
        build_report(cfg)
    elif args.command in ("all", "smoke"):
        prepare(cfg.data)
        build_eda(cfg)
        run_experiments(cfg, force=args.force)
        build_report(cfg)
    else:  # pragma: no cover - argparse rejects unknown commands
        raise SystemExit(f"unknown command {args.command}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecgmtl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("download", "fetch the raw PTB-XL archive"),
        ("prepare", "build the signal memmap and the label manifest"),
        ("eda", "write report/eda.html"),
        ("train", "train one condition on one fold"),
        ("experiments", "run the whole grid and aggregate it"),
        ("summarize", "aggregate finished runs into summary.json"),
        ("report", "write report/index.html"),
        ("all", "prepare, eda, experiments, report"),
        ("smoke", "same as all, meant for the smoke config"),
    ):
        command = sub.add_parser(name, help=help_text)
        _add_common(command)
        command.add_argument("--force", action="store_true", help="rerun finished fold runs")
        if name == "train":
            command.add_argument("--condition", default="mtl_equal")
            command.add_argument("--folds", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
