from __future__ import annotations

import argparse
import sys
from typing import Sequence

from fusion_slu.analysis import build_analysis
from fusion_slu.config import ConfigError, load_config
from fusion_slu.data.prepare import PrepareError, prepare_data
from fusion_slu.evaluate import evaluate_all
from fusion_slu.report.build import build_report
from fusion_slu.train import train_all
from fusion_slu.utils import configure_torch, ensure_dir, format_seconds, get_logger, timer


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set training.epochs=2",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusion-slu",
        description="Cross-attention fusion of text and audio for Fluent Speech Commands",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build the log-mel feature cache")
    _add_common(prepare)
    prepare.add_argument("--force", action="store_true")

    train = subparsers.add_parser("train", help="train selected models")
    _add_common(train)
    train.add_argument("--modality", action="append", choices=["fusion", "text", "audio", "late"])
    train.add_argument("--seed", action="append", type=int)
    train.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate checkpoints on clean and degraded inputs")
    _add_common(evaluate)
    evaluate.add_argument("--modality", action="append", choices=["fusion", "text", "audio", "late"])
    evaluate.add_argument("--seed", action="append", type=int)
    evaluate.add_argument("--force", action="store_true")

    analyze = subparsers.add_parser("analyze", help="aggregate runs and build figures")
    _add_common(analyze)

    report = subparsers.add_parser("report", help="build the self-contained HTML report")
    _add_common(report)

    all_parser = subparsers.add_parser("all", help="prepare, train, evaluate, analyze and build the report")
    _add_common(all_parser)
    all_parser.add_argument("--modality", action="append", choices=["fusion", "text", "audio", "late"])
    all_parser.add_argument("--seed", action="append", type=int)
    all_parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    ensure_dir(config.results_dir)
    configure_torch(config.training.num_threads)
    logger = get_logger("cli")
    command = args.command
    if command == "prepare":
        prepared = prepare_data(config, force=args.force)
        logger.info("prepared %s splits: %s", len(prepared.split_sizes), prepared.split_sizes)
        return 0

    prepared = prepare_data(config, force=False)

    modalities = getattr(args, "modality", None)
    seeds = getattr(args, "seed", None)
    force = bool(getattr(args, "force", False))

    if command == "train":
        with timer() as clock:
            train_results = train_all(
                config,
                prepared,
                force=force,
                modalities=modalities,
                seeds=seeds,
            )
        logger.info("trained %d runs in %s", len(train_results), format_seconds(clock["elapsed"]))
        return 0

    if command == "evaluate":
        with timer() as clock:
            eval_results = evaluate_all(
                config,
                prepared,
                force=force,
                modalities=modalities,
                seeds=seeds,
            )
        logger.info("evaluated %d runs in %s", len(eval_results), format_seconds(clock["elapsed"]))
        return 0

    if command == "analyze":
        with timer() as clock:
            bundle = build_analysis(config, prepared)
        logger.info(
            "figures: %d, best: %s seed %d, analysis built in %s",
            len(bundle.figures),
            bundle.best["modality"],
            bundle.best["seed"],
            format_seconds(clock["elapsed"]),
        )
        return 0

    if command == "report":
        with timer() as clock:
            bundle = build_analysis(config, prepared)
            output = build_report(config, prepared, bundle)
        logger.info("report written to %s in %s", output, format_seconds(clock["elapsed"]))
        return 0

    if command == "all":
        with timer() as clock:
            train_all(config, prepared, force=force, modalities=modalities, seeds=seeds)
            evaluate_all(config, prepared, force=force, modalities=modalities, seeds=seeds)
            bundle = build_analysis(config, prepared)
            output = build_report(config, prepared, bundle)
        logger.info("pipeline finished in %s; report: %s", format_seconds(clock["elapsed"]), output)
        return 0

    raise RuntimeError(f"unknown command {command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ConfigError, PrepareError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
