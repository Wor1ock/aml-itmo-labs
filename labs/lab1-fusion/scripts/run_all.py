"""Run the whole pipeline; thin wrapper around ``fusion_slu.cli``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fusion_slu.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    args, rest = parser.parse_known_args(argv)
    command = ["all", "--config", args.config, *rest]
    if args.force:
        command.append("--force")
    return cli_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
