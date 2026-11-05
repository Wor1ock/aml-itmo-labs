"""Download the Fluent Speech Commands Kaggle dataset.

Requires a Kaggle API token in ``~/.kaggle/kaggle.json`` (or KAGGLE_USERNAME /
KAGGLE_KEY). The Kaggle CLI is a development dependency.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET = "tommyngx/fluent-speech-corpus"
EXPECTED = Path("data/fluent_speech_commands_dataset/data/train_data.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if EXPECTED.is_file() and not args.force:
        print(f"dataset already present at {EXPECTED}")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(args.output),
        "--unzip",
    ]
    print("running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            "download failed. Install the Kaggle CLI (`uv sync`) and configure credentials, "
            "then retry. Dataset: https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus",
            file=sys.stderr,
        )
        return completed.returncode
    if not EXPECTED.is_file():
        print(f"download finished but {EXPECTED} was not found", file=sys.stderr)
        return 1
    print(f"dataset is ready at {EXPECTED.parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
