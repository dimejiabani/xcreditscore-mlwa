"""Entry point.

Selects a dataset from :mod:`src.datasets` via ``--dataset`` (or the
``DATASET_NAME`` env var) and hands off to :mod:`src.train_all`. The
env var is set *before* importing train_all so ``src.config`` picks up
the right dataset module at import time.
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the XCreditScore pipeline on the selected dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("DATASET_NAME", "heloc"),
        help="Dataset to run against (default: heloc). See src/datasets/.",
    )
    return parser.parse_args(argv)


def _main() -> None:
    args = _parse_args(sys.argv[1:])
    # Must be set BEFORE importing src.train_all (which imports src.config,
    # which reads this env var to pick a dataset module).
    os.environ["DATASET_NAME"] = str(args.dataset).strip().lower()

    from src.train_all import main

    main()


if __name__ == "__main__":
    _main()
