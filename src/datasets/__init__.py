"""Per-dataset configuration modules.

Each dataset module exposes the same public surface so ``src.config`` can pick
one at import time based on the ``DATASET_NAME`` environment variable and
re-export its symbols. Every existing consumer of ``src.config`` keeps its
import unchanged.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_REGISTRY = {
    "heloc": "src.datasets.heloc",
    "gmsc": "src.datasets.gmsc",
    "taiwan": "src.datasets.taiwan",
}


def available_datasets() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_dataset_module(name: str) -> Any:
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {available_datasets()}"
        )
    return import_module(_REGISTRY[key])
