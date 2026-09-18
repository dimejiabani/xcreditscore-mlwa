from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "joblib",
    "xgboost",
    "tensorflow",
    "tensorflow-lattice",
    "tf-keras",
    "tf2onnx",
    "onnxruntime",
    "pulp",
]

TRACKED_ENV_VARS = [
    "RUN_PROFILE",
    "ARTIFACT_RUN_SUFFIX",
    "ENABLE_THRESHOLD_TUNING",
    "THRESHOLD_TUNE_OBJECTIVE",
    "FEATURE_ENGINEERING_ENABLED",
    "CV_ENABLED",
    "CV_N_SPLITS",
    "CV_N_REPEATS",
    "BASELINE_USE_TUNED",
    "SKIP_XGBOOST",
]


def _safe_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_head(workspace_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def build_reproducibility_manifest(
    workspace_root: Path,
    data_path: Path,
    artifact_root: Path,
    config_snapshot: dict[str, Any],
) -> dict[str, Any]:
    config_serialized = json.dumps(config_snapshot, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_serialized.encode("utf-8")).hexdigest()

    dependencies = {name: _safe_version(name) for name in TRACKED_PACKAGES}
    env_snapshot = {k: os.getenv(k) for k in TRACKED_ENV_VARS if os.getenv(k) is not None}

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": {
            "head": _git_head(workspace_root),
        },
        "artifact_root": str(artifact_root),
        "dataset": {
            "path": str(data_path),
            "sha256": _sha256_file(data_path) if data_path.exists() else None,
        },
        "config_snapshot": config_snapshot,
        "config_hash_sha256": config_hash,
        "dependencies": dependencies,
        "environment_overrides": env_snapshot,
    }
    return manifest


def write_reproducibility_manifest(metrics_dir: Path, manifest: dict[str, Any]) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "environment_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out
