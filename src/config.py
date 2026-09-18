"""Runtime configuration.

Historically HELOC-only. Now a thin dataset-dispatch layer: the ``DATASET_NAME``
environment variable selects a module from :mod:`src.datasets` and every
dataset-specific symbol (feature list, target, constraints, lattice groups,
cost weights, feature-engineering hook, artifact directory) is re-exported so
existing consumers (``from .config import FOO``) keep working unchanged.

With ``DATASET_NAME`` unset the default is ``heloc`` and the exported symbols
are identical to the pre-refactor definitions — so the current HELOC pipeline
runs byte-for-byte the same on the deterministic artifact set.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .datasets import load_dataset_module

RANDOM_STATE = 42
# Single operating threshold used consistently across predictor, explainability,
# and counterfactual engines.
THRESHOLD = 0.36

# Monotonic sanity-check tolerance and perturbation magnitudes.
MONOTONIC_CHECK_EPS = 1e-6
MONOTONIC_CHECK_STEPS: tuple[float, ...] = (0.10, 0.05, 0.02, 0.01)

RUN_PROFILE = os.getenv("RUN_PROFILE", "final").strip().lower()
if RUN_PROFILE not in {"final", "fast"}:
    RUN_PROFILE = "final"

FINAL_LATTICE_EPOCHS = 36
FINAL_LATTICE_BATCH_SIZE = 128
FINAL_LATTICE_CALIBRATION_KEYPOINTS = 15
FINAL_LATTICE_SIZE = 4
FINAL_LATTICE_LEARNING_RATE = 0.005
FINAL_LATTICE_EARLY_STOPPING_PATIENCE = 4
FINAL_COUNTERFACTUAL_MAX_CASES = 80

FAST_LATTICE_EPOCHS = 12
FAST_LATTICE_BATCH_SIZE = 128
FAST_LATTICE_CALIBRATION_KEYPOINTS = 10
FAST_LATTICE_SIZE = 3
FAST_LATTICE_LEARNING_RATE = 0.005
FAST_LATTICE_EARLY_STOPPING_PATIENCE = 4
FAST_COUNTERFACTUAL_MAX_CASES = 20

if RUN_PROFILE == "fast":
    LATTICE_EPOCHS = FAST_LATTICE_EPOCHS
    LATTICE_BATCH_SIZE = FAST_LATTICE_BATCH_SIZE
    LATTICE_CALIBRATION_KEYPOINTS = FAST_LATTICE_CALIBRATION_KEYPOINTS
    LATTICE_SIZE = FAST_LATTICE_SIZE
    LATTICE_LEARNING_RATE = FAST_LATTICE_LEARNING_RATE
    LATTICE_EARLY_STOPPING_PATIENCE = FAST_LATTICE_EARLY_STOPPING_PATIENCE
    COUNTERFACTUAL_MAX_CASES = FAST_COUNTERFACTUAL_MAX_CASES
    MONOTONIC_PAIR_STRESS_SAMPLES = 20
    EXPLAIN_STABILITY_MAX_ROWS = 120
    EXPLAIN_STABILITY_PERTURBATIONS = 4
    EXPLAIN_STABILITY_SIGMA = 0.01
else:
    LATTICE_EPOCHS = FINAL_LATTICE_EPOCHS
    LATTICE_BATCH_SIZE = FINAL_LATTICE_BATCH_SIZE
    LATTICE_CALIBRATION_KEYPOINTS = FINAL_LATTICE_CALIBRATION_KEYPOINTS
    LATTICE_SIZE = FINAL_LATTICE_SIZE
    LATTICE_LEARNING_RATE = FINAL_LATTICE_LEARNING_RATE
    LATTICE_EARLY_STOPPING_PATIENCE = FINAL_LATTICE_EARLY_STOPPING_PATIENCE
    COUNTERFACTUAL_MAX_CASES = FINAL_COUNTERFACTUAL_MAX_CASES
    MONOTONIC_PAIR_STRESS_SAMPLES = 64
    EXPLAIN_STABILITY_MAX_ROWS = 400
    EXPLAIN_STABILITY_PERTURBATIONS = 8
    EXPLAIN_STABILITY_SIGMA = 0.01

# Counterfactual search and reporting controls.
COUNTERFACTUAL_NEAR_MARGIN = 0.02
COUNTERFACTUAL_DEFAULT_MAX_STEP_FRACTION = 0.20

FINAL_COUNTERFACTUAL_EXACT_TIME_LIMIT = 2.0
FINAL_COUNTERFACTUAL_EXACT_MIP_GAP = 0.02
FINAL_COUNTERFACTUAL_REFINE_MAX_ROUNDS = 32
FINAL_COUNTERFACTUAL_EXACT_MAX_CASES = 40

FAST_COUNTERFACTUAL_EXACT_TIME_LIMIT = 0.75
FAST_COUNTERFACTUAL_EXACT_MIP_GAP = 0.05
FAST_COUNTERFACTUAL_REFINE_MAX_ROUNDS = 12
FAST_COUNTERFACTUAL_EXACT_MAX_CASES = 0

if RUN_PROFILE == "fast":
    COUNTERFACTUAL_EXACT_TIME_LIMIT = FAST_COUNTERFACTUAL_EXACT_TIME_LIMIT
    COUNTERFACTUAL_EXACT_MIP_GAP = FAST_COUNTERFACTUAL_EXACT_MIP_GAP
    COUNTERFACTUAL_REFINE_MAX_ROUNDS = FAST_COUNTERFACTUAL_REFINE_MAX_ROUNDS
    COUNTERFACTUAL_EXACT_MAX_CASES = FAST_COUNTERFACTUAL_EXACT_MAX_CASES
else:
    COUNTERFACTUAL_EXACT_TIME_LIMIT = FINAL_COUNTERFACTUAL_EXACT_TIME_LIMIT
    COUNTERFACTUAL_EXACT_MIP_GAP = FINAL_COUNTERFACTUAL_EXACT_MIP_GAP
    COUNTERFACTUAL_REFINE_MAX_ROUNDS = FINAL_COUNTERFACTUAL_REFINE_MAX_ROUNDS
    COUNTERFACTUAL_EXACT_MAX_CASES = FINAL_COUNTERFACTUAL_EXACT_MAX_CASES

# Baseline runtime controls.
SKIP_XGBOOST = os.getenv("SKIP_XGBOOST", "0").strip() == "1"
if RUN_PROFILE == "fast":
    BASELINE_RF_N_ESTIMATORS = 180
    BASELINE_XGB_N_ESTIMATORS = 220
else:
    BASELINE_RF_N_ESTIMATORS = 320
    BASELINE_XGB_N_ESTIMATORS = 420

# Cross-validation controls for robust model evaluation.
CV_ENABLED = os.getenv("CV_ENABLED", "1").strip() == "1"
if RUN_PROFILE == "fast":
    CV_N_SPLITS = 3
    CV_N_REPEATS = 1
else:
    CV_N_SPLITS = 5
    CV_N_REPEATS = 2

# Grouping strategy for lattice interaction blocks.
# - "correlation": rank features by |corr(feature, target)| and chunk by max dim.
# - "fixed": use LATTICE_FIXED_GROUPS exactly as declared below.
LATTICE_GROUPING_MODE = "fixed"
LATTICE_MAX_GROUP_DIM = 2

# Tune threshold by default to improve operating-point quality.
ENABLE_THRESHOLD_TUNING = os.getenv("ENABLE_THRESHOLD_TUNING", "1").strip() == "1"
THRESHOLD_TUNE_MIN = 0.30
THRESHOLD_TUNE_MAX = 0.70
THRESHOLD_TUNE_STEP = 0.01
THRESHOLD_TUNE_OBJECTIVE = os.getenv("THRESHOLD_TUNE_OBJECTIVE", "f1").strip().lower()
FEATURE_ENGINEERING_ENABLED = os.getenv("FEATURE_ENGINEERING_ENABLED", "1").strip() == "1"
LATTICE_USE_TUNED = os.getenv("LATTICE_USE_TUNED", "1").strip() == "1"
PRIMARY_COMPARISON_METRIC = os.getenv("PRIMARY_COMPARISON_METRIC", "auc").strip().lower()

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Dataset-scoped section: DATASET_NAME selects a module from src/datasets/ and
# every dataset-specific symbol below is re-exported from it. Defaults to
# "heloc" so consumers that do `from .config import FEATURE_COLS` (etc.) keep
# working exactly as before.
# ---------------------------------------------------------------------------

DATASET_NAME = os.getenv("DATASET_NAME", "heloc").strip().lower()
DATASET = load_dataset_module(DATASET_NAME)

TARGET_COL = DATASET.TARGET_COL
TARGET_MAP = dict(DATASET.TARGET_MAP)
FEATURE_COLS = list(DATASET.FEATURE_COLS)
SENTINEL_VALUES = list(DATASET.SENTINEL_VALUES)
MONOTONIC_CONSTRAINTS = dict(DATASET.MONOTONIC_CONSTRAINTS)
MONOTONIC_JUSTIFICATIONS = dict(getattr(DATASET, "MONOTONIC_JUSTIFICATIONS", {}))
IMMUTABLE_FEATURES = set(DATASET.IMMUTABLE_FEATURES)
LATTICE_FIXED_GROUPS = tuple(tuple(g) for g in DATASET.LATTICE_FIXED_GROUPS)
COUNTERFACTUAL_BASE_COST_WEIGHTS = dict(DATASET.COUNTERFACTUAL_BASE_COST_WEIGHTS)
COUNTERFACTUAL_MAX_STEP_FRACTIONS = dict(DATASET.COUNTERFACTUAL_MAX_STEP_FRACTIONS)

DATA_PATH = WORKSPACE_ROOT / DATASET.DATA_FILENAME

_artifact_run_suffix_raw = os.getenv("ARTIFACT_RUN_SUFFIX", "").strip()
_artifact_run_suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", _artifact_run_suffix_raw).strip("_")
if _artifact_run_suffix:
    PACK_DIR = ARTIFACTS_DIR / f"{DATASET.ARTIFACTS_SUBDIR}_{_artifact_run_suffix}"
else:
    PACK_DIR = ARTIFACTS_DIR / DATASET.ARTIFACTS_SUBDIR


def add_engineered_features(X):
    """Delegate to the active dataset's feature-engineering hook."""
    return DATASET.add_engineered_features(X)


def prepare_target(raw_target):
    """Delegate to the active dataset's target preparation."""
    return DATASET.prepare_target(raw_target)
