from __future__ import annotations

import numpy as np

from src.train_all import tune_decision_threshold


def test_tune_decision_threshold_returns_candidate_in_range() -> None:
    y_valid = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    y_score = np.array([0.10, 0.20, 0.30, 0.60, 0.70, 0.80], dtype=float)

    threshold, metrics = tune_decision_threshold(y_valid, y_score)

    assert 0.30 <= threshold <= 0.70
    assert {"accuracy", "precision", "recall", "f1", "balanced_accuracy"}.issubset(metrics.keys())
