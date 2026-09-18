from __future__ import annotations

import numpy as np

from src.counterfactual import _build_feature_costs, _build_step_limits, _construct_bounds


def test_build_step_limits_respects_overrides_and_defaults() -> None:
    feature_names = [
        "ExternalRiskEstimate",  # has override
        "NumTotalTrades",  # uses default fraction
    ]
    mins = np.array([0.0, 0.0], dtype=float)
    maxs = np.array([1.0, 1.0], dtype=float)

    limits = _build_step_limits(feature_names, mins, maxs)

    assert limits.shape == (2,)
    assert limits[0] == 0.08
    assert limits[1] == 0.20


def test_construct_bounds_applies_direction_immutability_and_step_limit() -> None:
    feature_names = [
        "ExternalRiskEstimate",  # monotonic -1
        "NumInqLast6M",  # monotonic +1
        "MSinceOldestTradeOpen",  # immutable
    ]
    x0 = np.array([0.50, 0.50, 0.50], dtype=float)
    mins = np.array([0.00, 0.00, 0.00], dtype=float)
    maxs = np.array([1.00, 1.00, 1.00], dtype=float)
    step_limits = np.array([0.10, 0.20, 0.30], dtype=float)

    bounds = _construct_bounds(
        x0=x0,
        feature_names=feature_names,
        mins=mins,
        maxs=maxs,
        step_limits=step_limits,
    )

    # -1 direction -> only upward from x0, clipped by step limit.
    assert bounds[0] == (0.5, 0.6)
    # +1 direction -> only downward toward mins, clipped by step limit.
    assert bounds[1] == (0.3, 0.5)
    # immutable -> exact point bound.
    assert bounds[2] == (0.5, 0.5)


def test_build_feature_costs_scales_by_observed_volatility() -> None:
    feature_names = ["ExternalRiskEstimate", "NumInqLast6M"]
    # First feature is nearly constant -> higher scaled cost expected.
    X_ref = np.array(
        [
            [0.40, 0.10],
            [0.41, 0.90],
            [0.39, 0.20],
            [0.40, 0.80],
        ],
        dtype=float,
    )

    costs = _build_feature_costs(feature_names, X_reference=X_ref)

    assert costs.shape == (2,)
    assert costs[0] > costs[1]
    assert costs[0] > 0.0 and costs[1] > 0.0
