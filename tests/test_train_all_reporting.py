from __future__ import annotations

import pandas as pd
import pytest

from src.train_all import (
    _best_operating_point,
    build_explainability_global_summary,
    build_model_competitiveness_report,
)


def test_best_operating_point_picks_highest_metric_row() -> None:
    frontier = pd.DataFrame(
        [
            {"threshold": 0.3, "approve_rate": 0.2, "deny_rate": 0.8, "accuracy": 0.65, "precision": 0.60, "recall": 0.90, "f1": 0.72},
            {"threshold": 0.4, "approve_rate": 0.3, "deny_rate": 0.7, "accuracy": 0.70, "precision": 0.68, "recall": 0.86, "f1": 0.75},
        ]
    )

    best = _best_operating_point(frontier, "f1")

    assert best is not None
    assert best["threshold"] == 0.4
    assert best["f1"] == 0.75


def test_build_explainability_global_summary_aggregates_effects_and_reason_rate() -> None:
    explain_df = pd.DataFrame(
        [
            {
                "test_index": 0,
                "reason_codes": '["A", "B"]',
                "feature_effects": '{"A": 0.4, "B": 0.1}',
            },
            {
                "test_index": 1,
                "reason_codes": '["A"]',
                "feature_effects": '{"A": -0.2, "B": 0.3}',
            },
        ]
    )

    out = build_explainability_global_summary(explain_df, ["A", "B"])
    rows = {r["feature"]: r for _, r in out.iterrows()}

    assert set(rows.keys()) == {"A", "B"}
    assert rows["A"]["top_reason_rate"] == 1.0
    assert rows["B"]["top_reason_rate"] == 0.5


def test_build_model_competitiveness_report_uses_primary_metric_and_delta() -> None:
    predictor_metrics = {"auc": 0.80, "pr_auc": 0.78, "f1": 0.75, "accuracy": 0.72}
    baselines_df = pd.DataFrame(
        [
            {"model": "lr", "auc": 0.79, "pr_auc": 0.77, "f1": 0.74, "accuracy": 0.71},
            {"model": "xgb", "auc": 0.81, "pr_auc": 0.76, "f1": 0.73, "accuracy": 0.70},
        ]
    )

    report = build_model_competitiveness_report(
        predictor_metrics=predictor_metrics,
        baselines_df=baselines_df,
        primary_metric="auc",
    )

    assert report["primary_metric"] == "auc"
    assert report["best_baseline"]["model"] == "xgb"
    assert report["delta_vs_best_baseline"] == pytest.approx(-0.01)
    assert report["xcreditscore_beats_best_baseline"] is False
