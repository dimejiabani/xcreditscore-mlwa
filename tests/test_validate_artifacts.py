from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.validate_artifacts import ArtifactValidationError, validate_artifacts


def _write_contract(root: Path) -> None:
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "counterfactuals").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)

    (root / "metrics" / "run_summary.json").write_text(
        json.dumps(
            {
                "run_profile": "final",
                "threshold": 0.36,
                "model_family": "tfl",
                "threshold_tuning": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    (root / "metrics" / "environment_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-01-01T00:00:00+00:00",
                "config_hash_sha256": "abc",
                "dependencies": {"numpy": "1.0"},
                "dataset": {"path": "sample.csv", "sha256": "def"},
            }
        ),
        encoding="utf-8",
    )

    (root / "metrics" / "final_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-01-01T00:00:00+00:00",
                "run_profile": "final",
                "config_hash_sha256": "abc",
                "threshold": 0.36,
                "xcreditscore": {"auc": 0.8, "f1": 0.7},
                "best_baseline_auc": 0.75,
                "counterfactual": {"cases": 1, "feasible_rate": 1.0},
            }
        ),
        encoding="utf-8",
    )

    (root / "metrics" / "final_comparison_pack.json").write_text(
        json.dumps(
            {
                "run_profile": "final",
                "threshold": 0.36,
                "xcreditscore_metrics": {"auc": 0.8, "f1": 0.73},
                "baseline_metrics": [{"model": "log_reg", "auc": 0.75}],
                "model_competitiveness": {"primary_metric": "auc"},
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "model": "xcreditscore",
                "auc": 0.8,
                "accuracy": 0.7,
                "precision": 0.72,
                "recall": 0.74,
                "f1": 0.73,
            }
        ]
    ).to_csv(root / "metrics" / "xcreditscore_metrics.csv", index=False)

    pd.DataFrame([{"model": "log_reg", "auc": 0.75}]).to_csv(
        root / "metrics" / "baseline_metrics.csv", index=False
    )

    pd.DataFrame(
        [{"score_bin": 0, "n": 1, "avg_pred": 0.8, "event_rate": 1.0, "abs_gap": 0.2}]
    ).to_csv(root / "metrics" / "calibration_report.csv", index=False)

    (root / "metrics" / "calibration_summary.json").write_text(
        json.dumps({"ece": 0.05, "mce": 0.2, "bins": 1}),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "threshold": 0.36,
                "approve_rate": 0.3,
                "deny_rate": 0.7,
                "accuracy": 0.7,
                "precision": 0.7,
                "recall": 0.8,
                "f1": 0.75,
                "tn": 1,
                "fp": 1,
                "fn": 1,
                "tp": 1,
            }
        ]
    ).to_csv(root / "metrics" / "threshold_frontier.csv", index=False)

    (root / "metrics" / "counterfactual_summary.json").write_text(
        json.dumps({"cases": 1, "feasible_rate": 1.0, "median_total_cost": 0.2, "median_changed_features": 1.0}),
        encoding="utf-8",
    )

    pd.DataFrame([{"feature": "ExternalRiskEstimate", "count": 1}]).to_csv(
        root / "metrics" / "counterfactual_top_changed_features.csv", index=False
    )

    (root / "metrics" / "onnx_export_status.json").write_text(
        json.dumps({"ok": True, "detail": "ok_direct"}),
        encoding="utf-8",
    )

    (root / "metrics" / "failure_transparency.json").write_text(
        json.dumps(
            {
                "predictor": {"model_family": "tensorflow_lattice", "fallback_used": False, "fallback_reason": ""},
                "baselines": {"log_reg": {"status": "ok", "error_type": "", "fallback_used": False}},
                "onnx": {"export_status": {"ok": True}, "parity_status": {"ok": True}},
            }
        ),
        encoding="utf-8",
    )

    (root / "metrics" / "compute_budget_parity_report.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "parity": {"same_cv_splits": True, "same_parity_total_trials": True, "details": "ok"},
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "method": "raw",
                "auc": 0.8,
                "pr_auc": 0.79,
                "brier": 0.18,
                "log_loss": 0.54,
                "ece": 0.03,
                "mce": 0.06,
            }
        ]
    ).to_csv(root / "metrics" / "calibration_method_comparison.csv", index=False)

    (root / "metrics" / "onnx_parity_smoke.json").write_text(
        json.dumps(
            {
                "ok": True,
                "status": "ok",
                "rows": 8,
                "tolerance": 0.001,
                "max_abs_err": 0.0001,
                "mean_abs_err": 0.00005,
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [{"test_index": 0, "decision": "Deny", "pd_score": 0.9}]
    ).to_csv(root / "predictions" / "explainability_report.csv", index=False)

    pd.DataFrame(
        [
            {
                "test_index": 0,
                "feasible": True,
                "original_score": 0.9,
                "new_score": 0.3,
            }
        ]
    ).to_csv(root / "counterfactuals" / "counterfactual_results.csv", index=False)

    (root / "figures" / "xcreditscore_reliability_curve.png").write_bytes(b"PNG")


def test_validate_artifacts_success(tmp_path: Path) -> None:
    root = tmp_path / "heloc_pack"
    _write_contract(root)

    result = validate_artifacts(root)
    assert result["ok"] is True
    assert result["baseline_rows"] == 1


def test_validate_artifacts_missing_file_raises(tmp_path: Path) -> None:
    root = tmp_path / "heloc_pack"
    _write_contract(root)
    (root / "metrics" / "environment_manifest.json").unlink()

    with pytest.raises(ArtifactValidationError):
        validate_artifacts(root)


def test_validate_artifacts_onnx_parity_failure_raises(tmp_path: Path) -> None:
    root = tmp_path / "heloc_pack"
    _write_contract(root)

    (root / "metrics" / "onnx_export_status.json").write_text(
        json.dumps({"ok": True, "detail": "ok_direct"}),
        encoding="utf-8",
    )
    (root / "metrics" / "onnx_parity_smoke.json").write_text(
        json.dumps({"ok": False, "status": "parity_exceeds_tolerance", "rows": 8}),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError):
        validate_artifacts(root)
