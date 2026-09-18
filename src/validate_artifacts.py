from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


class ArtifactValidationError(ValueError):
    pass


def _require_file(path: Path) -> None:
    if not path.exists():
        raise ArtifactValidationError(f"Missing required artifact: {path}")


def _require_columns(path: Path, required_cols: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ArtifactValidationError(
            f"Artifact {path} missing required columns: {sorted(missing)}"
        )
    return df


def validate_artifacts(root: Path) -> dict[str, object]:
    metrics_dir = root / "metrics"
    predictions_dir = root / "predictions"
    counterfactuals_dir = root / "counterfactuals"
    figures_dir = root / "figures"

    required_files = [
        metrics_dir / "run_summary.json",
        metrics_dir / "final_manifest.json",
        metrics_dir / "final_comparison_pack.json",
        metrics_dir / "xcreditscore_metrics.csv",
        metrics_dir / "baseline_metrics.csv",
        metrics_dir / "calibration_report.csv",
        metrics_dir / "calibration_summary.json",
        metrics_dir / "threshold_frontier.csv",
        metrics_dir / "counterfactual_summary.json",
        metrics_dir / "counterfactual_top_changed_features.csv",
        metrics_dir / "onnx_export_status.json",
        metrics_dir / "onnx_parity_smoke.json",
        metrics_dir / "failure_transparency.json",
        metrics_dir / "compute_budget_parity_report.json",
        metrics_dir / "calibration_method_comparison.csv",
        metrics_dir / "environment_manifest.json",
        predictions_dir / "explainability_report.csv",
        counterfactuals_dir / "counterfactual_results.csv",
        figures_dir / "xcreditscore_reliability_curve.png",
    ]
    for p in required_files:
        _require_file(p)

    with open(metrics_dir / "run_summary.json", "r", encoding="utf-8") as f:
        run_summary = json.load(f)
    for key in ["run_profile", "threshold", "model_family", "threshold_tuning"]:
        if key not in run_summary:
            raise ArtifactValidationError(f"run_summary.json missing key: {key}")

    with open(metrics_dir / "environment_manifest.json", "r", encoding="utf-8") as f:
        env_manifest = json.load(f)
    for key in ["generated_at_utc", "config_hash_sha256", "dependencies", "dataset"]:
        if key not in env_manifest:
            raise ArtifactValidationError(f"environment_manifest.json missing key: {key}")

    with open(metrics_dir / "final_manifest.json", "r", encoding="utf-8") as f:
        final_manifest = json.load(f)
    for key in [
        "generated_at_utc",
        "run_profile",
        "config_hash_sha256",
        "threshold",
        "xcreditscore",
        "best_baseline_auc",
        "counterfactual",
    ]:
        if key not in final_manifest:
            raise ArtifactValidationError(f"final_manifest.json missing key: {key}")

    with open(metrics_dir / "onnx_export_status.json", "r", encoding="utf-8") as f:
        onnx_export = json.load(f)
    with open(metrics_dir / "onnx_parity_smoke.json", "r", encoding="utf-8") as f:
        onnx_parity = json.load(f)

    for key in ["ok", "status", "rows"]:
        if key not in onnx_parity:
            raise ArtifactValidationError(f"onnx_parity_smoke.json missing key: {key}")

    if bool(onnx_export.get("ok", False)) and not bool(onnx_parity.get("ok", False)):
        raise ArtifactValidationError(
            "ONNX export succeeded but parity smoke check failed. "
            f"status={onnx_parity.get('status')}"
        )

    x_df = _require_columns(
        metrics_dir / "xcreditscore_metrics.csv",
        {"model", "auc", "accuracy", "precision", "recall", "f1"},
    )
    b_df = _require_columns(metrics_dir / "baseline_metrics.csv", {"model", "auc"})
    _require_columns(
        predictions_dir / "explainability_report.csv",
        {"test_index", "decision", "pd_score"},
    )
    _require_columns(
        counterfactuals_dir / "counterfactual_results.csv",
        {"test_index", "feasible", "original_score", "new_score"},
    )

    auc = float(x_df.iloc[0]["auc"])
    if not (0.0 <= auc <= 1.0):
        raise ArtifactValidationError("xcreditscore_metrics.csv has AUC outside [0,1]")

    if b_df.empty:
        raise ArtifactValidationError("baseline_metrics.csv must contain at least one baseline row")

    return {
        "ok": True,
        "root": str(root),
        "xcreditscore_auc": auc,
        "baseline_rows": int(len(b_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate required Chapter 4 artifacts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/heloc_pack"),
        help="Artifact root containing metrics/, predictions/, and counterfactuals/",
    )
    args = parser.parse_args()
    result = validate_artifacts(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
