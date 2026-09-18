from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

from .baselines import (
    run_baselines,
    train_ebm,
    train_monotone_gam,
    train_monotone_xgboost,
)
from .config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    PACK_DIR,
    CV_ENABLED,
    CV_N_REPEATS,
    CV_N_SPLITS,
    COUNTERFACTUAL_MAX_CASES,
    COUNTERFACTUAL_EXACT_MAX_CASES,
    COUNTERFACTUAL_EXACT_MIP_GAP,
    COUNTERFACTUAL_EXACT_TIME_LIMIT,
    COUNTERFACTUAL_REFINE_MAX_ROUNDS,
    MONOTONIC_CHECK_EPS,
    MONOTONIC_PAIR_STRESS_SAMPLES,
    MONOTONIC_CHECK_STEPS,
    LATTICE_BATCH_SIZE,
    LATTICE_CALIBRATION_KEYPOINTS,
    LATTICE_EARLY_STOPPING_PATIENCE,
    LATTICE_EPOCHS,
    LATTICE_FIXED_GROUPS,
    LATTICE_GROUPING_MODE,
    LATTICE_LEARNING_RATE,
    LATTICE_MAX_GROUP_DIM,
    LATTICE_SIZE,
    MONOTONIC_CONSTRAINTS,
    ENABLE_THRESHOLD_TUNING,
    EXPLAIN_STABILITY_MAX_ROWS,
    EXPLAIN_STABILITY_PERTURBATIONS,
    EXPLAIN_STABILITY_SIGMA,
    FEATURE_ENGINEERING_ENABLED,
    LATTICE_USE_TUNED,
    PRIMARY_COMPARISON_METRIC,
    RUN_PROFILE,
    THRESHOLD_TUNE_MAX,
    THRESHOLD_TUNE_MIN,
    THRESHOLD_TUNE_OBJECTIVE,
    THRESHOLD_TUNE_STEP,
    THRESHOLD,
)
from .counterfactual import batch_counterfactuals
from .cv_evaluation import run_cross_validation
from .data_pipeline import load_raw_data, preprocess_and_split, save_imputer, save_scaler
from .explainability import build_explainability_report
from .onnx_export import export_keras_to_onnx, run_onnx_parity_smoke
from .predictor import save_predictor, train_and_evaluate
from .reproducibility import build_reproducibility_manifest, write_reproducibility_manifest


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "metrics": root / "metrics",
        "predictions": root / "predictions",
        "counterfactuals": root / "counterfactuals",
        "figures": root / "figures",
        "config": root / "config",
        "models": root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def monotonic_sanity_report(model, X_test: np.ndarray, feature_names: list[str]) -> list[dict]:
    def _violation_ci(violations: int, checks: int) -> tuple[float, float]:
        if checks <= 0:
            return 0.0, 0.0
        p = float(violations / checks)
        se = float(np.sqrt(max(p * (1.0 - p), 0.0) / checks))
        margin = 1.96 * se
        return float(max(0.0, p - margin)), float(min(1.0, p + margin))

    report = []
    baseline = model.predict_proba(X_test)[:, 1]
    eps = float(MONOTONIC_CHECK_EPS)
    steps = tuple(float(s) for s in MONOTONIC_CHECK_STEPS if float(s) > 0.0)
    if not steps:
        steps = (0.05,)

    for i, name in enumerate(feature_names):
        direction = MONOTONIC_CONSTRAINTS.get(name, 0)
        if direction == 0:
            continue

        feature_vals = X_test[:, i]
        span = np.nanmax(feature_vals) - np.nanmin(feature_vals)
        if span == 0:
            continue

        total_checks = 0
        total_violations = 0

        for step in steps:
            delta = span * step
            if delta <= 0:
                continue

            X_mod = X_test.copy()
            if direction < 0:
                # For protective features (higher is better), push feature up.
                updated_vals = np.minimum(X_mod[:, i] + delta, 1.0)
                moved_mask = updated_vals > (X_test[:, i] + eps)
            else:
                # For risk-increasing features (lower is better), push feature down.
                updated_vals = np.maximum(X_mod[:, i] - delta, 0.0)
                moved_mask = updated_vals < (X_test[:, i] - eps)

            checks = int(np.sum(moved_mask))
            if checks == 0:
                continue

            X_mod[:, i] = updated_vals
            mod_scores = model.predict_proba(X_mod)[:, 1]
            violations = int(np.sum((mod_scores > baseline + eps) & moved_mask))

            total_checks += checks
            total_violations += violations

        if total_checks == 0:
            continue

        report.append(
            {
                "check_type": "single_feature",
                "feature": name,
                "direction": direction,
                "checks": total_checks,
                "violations": total_violations,
                "violation_rate": float(total_violations / total_checks),
                "violation_rate_ci95_low": _violation_ci(total_violations, total_checks)[0],
                "violation_rate_ci95_high": _violation_ci(total_violations, total_checks)[1],
                "tested_steps": ",".join(str(s) for s in steps),
                "epsilon": eps,
            }
        )

    # Stress-test pairwise improvements on constrained features to catch interaction failures.
    constrained = [
        (i, name, MONOTONIC_CONSTRAINTS.get(name, 0))
        for i, name in enumerate(feature_names)
        if MONOTONIC_CONSTRAINTS.get(name, 0) != 0
    ]
    all_pairs = list(combinations(constrained, 2))
    if all_pairs:
        rng = np.random.default_rng(42)
        sample_n = min(int(MONOTONIC_PAIR_STRESS_SAMPLES), len(all_pairs))
        sample_idx = rng.choice(len(all_pairs), size=sample_n, replace=False)

        for idx in sample_idx:
            (i1, n1, d1), (i2, n2, d2) = all_pairs[int(idx)]

            total_checks = 0
            total_violations = 0
            for step in steps:
                span1 = float(np.nanmax(X_test[:, i1]) - np.nanmin(X_test[:, i1]))
                span2 = float(np.nanmax(X_test[:, i2]) - np.nanmin(X_test[:, i2]))
                delta1 = span1 * float(step)
                delta2 = span2 * float(step)
                if delta1 <= 0 and delta2 <= 0:
                    continue

                X_mod = X_test.copy()
                if delta1 > 0:
                    if d1 < 0:
                        updated1 = np.minimum(X_mod[:, i1] + delta1, 1.0)
                        moved1 = updated1 > (X_test[:, i1] + eps)
                    else:
                        updated1 = np.maximum(X_mod[:, i1] - delta1, 0.0)
                        moved1 = updated1 < (X_test[:, i1] - eps)
                    X_mod[:, i1] = updated1
                else:
                    moved1 = np.zeros(len(X_test), dtype=bool)

                if delta2 > 0:
                    if d2 < 0:
                        updated2 = np.minimum(X_mod[:, i2] + delta2, 1.0)
                        moved2 = updated2 > (X_test[:, i2] + eps)
                    else:
                        updated2 = np.maximum(X_mod[:, i2] - delta2, 0.0)
                        moved2 = updated2 < (X_test[:, i2] - eps)
                    X_mod[:, i2] = updated2
                else:
                    moved2 = np.zeros(len(X_test), dtype=bool)

                moved_mask = moved1 | moved2
                checks = int(np.sum(moved_mask))
                if checks == 0:
                    continue

                mod_scores = model.predict_proba(X_mod)[:, 1]
                violations = int(np.sum((mod_scores > baseline + eps) & moved_mask))

                total_checks += checks
                total_violations += violations

            if total_checks == 0:
                continue

            ci_low, ci_high = _violation_ci(total_violations, total_checks)
            report.append(
                {
                    "check_type": "pair_stress",
                    "feature": f"{n1}|{n2}",
                    "direction": f"{d1}|{d2}",
                    "checks": total_checks,
                    "violations": total_violations,
                    "violation_rate": float(total_violations / total_checks),
                    "violation_rate_ci95_low": ci_low,
                    "violation_rate_ci95_high": ci_high,
                    "tested_steps": ",".join(str(s) for s in steps),
                    "epsilon": eps,
                }
            )
    return report


def _metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float((recall_score(y_true, y_pred, zero_division=0) + recall_score(1 - y_true, 1 - y_pred, zero_division=0)) / 2.0),
    }


def _full_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_decision_threshold(y_valid: np.ndarray, y_valid_score: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.arange(
        THRESHOLD_TUNE_MIN,
        THRESHOLD_TUNE_MAX + (THRESHOLD_TUNE_STEP / 2.0),
        THRESHOLD_TUNE_STEP,
    )

    best_threshold = float(THRESHOLD)
    best_metrics = _metrics_at_threshold(y_valid, y_valid_score, best_threshold)
    objective = THRESHOLD_TUNE_OBJECTIVE.lower().strip()
    if objective not in {"f1", "accuracy", "precision", "recall", "balanced_accuracy"}:
        objective = "f1"

    for threshold in candidates:
        metrics = _metrics_at_threshold(y_valid, y_valid_score, float(threshold))
        if (
            metrics[objective] > best_metrics[objective]
            or (
                metrics[objective] == best_metrics[objective]
                and metrics["accuracy"] > best_metrics["accuracy"]
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _parse_reason_codes(reason_codes_json: str) -> set[str]:
    try:
        parsed = json.loads(reason_codes_json)
        if isinstance(parsed, list):
            return {str(v) for v in parsed}
    except Exception:
        pass
    return set()


def build_explainability_global_summary(
    explain_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    totals: dict[str, dict[str, float]] = {
        name: {
            "sum_abs": 0.0,
            "sum_signed": 0.0,
            "positive_count": 0.0,
            "rows": 0.0,
            "reason_count": 0.0,
        }
        for name in feature_names
    }

    for _, row in explain_df.iterrows():
        effects_raw = row.get("feature_effects", "{}")
        reasons = _parse_reason_codes(str(row.get("reason_codes", "[]")))
        try:
            effects = json.loads(effects_raw)
        except Exception:
            effects = {}

        for name in feature_names:
            val = float(effects.get(name, 0.0)) if isinstance(effects, dict) else 0.0
            totals[name]["sum_abs"] += abs(val)
            totals[name]["sum_signed"] += val
            totals[name]["positive_count"] += float(val > 0.0)
            totals[name]["rows"] += 1.0
            totals[name]["reason_count"] += float(name in reasons)

    rows = []
    for name in feature_names:
        n = max(totals[name]["rows"], 1.0)
        rows.append(
            {
                "feature": name,
                "mean_abs_effect": float(totals[name]["sum_abs"] / n),
                "mean_signed_effect": float(totals[name]["sum_signed"] / n),
                "positive_effect_rate": float(totals[name]["positive_count"] / n),
                "top_reason_rate": float(totals[name]["reason_count"] / n),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(by=["mean_abs_effect", "top_reason_rate"], ascending=[False, False])
    return out_df


def build_reason_code_stability_report(
    model,
    X_test: np.ndarray,
    X_reference: np.ndarray,
    feature_names: list[str],
    lattice_groups: list[list[str]],
    base_explain_df: pd.DataFrame,
    decision_threshold: float,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_rows = int(min(EXPLAIN_STABILITY_MAX_ROWS, len(X_test)))
    n_perturb = int(max(1, EXPLAIN_STABILITY_PERTURBATIONS))
    sigma = float(max(1e-6, EXPLAIN_STABILITY_SIGMA))

    if n_rows == 0:
        empty_df = pd.DataFrame(
            columns=["test_index", "perturbations", "exact_match_rate", "jaccard_mean"]
        )
        return empty_df, {
            "rows_evaluated": 0,
            "perturbations": n_perturb,
            "sigma": sigma,
            "overall_exact_match_rate": 0.0,
            "overall_jaccard_mean": 0.0,
        }

    base_subset = base_explain_df.head(n_rows).copy()
    base_reasons = {
        int(row["test_index"]): _parse_reason_codes(str(row["reason_codes"]))
        for _, row in base_subset.iterrows()
    }

    rng = np.random.default_rng(42)
    exact_counts = np.zeros(n_rows, dtype=float)
    jaccard_sums = np.zeros(n_rows, dtype=float)

    for _ in range(n_perturb):
        noise = rng.normal(loc=0.0, scale=sigma, size=X_test[:n_rows].shape)
        X_pert = np.clip(X_test[:n_rows] + noise, 0.0, 1.0)
        pert_df = build_explainability_report(
            model=model,
            X=X_pert,
            feature_names=feature_names,
            X_reference=X_reference,
            lattice_groups=lattice_groups,
            top_k=top_k,
            decision_threshold=decision_threshold,
        )

        for i, (_, row) in enumerate(pert_df.iterrows()):
            base_set = base_reasons.get(i, set())
            pert_set = _parse_reason_codes(str(row.get("reason_codes", "[]")))
            intersection = len(base_set & pert_set)
            union = len(base_set | pert_set)
            jaccard = float(intersection / union) if union > 0 else 1.0
            jaccard_sums[i] += jaccard
            exact_counts[i] += float(base_set == pert_set)

    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "test_index": int(i),
                "perturbations": int(n_perturb),
                "exact_match_rate": float(exact_counts[i] / n_perturb),
                "jaccard_mean": float(jaccard_sums[i] / n_perturb),
            }
        )
    out_df = pd.DataFrame(rows)
    summary = {
        "rows_evaluated": int(n_rows),
        "perturbations": int(n_perturb),
        "sigma": float(sigma),
        "overall_exact_match_rate": float(out_df["exact_match_rate"].mean()),
        "overall_jaccard_mean": float(out_df["jaccard_mean"].mean()),
    }
    return out_df, summary


def _score_band(score: float, threshold: float) -> str:
    delta = float(score - threshold)
    if delta < 0.05:
        return "near_threshold"
    if delta < 0.10:
        return "moderate_risk"
    if delta < 0.20:
        return "high_risk"
    return "very_high_risk"


def _policy_interpretation(approve_rate: float, precision: float, recall: float) -> str:
    if approve_rate >= 0.65 and recall < 0.75:
        return "approval_focused_policy_high_approvals_lower_risk_capture"
    if approve_rate <= 0.35 and precision >= 0.75:
        return "risk_containment_policy_lower_approvals_higher_precision"
    if recall >= 0.85:
        return "catch_defaults_policy_high_recall"
    return "balanced_policy_tradeoff"


def _ece_mce(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    df = pd.DataFrame({"y_true": y_true.astype(int), "y_score": y_score.astype(float)})
    df["score_bin"] = pd.qcut(df["y_score"], q=n_bins, labels=False, duplicates="drop")
    rep = (
        df.groupby("score_bin", observed=True)
        .agg(n=("y_true", "size"), avg_pred=("y_score", "mean"), event_rate=("y_true", "mean"))
        .reset_index()
    )
    rep["abs_gap"] = np.abs(rep["avg_pred"] - rep["event_rate"])
    total_n = max(int(rep["n"].sum()), 1)
    ece = float(np.sum((rep["n"] / total_n) * rep["abs_gap"]))
    mce = float(rep["abs_gap"].max())
    return ece, mce


def _build_tuning_parity_report(metrics_dir: Path) -> dict[str, Any]:
    lattice_cfg_path = metrics_dir / "lattice_tuning_run_config.json"
    baseline_cfg_path = metrics_dir / "baseline_tuning_run_config.json"

    report: dict[str, Any] = {
        "status": "incomplete",
        "lattice": None,
        "baselines": None,
        "parity": {
            "same_cv_splits": False,
            "same_parity_total_trials": False,
            "details": "missing_tuning_configs",
        },
    }
    if not lattice_cfg_path.exists() or not baseline_cfg_path.exists():
        return report

    try:
        lattice_cfg = json.loads(lattice_cfg_path.read_text(encoding="utf-8"))
        baseline_cfg = json.loads(baseline_cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["status"] = f"error:{type(exc).__name__}"
        report["parity"]["details"] = "config_parse_error"
        return report

    lattice_trials = int(lattice_cfg.get("parity_total_trials", 0))
    baseline_trials = int(baseline_cfg.get("parity_total_trials", 0))
    same_cv = int(lattice_cfg.get("cv_splits", -1)) == int(baseline_cfg.get("cv_splits", -2))
    same_trials = lattice_trials > 0 and baseline_trials > 0 and lattice_trials == baseline_trials

    report.update(
        {
            "status": "ok",
            "lattice": lattice_cfg,
            "baselines": baseline_cfg,
            "parity": {
                "same_cv_splits": bool(same_cv),
                "same_parity_total_trials": bool(same_trials),
                "details": "ok" if (same_cv and same_trials) else "mismatch_detected",
            },
        }
    )
    return report


def _best_operating_point(frontier_df: pd.DataFrame, metric: str) -> dict[str, float] | None:
    if frontier_df.empty or metric not in frontier_df.columns:
        return None
    vals = pd.to_numeric(frontier_df[metric], errors="coerce")
    if vals.notna().sum() == 0:
        return None
    row = frontier_df.loc[int(vals.idxmax())]
    return {
        "threshold": float(row["threshold"]),
        "approve_rate": float(row["approve_rate"]),
        "deny_rate": float(row["deny_rate"]),
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
    }


def _load_optional_csv_records(path: Path, max_rows: int = 200) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        df = pd.read_csv(path)
        if max_rows > 0 and len(df) > max_rows:
            df = df.head(max_rows)
        return {
            "status": "ok",
            "path": str(path),
            "rows": df.to_dict(orient="records"),
        }
    except Exception as exc:
        return {
            "status": f"error:{type(exc).__name__}",
            "path": str(path),
        }


def build_model_competitiveness_report(
    predictor_metrics: dict[str, Any],
    baselines_df: pd.DataFrame,
    primary_metric: str,
) -> dict[str, Any]:
    metric = primary_metric
    if metric not in {"auc", "pr_auc", "f1", "accuracy"}:
        metric = "auc"

    numeric_baselines = baselines_df.copy()
    for c in ["auc", "pr_auc", "f1", "accuracy"]:
        if c in numeric_baselines.columns:
            numeric_baselines[c] = pd.to_numeric(numeric_baselines[c], errors="coerce")

    best_baseline_row = None
    if metric in numeric_baselines.columns and numeric_baselines[metric].notna().any():
        best_baseline_row = numeric_baselines.loc[numeric_baselines[metric].idxmax()].to_dict()

    report = {
        "primary_metric": metric,
        "xcreditscore": {
            "auc": float(predictor_metrics.get("auc", 0.0)),
            "pr_auc": float(predictor_metrics.get("pr_auc", 0.0)),
            "f1": float(predictor_metrics.get("f1", 0.0)),
            "accuracy": float(predictor_metrics.get("accuracy", 0.0)),
        },
        "best_baseline": None,
        "delta_vs_best_baseline": None,
        "xcreditscore_beats_best_baseline": False,
    }
    if best_baseline_row is not None:
        baseline_metric = float(best_baseline_row.get(metric, 0.0) or 0.0)
        xscore_metric = float(report["xcreditscore"].get(metric, 0.0))
        report["best_baseline"] = {
            "model": str(best_baseline_row.get("model", "unknown")),
            "auc": float(best_baseline_row.get("auc", 0.0) or 0.0),
            "pr_auc": float(best_baseline_row.get("pr_auc", 0.0) or 0.0),
            "f1": float(best_baseline_row.get("f1", 0.0) or 0.0),
            "accuracy": float(best_baseline_row.get("accuracy", 0.0) or 0.0),
        }
        report["delta_vs_best_baseline"] = float(xscore_metric - baseline_metric)
        report["xcreditscore_beats_best_baseline"] = bool(xscore_metric > baseline_metric)
    return report


def main() -> None:
    dirs = ensure_dirs(PACK_DIR)
    print(f"Run profile: {RUN_PROFILE}")

    tuning_parity_report = _build_tuning_parity_report(dirs["metrics"])
    with open(dirs["metrics"] / "compute_budget_parity_report.json", "w", encoding="utf-8") as f:
        json.dump(tuning_parity_report, f, indent=2)

    lattice_cfg = {
        "epochs": int(LATTICE_EPOCHS),
        "batch_size": int(LATTICE_BATCH_SIZE),
        "calibration_keypoints": int(LATTICE_CALIBRATION_KEYPOINTS),
        "lattice_size": int(LATTICE_SIZE),
        "learning_rate": float(LATTICE_LEARNING_RATE),
        "early_stopping_patience": int(LATTICE_EARLY_STOPPING_PATIENCE),
        "grouping_mode": str(LATTICE_GROUPING_MODE),
        "max_group_dim": int(LATTICE_MAX_GROUP_DIM),
    }

    tuned_lattice_path = dirs["metrics"] / "lattice_tuning_best.json"
    if LATTICE_USE_TUNED and tuned_lattice_path.exists():
        try:
            tuned = json.loads(tuned_lattice_path.read_text(encoding="utf-8"))
            lattice_cfg.update(
                {
                    "epochs": int(tuned.get("epochs", lattice_cfg["epochs"])),
                    "batch_size": int(tuned.get("batch_size", lattice_cfg["batch_size"])),
                    "calibration_keypoints": int(
                        tuned.get("calibration_keypoints", lattice_cfg["calibration_keypoints"])
                    ),
                    "lattice_size": int(tuned.get("lattice_size", lattice_cfg["lattice_size"])),
                    "learning_rate": float(tuned.get("learning_rate", lattice_cfg["learning_rate"])),
                    "early_stopping_patience": int(
                        tuned.get("early_stopping_patience", lattice_cfg["early_stopping_patience"])
                    ),
                    "grouping_mode": str(tuned.get("grouping_mode", lattice_cfg["grouping_mode"])),
                    "max_group_dim": int(tuned.get("max_group_dim", lattice_cfg["max_group_dim"])),
                }
            )
            print(f"Loaded tuned lattice config from: {tuned_lattice_path}")
        except Exception as exc:
            print(f"Warning: failed to load tuned lattice config: {type(exc).__name__}")

    raw = load_raw_data()
    cv_summary = None
    if CV_ENABLED:
        cv_summary = run_cross_validation(
            raw_df=raw,
            metrics_dir=dirs["metrics"],
            n_splits=CV_N_SPLITS,
            n_repeats=CV_N_REPEATS,
        )
    bundle, transformers = preprocess_and_split(raw)

    predictor_result = train_and_evaluate(
        bundle.X_train,
        bundle.y_train,
        bundle.X_valid,
        bundle.y_valid,
        bundle.X_test,
        bundle.y_test,
        bundle.feature_names,
        epochs=int(lattice_cfg["epochs"]),
        batch_size=int(lattice_cfg["batch_size"]),
        calibration_keypoints=int(lattice_cfg["calibration_keypoints"]),
        lattice_size=int(lattice_cfg["lattice_size"]),
        learning_rate=float(lattice_cfg["learning_rate"]),
        early_stopping_patience=int(lattice_cfg["early_stopping_patience"]),
        grouping_mode=str(lattice_cfg["grouping_mode"]),
        max_group_dim=int(lattice_cfg["max_group_dim"]),
        fixed_groups=LATTICE_FIXED_GROUPS,
        decision_threshold=float(THRESHOLD),
    )

    decision_threshold = float(THRESHOLD)
    y_valid_score = predictor_result.model.predict_proba(bundle.X_valid)[:, 1]
    threshold_tuning = {
        "enabled": bool(ENABLE_THRESHOLD_TUNING),
        "objective": THRESHOLD_TUNE_OBJECTIVE,
        "base_threshold": float(THRESHOLD),
        "selected_threshold": float(THRESHOLD),
    }
    if ENABLE_THRESHOLD_TUNING:
        tuned_threshold, tuned_valid_metrics = tune_decision_threshold(bundle.y_valid, y_valid_score)
        decision_threshold = float(tuned_threshold)
        threshold_tuning["selected_threshold"] = decision_threshold
        threshold_tuning["valid_metrics_at_selected_threshold"] = tuned_valid_metrics

    y_test_proba_safe = np.clip(predictor_result.y_proba, 1e-6, 1 - 1e-6)
    # Ensure exported metrics are computed at the final decision threshold.
    predictor_result.metrics = {
        "auc": predictor_result.metrics["auc"],
        "pr_auc": float(average_precision_score(bundle.y_test, predictor_result.y_proba)),
        "brier": float(brier_score_loss(bundle.y_test, predictor_result.y_proba)),
        "log_loss": float(log_loss(bundle.y_test, y_test_proba_safe)),
        **_full_metrics_at_threshold(bundle.y_test, predictor_result.y_proba, decision_threshold),
    }

    denied_indices = np.where(predictor_result.y_proba >= decision_threshold)[0]
    cfs = batch_counterfactuals(
        predictor_result.model,
        bundle.X_test,
        denied_indices,
        bundle.feature_names,
        lattice_groups=predictor_result.lattice_groups,
        max_cases=COUNTERFACTUAL_MAX_CASES,
        exact_max_cases=COUNTERFACTUAL_EXACT_MAX_CASES,
        decision_threshold=decision_threshold,
    )

    baseline_test_probas: dict[str, np.ndarray] = {}
    baseline_fitted_models: dict[str, Any] = {}
    baseline_results = run_baselines(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        decision_threshold=decision_threshold,
        fitted_probas_out=baseline_test_probas,
        fitted_models_out=baseline_fitted_models,
    )

    monotone_xgb_metrics, monotone_xgb_model = train_monotone_xgboost(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        list(bundle.feature_names),
        decision_threshold=decision_threshold,
    )
    baseline_results["monotone_xgboost"] = monotone_xgb_metrics

    # Interpretable-by-design competitors. Ensure the 40-trial parity-
    # controlled tuning has been performed once; the cached best-params file is
    # then loaded by train_ebm/train_monotone_gam.
    interpretable_tuning_path = (
        dirs["metrics"] / "interpretable_baseline_tuning_best.json"
    )
    if not interpretable_tuning_path.exists() and os.getenv(
        "SKIP_INTERPRETABLE_TUNING", "0"
    ).strip() != "1":
        print(
            "[interpretable_baselines] no tuned params on disk; running "
            "40-trial parity tuning (this happens once and is cached)"
        )
        try:
            from .tune_interpretable_baselines import run_tuning as _run_interpretable_tuning

            _run_interpretable_tuning()
        except Exception as exc:
            print(
                "[interpretable_baselines] tuning failed with "
                f"{type(exc).__name__}: {exc}; falling back to library defaults"
            )

    ebm_metrics, ebm_model = train_ebm(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        list(bundle.feature_names),
        decision_threshold=decision_threshold,
    )
    baseline_results["ebm"] = ebm_metrics

    monotone_gam_metrics, monotone_gam_model = train_monotone_gam(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        list(bundle.feature_names),
        decision_threshold=decision_threshold,
    )
    baseline_results["monotone_gam"] = monotone_gam_metrics

    # On HELOC, LR is within ~0.003 AUC of XGBoost. Print the gap so every
    # run records a current number for the linear-versus-boosted comparison.
    _lr = baseline_results.get("logistic_regression", {})
    _xgb = baseline_results.get("xgboost", {})
    _lr_auc = _lr.get("auc")
    _xgb_auc = _xgb.get("auc")
    lr_vs_xgb_gap_record: dict[str, Any] = {"status": "unavailable"}
    if (
        isinstance(_lr_auc, (int, float))
        and isinstance(_xgb_auc, (int, float))
        and not (np.isnan(float(_lr_auc)) or np.isnan(float(_xgb_auc)))
    ):
        _gap = float(_xgb_auc) - float(_lr_auc)
        from .config import DATASET_NAME as _DATASET_NAME_FOR_GAP
        print(
            f"[dataset={_DATASET_NAME_FOR_GAP}] LR-vs-XGBoost AUC gap: "
            f"xgb={float(_xgb_auc):.6f}, lr={float(_lr_auc):.6f}, "
            f"gap(xgb-lr)={_gap:+.6f}"
        )
        lr_vs_xgb_gap_record = {
            "status": "ok",
            "dataset": _DATASET_NAME_FOR_GAP,
            "xgboost_auc": float(_xgb_auc),
            "logistic_regression_auc": float(_lr_auc),
            "gap_xgb_minus_lr": _gap,
        }

    monotonic_report = monotonic_sanity_report(
        predictor_result.model,
        bundle.X_test,
        bundle.feature_names,
    )

    if monotone_xgb_model is not None:
        monotone_xgb_sanity = monotonic_sanity_report(
            monotone_xgb_model,
            bundle.X_test,
            list(bundle.feature_names),
        )
    else:
        monotone_xgb_sanity = []

    if ebm_model is not None:
        ebm_sanity = monotonic_sanity_report(
            ebm_model,
            bundle.X_test,
            list(bundle.feature_names),
        )
    else:
        ebm_sanity = []

    if monotone_gam_model is not None:
        monotone_gam_sanity = monotonic_sanity_report(
            monotone_gam_model,
            bundle.X_test,
            list(bundle.feature_names),
        )
    else:
        monotone_gam_sanity = []

    metrics_df = pd.DataFrame([{"model": "xcreditscore", **predictor_result.metrics}])
    metrics_df.to_csv(dirs["metrics"] / "xcreditscore_metrics.csv", index=False)

    baselines_df = pd.DataFrame(
        [{"model": name, **vals} for name, vals in baseline_results.items()]
    )
    baselines_df.to_csv(dirs["metrics"] / "baseline_metrics.csv", index=False)

    model_competitiveness = build_model_competitiveness_report(
        predictor_metrics=predictor_result.metrics,
        baselines_df=baselines_df,
        primary_metric=PRIMARY_COMPARISON_METRIC,
    )

    with open(dirs["metrics"] / "model_competitiveness.json", "w", encoding="utf-8") as f:
        json.dump(model_competitiveness, f, indent=2)

    preds_df = pd.DataFrame(
        {
            "y_true": bundle.y_test,
            "y_proba_bad": predictor_result.y_proba,
            "y_pred_bad": (predictor_result.y_proba >= decision_threshold).astype(int),
            "decision": np.where(predictor_result.y_proba >= decision_threshold, "Deny", "Approve"),
        }
    )
    preds_df.to_csv(dirs["predictions"] / "xcreditscore_test_predictions.csv", index=False)

    explain_df = build_explainability_report(
        model=predictor_result.model,
        X=bundle.X_test,
        feature_names=bundle.feature_names,
        X_reference=bundle.X_train,
        lattice_groups=predictor_result.lattice_groups,
        top_k=4,
        decision_threshold=decision_threshold,
    )
    explain_df.to_csv(dirs["predictions"] / "explainability_report.csv", index=False)

    # ---------------------------------------------------------------------- #
    # Two-player Shapley + pure interaction, median reference.              #
    # Additive artifact; the direction-informed report above stays untouched. #
    # ---------------------------------------------------------------------- #
    if predictor_result.lattice_groups and hasattr(predictor_result.model, "model"):
        from .shapley_attribution import build_shapley_2d_report, side_by_side_audit

        try:
            shapley_df, shapley_summary = build_shapley_2d_report(
                model=predictor_result.model,
                X=bundle.X_test,
                feature_names=list(bundle.feature_names),
                X_reference=bundle.X_train,
                lattice_groups=predictor_result.lattice_groups,
            )
            if not shapley_df.empty:
                shapley_df.to_csv(
                    dirs["predictions"] / "explainability_shapley_2d_report.csv",
                    index=False,
                )
                with open(
                    dirs["metrics"] / "explainability_shapley_2d_summary.json",
                    "w",
                    encoding="utf-8",
                ) as _f:
                    json.dump(shapley_summary, _f, indent=2)
                print(
                    "[shapley-2d] emitted "
                    f"{shapley_summary['n_rows_total']} instance-block rows "
                    f"(baseline={shapley_summary['baseline']}, "
                    f"max shapley identity residual="
                    f"{shapley_summary['shapley_efficiency_max_abs_residual']:.3e}, "
                    "max main-effect identity residual="
                    f"{shapley_summary['main_effect_max_abs_residual']:.3e})"
                )

                # Self-audit — 5 random test instances, old vs new phi side by side.
                audit_df = side_by_side_audit(
                    model=predictor_result.model,
                    X=bundle.X_test,
                    feature_names=list(bundle.feature_names),
                    X_reference=bundle.X_train,
                    lattice_groups=predictor_result.lattice_groups,
                    n_instances=5,
                    seed=42,
                )
                audit_df.to_csv(
                    dirs["metrics"] / "explainability_shapley_2d_audit_5_instances.csv",
                    index=False,
                )
                print(
                    "[shapley-2d] side-by-side audit (5 random test instances × "
                    f"{shapley_summary['n_2d_blocks']} 2-D blocks):"
                )
                # Print a compact side-by-side view for the first few audit rows.
                for _, r in audit_df.head(15).iterrows():
                    print(
                        f"  test_idx={int(r['test_index']):>4} block={int(r['block_index'])} "
                        f"({r['feature_a']} × {r['feature_b']}): "
                        f"old phi=({r['old_phi_a']:+.4f}, {r['old_phi_b']:+.4f})  "
                        f"new phi=({r['new_phi_a']:+.4f}, {r['new_phi_b']:+.4f})  "
                        f"iota={r['new_pure_interaction']:+.4f}  "
                        f"residuals: old={r['old_identity_residual']:.2e} "
                        f"new={r['new_identity_residual']:.2e}"
                    )
        except Exception as _exc:
            print(
                f"[shapley-2d] skipped: {type(_exc).__name__}: {_exc}; existing "
                "explainability_report.csv is unaffected."
            )

    explain_global_df = build_explainability_global_summary(
        explain_df=explain_df,
        feature_names=bundle.feature_names,
    )
    explain_global_df.to_csv(
        dirs["metrics"] / "explainability_global_contributions.csv",
        index=False,
    )

    stability_df, stability_summary = build_reason_code_stability_report(
        model=predictor_result.model,
        X_test=bundle.X_test,
        X_reference=bundle.X_train,
        feature_names=bundle.feature_names,
        lattice_groups=predictor_result.lattice_groups,
        base_explain_df=explain_df,
        decision_threshold=decision_threshold,
        top_k=4,
    )
    stability_df.to_csv(
        dirs["metrics"] / "explainability_reason_stability.csv",
        index=False,
    )
    with open(dirs["metrics"] / "explainability_reason_stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(stability_summary, f, indent=2)

    # ----------------------------------------------------------------------
    # Head-to-head KernelSHAP + LIME + exact-lattice-logit
    # benchmark across every model. The existing exact-lattice-only stability
    # report above stays untouched; this appends a new comparison alongside
    # so both can be read. Gated by SKIP_EXPLAIN_STABILITY_BENCHMARK
    # (set to 1 for compute-constrained runs).
    # ----------------------------------------------------------------------
    from .explain_stability_benchmark import maybe_run_from_train_all as _run_stability_benchmark

    _stability_bench_models: dict[str, Any] = {}
    for _n, _m in baseline_fitted_models.items():
        if _m is not None:
            _stability_bench_models[_n] = _m
    for _n, _m in (
        ("monotone_xgboost", monotone_xgb_model),
        ("ebm", ebm_model),
        ("monotone_gam", monotone_gam_model),
    ):
        if _m is not None:
            _stability_bench_models[_n] = _m
    try:
        _run_stability_benchmark(
            predictor_model=predictor_result.model,
            predictor_lattice_groups=predictor_result.lattice_groups,
            baseline_models=_stability_bench_models,
            X_train_ref=bundle.X_train,
            X_test=bundle.X_test,
            feature_names=list(bundle.feature_names),
            decision_threshold=decision_threshold,
            output_dir=dirs["metrics"],
        )
    except Exception as exc:
        print(
            f"[stability-bench] failed: {type(exc).__name__}: {exc}; existing "
            "stability report is unaffected."
        )

    cf_rows = []
    for r in cfs:
        score_band = _score_band(r.original_score, decision_threshold)
        cf_rows.append(
            {
                "test_index": r.index,
                "feasible": r.feasible,
                "recourse_band": r.recourse_band,
                "original_score_band": score_band,
                "solver_status": r.solver_status,
                "original_score": r.original_score,
                "new_score": r.new_score,
                "score_gap_to_threshold": r.score_gap_to_threshold,
                "total_cost": r.total_cost,
                "changed_features": json.dumps(r.changed_features),
            }
        )
    pd.DataFrame(cf_rows).to_csv(
        dirs["counterfactuals"] / "counterfactual_results.csv",
        index=False,
    )

    cf_df = pd.DataFrame(cf_rows)
    if not cf_df.empty:
        by_band = (
            cf_df.groupby("original_score_band", observed=True)
            .agg(
                cases=("test_index", "count"),
                feasible_rate=("feasible", "mean"),
                near_feasible_rate=("recourse_band", lambda s: float(np.mean(s == "near_feasible"))),
                median_total_cost=("total_cost", "median"),
                median_changes=("changed_features", lambda s: float(np.median([len(json.loads(v)) for v in s]))),
            )
            .reset_index()
        )
    else:
        by_band = pd.DataFrame(
            columns=[
                "original_score_band",
                "cases",
                "feasible_rate",
                "near_feasible_rate",
                "median_total_cost",
                "median_changes",
            ]
        )
    by_band.to_csv(dirs["metrics"] / "counterfactual_success_by_score_band.csv", index=False)

    pd.DataFrame(monotonic_report).to_csv(
        dirs["metrics"] / "monotonic_sanity_report.csv",
        index=False,
    )

    # Perturbation protocol against monotone-constrained XGBoost.
    pd.DataFrame(monotone_xgb_sanity).to_csv(
        dirs["metrics"] / "monotonic_sanity_report_monotone_xgboost.csv",
        index=False,
    )

    # Perturbation protocol against the interpretable-by-design baselines.
    pd.DataFrame(ebm_sanity).to_csv(
        dirs["metrics"] / "monotonic_sanity_report_ebm.csv",
        index=False,
    )
    pd.DataFrame(monotone_gam_sanity).to_csv(
        dirs["metrics"] / "monotonic_sanity_report_monotone_gam.csv",
        index=False,
    )

    # Constrained-vs-constrained monotonicity comparison table.
    def _sanity_totals(report_rows: list[dict]) -> dict[str, int | float]:
        checks = sum(int(r.get("checks", 0)) for r in report_rows)
        vios = sum(int(r.get("violations", 0)) for r in report_rows)
        rate = float(vios / checks) if checks > 0 else 0.0
        return {"total_checks": int(checks), "total_violations": int(vios), "violation_rate": rate}

    xcs_totals = _sanity_totals(monotonic_report)
    mxgb_totals = _sanity_totals(monotone_xgb_sanity)
    ebm_totals = _sanity_totals(ebm_sanity)
    mgam_totals = _sanity_totals(monotone_gam_sanity)
    monotone_xgb_status = str(monotone_xgb_metrics.get("status", "unknown"))
    ebm_status = str(ebm_metrics.get("status", "unknown"))
    monotone_gam_status = str(monotone_gam_metrics.get("status", "unknown"))
    monotonic_comparison_rows = [
        {"model": "xcreditscore", "status": "ok", **xcs_totals},
        {"model": "monotone_xgboost", "status": monotone_xgb_status, **mxgb_totals},
        {"model": "ebm", "status": ebm_status, **ebm_totals},
        {"model": "monotone_gam", "status": monotone_gam_status, **mgam_totals},
    ]
    pd.DataFrame(monotonic_comparison_rows).to_csv(
        dirs["metrics"] / "monotonic_sanity_comparison.csv",
        index=False,
    )
    print(
        "[monotone_xgboost] perturbation protocol: "
        f"violations={mxgb_totals['total_violations']} of {mxgb_totals['total_checks']} checks "
        f"(rate={mxgb_totals['violation_rate']:.6f})"
    )
    print(
        "[ebm] perturbation protocol: "
        f"violations={ebm_totals['total_violations']} of {ebm_totals['total_checks']} checks "
        f"(rate={ebm_totals['violation_rate']:.6f})"
    )
    print(
        "[monotone_gam] perturbation protocol: "
        f"violations={mgam_totals['total_violations']} of {mgam_totals['total_checks']} checks "
        f"(rate={mgam_totals['violation_rate']:.6f})"
    )

    # Documented AUC delta vs unconstrained xgboost.
    xgb_metrics = baseline_results.get("xgboost", {})
    xgb_auc = xgb_metrics.get("auc", np.nan)
    mxgb_auc = monotone_xgb_metrics.get("auc", np.nan)
    if isinstance(xgb_auc, (int, float)) and isinstance(mxgb_auc, (int, float)) and not (
        np.isnan(float(xgb_auc)) or np.isnan(float(mxgb_auc))
    ):
        auc_delta = float(mxgb_auc) - float(xgb_auc)
        print(
            "[monotone_xgboost] AUC vs unconstrained xgboost: "
            f"monotone={float(mxgb_auc):.6f}, unconstrained={float(xgb_auc):.6f}, "
            f"delta={auc_delta:+.6f}"
        )
        monotone_xgb_auc_delta = {
            "monotone_xgboost_auc": float(mxgb_auc),
            "unconstrained_xgboost_auc": float(xgb_auc),
            "auc_delta_monotone_minus_unconstrained": auc_delta,
            "status": "ok",
        }
    else:
        print(
            "[monotone_xgboost] AUC delta unavailable: "
            f"monotone_status={monotone_xgb_status}, "
            f"unconstrained_status={xgb_metrics.get('status', 'unknown')}"
        )
        monotone_xgb_auc_delta = {
            "monotone_xgboost_auc": None if not isinstance(mxgb_auc, (int, float)) or np.isnan(float(mxgb_auc)) else float(mxgb_auc),
            "unconstrained_xgboost_auc": None if not isinstance(xgb_auc, (int, float)) or np.isnan(float(xgb_auc)) else float(xgb_auc),
            "auc_delta_monotone_minus_unconstrained": None,
            "status": "unavailable",
        }
    with open(dirs["metrics"] / "monotone_xgboost_auc_delta.json", "w", encoding="utf-8") as f:
        json.dump(monotone_xgb_auc_delta, f, indent=2)

    # -----------------------------------------------------------------------
    # Statistical corrections for multiplicity and fold dependence.
    # -----------------------------------------------------------------------
    from .stats_corrections import (
        bootstrap_ci_mean,
        delong_paired_auc_test,
        find_threshold_for_denial_rate,
        find_threshold_for_precision,
        holm_bonferroni,
        matched_operating_point_row,
        nadeau_bengio_ttest,
        paired_bootstrap_metric_diff,
        paired_sign_flip_pvalue,
    )
    from .config import DATASET_NAME as _DATASET_NAME_B2

    # Consolidate every model's test-set positive-class probability. LR/RF/XGB
    # come from run_baselines via fitted_probas_out; the three new models are
    # re-scored from their fitted objects; XCreditScore uses the lattice's
    # already-computed y_proba. No refits, no perturbation of any metric row.
    test_probas: dict[str, np.ndarray] = {"xcreditscore": np.asarray(predictor_result.y_proba, dtype=float)}
    for _name, _proba in baseline_test_probas.items():
        test_probas[_name] = np.asarray(_proba, dtype=float)
    for _name, _model in (
        ("monotone_xgboost", monotone_xgb_model),
        ("ebm", ebm_model),
        ("monotone_gam", monotone_gam_model),
    ):
        if _model is None:
            continue
        try:
            test_probas[_name] = np.asarray(_model.predict_proba(bundle.X_test)[:, 1], dtype=float)
        except Exception as exc:
            print(f"[stats] skipped {_name}: predict_proba failed ({type(exc).__name__})")

    # ---- PR-AUC reconciliation on the test partition ------------------------
    # The paper's §4.1 claim was +0.0113 test-set PR-AUC lead over XGBoost;
    # Table 8 CV reported -0.001019 favouring XGBoost. Compute a paired 10 000-
    # sample bootstrap CI and a DeLong-equivalent test on the test partition,
    # and record the CV number alongside so both can be read in one place.
    pr_auc_reconciliation: dict[str, Any] = {
        "dataset": _DATASET_NAME_B2,
        "note": (
            "PR-AUC test-set difference reconciled with cross-validated delta; "
            "logistic regression's PR-AUC ranking is reported explicitly."
        ),
    }
    xcs_proba = test_probas.get("xcreditscore")
    xgb_proba = test_probas.get("xgboost")
    lr_proba = test_probas.get("logistic_regression")
    if xcs_proba is not None and xgb_proba is not None:
        pr_boot_xgb = paired_bootstrap_metric_diff(
            bundle.y_test, xcs_proba, xgb_proba, metric="pr_auc", n_boot=10_000, seed=42
        )
        auc_delong_xgb = delong_paired_auc_test(bundle.y_test, xcs_proba, xgb_proba)
        pr_auc_reconciliation["test_partition_xcreditscore_vs_xgboost"] = {
            "pr_auc_paired_bootstrap": pr_boot_xgb,
            "auc_delong": auc_delong_xgb,
        }
    if xcs_proba is not None and lr_proba is not None:
        pr_boot_lr = paired_bootstrap_metric_diff(
            bundle.y_test, xcs_proba, lr_proba, metric="pr_auc", n_boot=10_000, seed=42
        )
        pr_auc_reconciliation["test_partition_xcreditscore_vs_logistic_regression"] = {
            "pr_auc_paired_bootstrap": pr_boot_lr,
        }
    # Reference the CV PR-AUC delta from cv_model_metrics.csv if it exists.
    cv_metrics_path = dirs["metrics"] / "cv_model_metrics.csv"
    if cv_metrics_path.exists():
        try:
            cv_df_ref = pd.read_csv(cv_metrics_path)
            _x = cv_df_ref[cv_df_ref["model"] == "xcreditscore"][["repeat", "fold", "pr_auc"]].rename(columns={"pr_auc": "xcs"})
            _b = cv_df_ref[cv_df_ref["model"] == "xgboost"][["repeat", "fold", "pr_auc"]].rename(columns={"pr_auc": "xgb"})
            _m = _x.merge(_b, on=["repeat", "fold"], how="inner")
            if len(_m) > 0:
                _d = (_m["xcs"] - _m["xgb"]).to_numpy(dtype=float)
                pr_auc_reconciliation["cv_partition_xcreditscore_vs_xgboost"] = {
                    "n_fold_pairs": int(_d.shape[0]),
                    "mean_delta_pr_auc": float(_d.mean()),
                    "sd_delta_pr_auc": float(_d.std(ddof=1) if len(_d) > 1 else 0.0),
                }
        except Exception as exc:
            pr_auc_reconciliation["cv_partition_xcreditscore_vs_xgboost"] = {
                "status": f"error:{type(exc).__name__}"
            }

    # Explicit ranking commentary so the reader isn't left inferring it.
    pr_auc_ranking = sorted(
        [
            (name, float(baseline_results[name].get("pr_auc"))) if name != "xcreditscore"
            else (name, float(predictor_result.metrics.get("pr_auc", 0.0)))
            for name in (["xcreditscore"] + list(baseline_results.keys()))
            if (name == "xcreditscore") or (
                isinstance(baseline_results[name].get("pr_auc"), (int, float))
                and not pd.isna(baseline_results[name].get("pr_auc"))
            )
        ],
        key=lambda kv: kv[1],
        reverse=True,
    )
    pr_auc_reconciliation["pr_auc_ranking_test_partition"] = [
        {"model": name, "pr_auc": val} for name, val in pr_auc_ranking
    ]
    lr_rank = next(
        (i for i, (name, _) in enumerate(pr_auc_ranking, start=1) if name == "logistic_regression"),
        None,
    )
    pr_auc_reconciliation["logistic_regression_pr_auc_rank"] = lr_rank
    pr_auc_reconciliation["logistic_regression_leads_pr_auc"] = bool(lr_rank == 1)

    with open(dirs["metrics"] / "pr_auc_reconciliation.json", "w", encoding="utf-8") as f:
        json.dump(pr_auc_reconciliation, f, indent=2)

    # ---- Extend Table 8 (statistical_comparison) in-place -------------------
    # Compute paired sign-flip (original) + Nadeau-Bengio (correction) side by
    # side from the very same cv_model_metrics fold-pair deltas, then extend
    # statistical_comparison.csv with Holm-Bonferroni columns for BOTH tests.
    # No sidecar file, no dependency on an offline script running first.
    statistical_corrections: dict[str, Any] = {
        "dataset": _DATASET_NAME_B2,
        "corrections": [
            "holm_bonferroni_family_wise",
            "nadeau_bengio_corrected_resampled_t_test",
        ],
        "fold_dependence_acknowledgement": (
            "Repeated k-fold cross-validation produces overlapping training "
            "sets across fold-pairs; per-fold-pair differences are not "
            "independent. The Nadeau-Bengio (2003) correction inflates the "
            "paired-t variance by (1/n + n_test/n_train) so the reported "
            "p-values do not understate variance the way the naive paired "
            "test does."
        ),
    }
    table8_rows: list[dict[str, Any]] = []
    metrics_of_interest = ["auc", "pr_auc", "f1", "accuracy", "precision", "recall"]

    if cv_metrics_path.exists():
        try:
            cv_df_ref = pd.read_csv(cv_metrics_path)
            # Pin the reference baseline to xgboost — the comparisons concern
            # Table 8 as XCreditScore-vs-XGBoost specifically, not whichever
            # model happens to top model_competitiveness this run. Fall back
            # to the top model only if xgboost was skipped from CV entirely
            # (e.g. SKIP_XGBOOST=1).
            _cv_models = set(cv_df_ref["model"].unique())
            if "xgboost" in _cv_models:
                _bl_name = "xgboost"
            else:
                _bb_obj = model_competitiveness.get("best_baseline") or {}
                _bl_name = str(_bb_obj.get("model") or "xgboost")
            # Nadeau-Bengio needs n_test / n_train. In stratified k-fold on the
            # full dataset those are ceil(N/K) and N - ceil(N/K) respectively.
            n_full = int(len(bundle.y_train) + len(bundle.y_valid) + len(bundle.y_test))
            n_test_fold = int(math.ceil(n_full / max(CV_N_SPLITS, 1)))
            n_train_fold = int(n_full - n_test_fold)
            for metric in metrics_of_interest:
                _x = cv_df_ref[cv_df_ref["model"] == "xcreditscore"][["repeat", "fold", metric]].rename(columns={metric: "x"})
                _b = cv_df_ref[cv_df_ref["model"] == _bl_name][["repeat", "fold", metric]].rename(columns={metric: "b"})
                _m = _x.merge(_b, on=["repeat", "fold"], how="inner")
                if len(_m) == 0:
                    continue
                deltas = (_m["x"] - _m["b"]).to_numpy(dtype=float)
                ci_lo, ci_hi = bootstrap_ci_mean(deltas)
                p_sign = paired_sign_flip_pvalue(deltas)
                nb = nadeau_bengio_ttest(deltas, n_test=n_test_fold, n_train=n_train_fold)
                table8_rows.append(
                    {
                        # ---- original Table-8 columns (order preserved) -----
                        "metric": metric,
                        "n_pairs": int(deltas.shape[0]),
                        "mean_delta_xcredit_minus_baseline": float(deltas.mean()),
                        "delta_ci95_low": float(ci_lo),
                        "delta_ci95_high": float(ci_hi),
                        "paired_sign_flip_pvalue": float(p_sign),
                        "xcredit_better_on_more_folds": float((deltas > 0).mean()),
                        # ---- corrections appended (never mutate above) ------
                        "nadeau_bengio_t_stat": float(nb.get("t_stat", np.nan)),
                        "nadeau_bengio_pvalue": float(nb.get("p_value", np.nan)),
                        "nadeau_bengio_df": float(nb.get("df", np.nan)),
                        "nadeau_bengio_variance_multiplier": float(
                            nb.get("variance_multiplier", np.nan)
                        ),
                    }
                )
            # Holm-Bonferroni over the six metrics: apply to BOTH the sign-flip
            # p-values (the paper's original test) and the Nadeau-Bengio p-values
            # (the fold-dependence correction). Both evidentiary standards
            # therefore have a family-wise-safe reading available.
            if table8_rows:
                sign_p = [r["paired_sign_flip_pvalue"] for r in table8_rows]
                nb_p = [r["nadeau_bengio_pvalue"] for r in table8_rows]

                def _adjust_safe(pvals: list[float]) -> list[float]:
                    valid_idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
                    valid_pv = [pvals[i] for i in valid_idx]
                    adjusted_valid = holm_bonferroni(valid_pv)
                    out = [float("nan")] * len(pvals)
                    for i, val in zip(valid_idx, adjusted_valid):
                        out[i] = val
                    return out

                sign_adj = _adjust_safe(sign_p)
                nb_adj = _adjust_safe(nb_p)
                for row, sa, na in zip(table8_rows, sign_adj, nb_adj):
                    row["paired_sign_flip_pvalue_holm_bonferroni"] = float(sa)
                    row["nadeau_bengio_pvalue_holm_bonferroni"] = float(na)

            # Extend Table 8 in place — original columns first, corrections
            # after — so a diff against the prior file shows appended columns.
            pd.DataFrame(table8_rows).to_csv(
                dirs["metrics"] / "statistical_comparison.csv", index=False
            )
            statistical_corrections["results"] = table8_rows
            statistical_corrections["method"] = {
                "ci": "bootstrap_95pct_mean_delta",
                "ci_bootstrap_samples": 20000,
                "test_original": "paired_sign_flip_permutation_two_sided",
                "test_original_permutations": 20000,
                "test_correction": "nadeau_bengio_corrected_resampled_t_test",
                "multiplicity_adjustment": "holm_bonferroni_step_down",
            }
            statistical_corrections["nadeau_bengio_context"] = {
                "cv_n_splits": int(CV_N_SPLITS),
                "cv_n_repeats": int(CV_N_REPEATS),
                "n_test_used": int(n_test_fold),
                "n_train_used": int(n_train_fold),
                "n_dataset_rows_observed": int(n_full),
                "best_baseline_model": _bl_name,
            }
            # Rewrite the historical JSON companion so both surfaces agree.
            with open(
                dirs["metrics"] / "statistical_comparison.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "comparison_scope": "paired_cv_folds",
                        "fold_pairs": int(table8_rows[0]["n_pairs"]) if table8_rows else 0,
                        "xcreditscore_model": "xcreditscore",
                        "best_baseline_model": _bl_name,
                        "method": statistical_corrections["method"],
                        "fold_dependence_acknowledgement": statistical_corrections[
                            "fold_dependence_acknowledgement"
                        ],
                        "results": table8_rows,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            statistical_corrections["status"] = f"error:{type(exc).__name__}"
            print(f"[stats] statistical_comparison rebuild failed: {type(exc).__name__}: {exc}")

    # Print side-by-side p-values so the correction impact is visible.
    print("[stats] Table 8 side-by-side p-values (sign_flip / +Holm / NB / NB+Holm):")
    for row in table8_rows:
        print(
            f"  {row['metric']:<10} "
            f"sign={row['paired_sign_flip_pvalue']:.4f}  "
            f"sign_holm={row.get('paired_sign_flip_pvalue_holm_bonferroni', float('nan')):.4f}  "
            f"NB={row['nadeau_bengio_pvalue']:.4f}  "
            f"NB_holm={row.get('nadeau_bengio_pvalue_holm_bonferroni', float('nan')):.4f}"
        )

    # ---- matched operating-point comparisons --------------------------------
    # Match XCreditScore's own denial rate and precision (measured at the
    # tuned tau), then re-score every model at the τ that hits the same
    # denial-rate / precision on the test partition. Recall/precision/F1 at
    # each matched point are what should be interpreted.
    xcs_denial_rate = float((xcs_proba >= decision_threshold).mean()) if xcs_proba is not None else float("nan")
    xcs_pred = (xcs_proba >= decision_threshold).astype(int) if xcs_proba is not None else None
    if xcs_pred is not None:
        _tp = int(((bundle.y_test == 1) & (xcs_pred == 1)).sum())
        _fp = int(((bundle.y_test == 0) & (xcs_pred == 1)).sum())
        xcs_precision = float(_tp / max(_tp + _fp, 1))
    else:
        xcs_precision = float("nan")

    matched_denial_rows: list[dict[str, Any]] = []
    matched_precision_rows: list[dict[str, Any]] = []
    for model_name, proba in test_probas.items():
        if proba is None:
            continue
        # (i) equal denial-rate operating point
        if not math.isnan(xcs_denial_rate):
            thr_dr = find_threshold_for_denial_rate(proba, xcs_denial_rate)
            matched_denial_rows.append(
                {
                    "target_denial_rate": xcs_denial_rate,
                    **matched_operating_point_row(model_name, bundle.y_test, proba, thr_dr),
                }
            )
        # (ii) equal precision operating point
        if not math.isnan(xcs_precision):
            thr_pr, achieved_pr, recall_at = find_threshold_for_precision(
                bundle.y_test, proba, xcs_precision
            )
            row = matched_operating_point_row(model_name, bundle.y_test, proba, thr_pr)
            row["target_precision"] = xcs_precision
            row["achieved_precision_at_threshold"] = achieved_pr
            matched_precision_rows.append(row)

    pd.DataFrame(matched_denial_rows).to_csv(
        dirs["metrics"] / "matched_operating_points_denial_rate.csv", index=False
    )
    pd.DataFrame(matched_precision_rows).to_csv(
        dirs["metrics"] / "matched_operating_points_precision.csv", index=False
    )

    # ---- primary-endpoint pre-specification ---------------------------------
    primary_endpoint = {
        "dataset": _DATASET_NAME_B2,
        "primary_endpoint": "recall",
        "primary_endpoint_rationale": (
            "Recall is designated as the primary confirmatory endpoint before "
            "any test-set inference is drawn. Rationale: bad-risk detection "
            "under a financial-inclusion mandate — false negatives (defaults "
            "extended credit) impose direct portfolio loss on the lender, "
            "while false positives (denied but repayable applicants) impede an "
            "external policy goal the model does not control. This pre-"
            "specification is a §3.7 methodology commitment, not a post hoc "
            "endpoint pick."
        ),
        "exploratory_endpoints": [
            "auc",
            "pr_auc",
            "f1",
            "accuracy",
            "precision",
        ],
        "multiplicity_treatment": (
            "Holm-Bonferroni-adjusted p-values are reported alongside raw "
            "p-values for all six metrics; recall's primary status insulates "
            "it from family-wise dilution, but its Holm-adjusted p is "
            "disclosed for transparency."
        ),
    }
    with open(dirs["metrics"] / "primary_endpoint_prespecification.json", "w", encoding="utf-8") as f:
        json.dump(primary_endpoint, f, indent=2)

    # ---- class-balance framing ----------------------------------------------
    # Report the raw class ratio and justify PR-AUC on decision-cost grounds
    # rather than a blanket "class-imbalanced" label, which is inaccurate on
    # HELOC. The phrasing here reports numbers directly; no editorial label.
    y_test_arr = np.asarray(bundle.y_test, dtype=int).ravel()
    pos_rate = float((y_test_arr == 1).mean())
    neg_rate = 1.0 - pos_rate
    pr_auc_rationale = {
        "dataset": _DATASET_NAME_B2,
        "class_ratio": {"positive": pos_rate, "negative": neg_rate},
        "class_ratio_description": (
            f"Positive/negative ratio on the test partition: "
            f"{pos_rate * 100:.1f} / {neg_rate * 100:.1f}."
        ),
        "pr_auc_emphasis_rationale": (
            "PR-AUC is reported as a secondary discriminator on decision-cost "
            "grounds. Adverse-action denials carry asymmetric operational "
            "cost (regulatory disclosure, customer friction, appeal handling), "
            "so precision/recall trade-offs above ROC-AUC's rank-only view "
            "are policy-relevant. The prevalence-driven interpretation of "
            "PR-AUC is dataset-dependent — reported here as a raw ratio "
            "rather than assigned a categorical label."
        ),
    }
    with open(dirs["metrics"] / "pr_auc_emphasis_rationale.json", "w", encoding="utf-8") as f:
        json.dump(pr_auc_rationale, f, indent=2)

    calibration_df = pd.DataFrame(
        {
            "y_true": bundle.y_test.astype(int),
            "y_proba_bad": predictor_result.y_proba.astype(float),
        }
    )
    calibration_df["score_bin"] = pd.qcut(
        calibration_df["y_proba_bad"],
        q=10,
        labels=False,
        duplicates="drop",
    )
    calibration_report = (
        calibration_df.groupby("score_bin", observed=True)
        .agg(
            n=("y_true", "size"),
            avg_pred=("y_proba_bad", "mean"),
            event_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    calibration_report["abs_gap"] = np.abs(
        calibration_report["avg_pred"] - calibration_report["event_rate"]
    )
    calibration_report.to_csv(dirs["metrics"] / "calibration_report.csv", index=False)

    total_n = max(int(calibration_report["n"].sum()), 1)
    ece = float(np.sum((calibration_report["n"] / total_n) * calibration_report["abs_gap"]))
    mce = float(calibration_report["abs_gap"].max())
    with open(dirs["metrics"] / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ece": ece,
                "mce": mce,
                "bins": int(len(calibration_report)),
            },
            f,
            indent=2,
        )

    # Validation-only post-calibration comparison (raw vs isotonic vs Platt).
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_valid_score, bundle.y_valid)
    iso_test_score = np.clip(iso.predict(predictor_result.y_proba), 1e-6, 1 - 1e-6)

    platt = LogisticRegression(max_iter=2000, solver="lbfgs")
    platt.fit(y_valid_score.reshape(-1, 1), bundle.y_valid)
    platt_test_score = np.clip(
        platt.predict_proba(predictor_result.y_proba.reshape(-1, 1))[:, 1],
        1e-6,
        1 - 1e-6,
    )

    method_rows = []
    for method_name, scores in [
        ("raw", predictor_result.y_proba),
        ("isotonic", iso_test_score),
        ("platt", platt_test_score),
    ]:
        ece_m, mce_m = _ece_mce(bundle.y_test, scores, n_bins=10)
        method_rows.append(
            {
                "method": method_name,
                "auc": float(roc_auc_score(bundle.y_test, scores)),
                "pr_auc": float(average_precision_score(bundle.y_test, scores)),
                "brier": float(brier_score_loss(bundle.y_test, scores)),
                "log_loss": float(log_loss(bundle.y_test, np.clip(scores, 1e-6, 1 - 1e-6))),
                "ece": float(ece_m),
                "mce": float(mce_m),
            }
        )
    pd.DataFrame(method_rows).to_csv(dirs["metrics"] / "calibration_method_comparison.csv", index=False)

    prob_true, prob_pred = calibration_curve(
        bundle.y_test,
        predictor_result.y_proba,
        n_bins=10,
        strategy="quantile",
    )
    plt.figure(figsize=(5.6, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1.6, label="XCreditScore")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("XCreditScore Reliability Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "xcreditscore_reliability_curve.png", dpi=400)
    plt.close()

    frontier_rows = []
    for thr in np.arange(THRESHOLD_TUNE_MIN, THRESHOLD_TUNE_MAX + (THRESHOLD_TUNE_STEP / 2.0), THRESHOLD_TUNE_STEP):
        t = float(thr)
        m = _full_metrics_at_threshold(bundle.y_test, predictor_result.y_proba, t)
        y_pred = (predictor_result.y_proba >= t).astype(int)
        frontier_rows.append(
            {
                "threshold": t,
                "approve_rate": float(np.mean(y_pred == 0)),
                "deny_rate": float(np.mean(y_pred == 1)),
                "policy_interpretation": _policy_interpretation(
                    approve_rate=float(np.mean(y_pred == 0)),
                    precision=float(m["precision"]),
                    recall=float(m["recall"]),
                ),
                **m,
            }
        )
    frontier_df = pd.DataFrame(frontier_rows)
    frontier_df.to_csv(dirs["metrics"] / "threshold_frontier.csv", index=False)

    objective_sweep_summary = {
        "f1": _best_operating_point(frontier_df, "f1"),
        "precision": _best_operating_point(frontier_df, "precision"),
        "recall": _best_operating_point(frontier_df, "recall"),
    }

    RocCurveDisplay.from_predictions(bundle.y_test, predictor_result.y_proba)
    plt.title("XCreditScore ROC Curve")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "xcreditscore_roc_curve.png", dpi=400)
    plt.close()

    PrecisionRecallDisplay.from_predictions(bundle.y_test, predictor_result.y_proba)
    plt.title("XCreditScore Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "xcreditscore_pr_curve.png", dpi=400)
    plt.close()

    changed_counts = [int(len(r.changed_features)) for r in cfs]
    top_changed = {}
    for r in cfs:
        for name in r.changed_features:
            top_changed[name] = top_changed.get(name, 0) + 1
    top_changed_rows = [
        {"feature": k, "count": int(v)} for k, v in sorted(top_changed.items(), key=lambda kv: kv[1], reverse=True)
    ]
    pd.DataFrame(top_changed_rows).to_csv(
        dirs["metrics"] / "counterfactual_top_changed_features.csv",
        index=False,
    )
    with open(dirs["metrics"] / "counterfactual_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": int(len(cfs)),
                "feasible_rate": float(sum(int(r.feasible) for r in cfs) / max(len(cfs), 1)),
                "median_total_cost": float(np.median([float(r.total_cost) for r in cfs])) if cfs else 0.0,
                "median_changed_features": float(np.median(changed_counts)) if changed_counts else 0.0,
            },
            f,
            indent=2,
        )

    # ----------------------------------------------------------------------
    # Full-population recourse evaluation with Wilson CIs and a
    # per-plan validity check. Strictly additive: the 80-case artifacts
    # above stay untouched. Gated by SKIP_FULL_RECOURSE_EVALUATION so
    # compute-constrained runs can opt out; default is to run.
    # ----------------------------------------------------------------------
    if os.getenv("SKIP_FULL_RECOURSE_EVALUATION", "0").strip() != "1":
        from .recourse_full_evaluation import run_full_recourse_evaluation

        try:
            run_full_recourse_evaluation(
                model=predictor_result.model,
                X_test=bundle.X_test,
                denied_indices=denied_indices,
                feature_names=list(bundle.feature_names),
                lattice_groups=predictor_result.lattice_groups,
                decision_threshold=decision_threshold,
                output_dir=dirs["counterfactuals"],
                verbose=True,
            )
        except Exception as exc:
            print(
                f"[recourse-full] failed: {type(exc).__name__}: {exc}; the "
                "existing 80-case artifacts are unaffected."
            )
    else:
        print("[recourse-full] skipped (SKIP_FULL_RECOURSE_EVALUATION=1)")

    # ----------------------------------------------------------------------
    # §5.6 follow-up — head-to-head recourse: same MIP engine + same
    # registry + same denied cases, only the fitted predictor differs
    # (XCreditScore lattice vs monotone-XGBoost). Opt-in via env var
    # because Gurobi × N_denied × 2 predictors is expensive.
    # ----------------------------------------------------------------------
    if os.getenv("ENABLE_HEAD_TO_HEAD_RECOURSE", "0").strip() == "1":
        from .recourse_head_to_head import run_head_to_head

        if monotone_xgb_model is None:
            print("[recourse-h2h] skipped: monotone_xgb_model is None on this run")
        else:
            try:
                run_head_to_head(
                    lattice_model=predictor_result.model,
                    lattice_groups=predictor_result.lattice_groups,
                    monotone_xgb_model=monotone_xgb_model,
                    X_test=bundle.X_test,
                    denied_indices=denied_indices,
                    feature_names=list(bundle.feature_names),
                    decision_threshold=decision_threshold,
                    output_dir=dirs["counterfactuals"],
                    verbose=True,
                )
            except Exception as exc:
                print(
                    f"[recourse-h2h] failed: {type(exc).__name__}: {exc}; existing "
                    "recourse artifacts are unaffected."
                )

    # ----------------------------------------------------------------------
    # Group fairness. Only Taiwan ships protected demographic attributes
    # (SEX, MARRIAGE, EDUCATION, AGE), so this runs on that dataset alone and
    # is a no-op elsewhere. The registry already declares those attributes
    # immutable; this measures whether scores and offered recourse
    # nonetheless fall differently across groups.
    # ----------------------------------------------------------------------
    if str(os.getenv("DATASET_NAME", "heloc")).strip().lower() == "taiwan":
        try:
            from .fairness_evaluation import run_fairness_evaluation

            _proba = np.asarray(predictor_result.y_proba, dtype=float).ravel()
            _fair = run_fairness_evaluation(
                bundle, _proba, float(decision_threshold), dirs["metrics"]
            )
            _di = {a: b.get("disparate_impact_ratio")
                   for a, b in _fair["prediction_fairness"].items()}
            print(f"[fairness] disparate-impact ratios (four-fifths = 0.80): {_di}")
        except Exception as exc:
            print(
                f"[fairness] failed: {type(exc).__name__}: {exc}; existing "
                "artifacts are unaffected."
            )

    final_comparison_pack = {
        "run_profile": RUN_PROFILE,
        "threshold": float(decision_threshold),
        "xcreditscore_metrics": {
            k: float(v)
            for k, v in predictor_result.metrics.items()
            if k in {"auc", "pr_auc", "brier", "log_loss", "accuracy", "precision", "recall", "f1"}
        },
        "baseline_metrics": baselines_df.to_dict(orient="records"),
        "model_competitiveness": model_competitiveness,
        "objective_sweep": objective_sweep_summary,
        "feature_engineering_ablation": _load_optional_csv_records(
            dirs["metrics"] / "feature_engineering_ablation.csv"
        ),
        "threshold_objective_comparison": _load_optional_csv_records(
            dirs["metrics"] / "threshold_objective_comparison.csv"
        ),
        "counterfactual": {
            "summary": {
                "cases": int(len(cfs)),
                "feasible_rate": float(sum(int(r.feasible) for r in cfs) / max(len(cfs), 1)),
                "median_total_cost": float(np.median([float(r.total_cost) for r in cfs])) if cfs else 0.0,
                "median_changed_features": float(np.median(changed_counts)) if changed_counts else 0.0,
            },
            "success_by_score_band": by_band.to_dict(orient="records"),
        },
    }
    with open(dirs["metrics"] / "final_comparison_pack.json", "w", encoding="utf-8") as f:
        json.dump(final_comparison_pack, f, indent=2)

    summary_rows = []
    for metric_name, metric_val in final_comparison_pack["xcreditscore_metrics"].items():
        summary_rows.append(
            {
                "section": "xcreditscore",
                "name": metric_name,
                "value": float(metric_val),
            }
        )
    for row in baselines_df.to_dict(orient="records"):
        model_name = str(row.get("model", "unknown"))
        for metric_name in ["auc", "pr_auc", "f1", "accuracy"]:
            if metric_name in row and pd.notna(row[metric_name]):
                summary_rows.append(
                    {
                        "section": f"baseline:{model_name}",
                        "name": metric_name,
                        "value": float(row[metric_name]),
                    }
                )
    for objective, point in objective_sweep_summary.items():
        if point is None:
            continue
        summary_rows.append(
            {
                "section": f"objective_sweep:{objective}",
                "name": "threshold",
                "value": float(point["threshold"]),
            }
        )
        summary_rows.append(
            {
                "section": f"objective_sweep:{objective}",
                "name": "f1",
                "value": float(point["f1"]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        dirs["metrics"] / "final_comparison_pack.csv",
        index=False,
    )

    summary = {
        "run_profile": RUN_PROFILE,
        "cv_enabled": bool(CV_ENABLED),
        "cv_n_splits": int(CV_N_SPLITS),
        "cv_n_repeats": int(CV_N_REPEATS),
        "cv_summary_available": bool(cv_summary is not None),
        "model_family": predictor_result.model_family,
        "predictor_fallback_used": bool(predictor_result.model_family != "tensorflow_lattice"),
        "fallback_reason": predictor_result.fallback_reason,
        "threshold": decision_threshold,
        "total_test_rows": int(len(bundle.y_test)),
        "total_denied": int(len(denied_indices)),
        "counterfactual_cases_attempted": int(len(cfs)),
        "counterfactual_feasible_count": int(sum(int(r.feasible) for r in cfs)),
        "counterfactual_near_feasible_count": int(
            sum(int(r.recourse_band == "near_feasible") for r in cfs)
        ),
        "counterfactual_exact_max_cases": int(COUNTERFACTUAL_EXACT_MAX_CASES),
        "counterfactual_exact_time_limit": float(COUNTERFACTUAL_EXACT_TIME_LIMIT),
        "counterfactual_exact_mip_gap": float(COUNTERFACTUAL_EXACT_MIP_GAP),
        "counterfactual_refine_max_rounds": int(COUNTERFACTUAL_REFINE_MAX_ROUNDS),
        "lattice_grouping_mode": LATTICE_GROUPING_MODE,
        "lattice_max_group_dim": int(LATTICE_MAX_GROUP_DIM),
        "lattice_group_count": int(len(predictor_result.lattice_groups)),
        "threshold_tuning": threshold_tuning,
        "compute_budget_parity": tuning_parity_report.get("parity", {}),
    }
    with open(dirs["metrics"] / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    config_snapshot = {
        "run_profile": RUN_PROFILE,
        "threshold": float(decision_threshold),
        "feature_engineering_enabled": bool(FEATURE_ENGINEERING_ENABLED),
        "cv_enabled": bool(CV_ENABLED),
        "cv_n_splits": int(CV_N_SPLITS),
        "cv_n_repeats": int(CV_N_REPEATS),
        "lattice": {
            "epochs": int(lattice_cfg["epochs"]),
            "batch_size": int(lattice_cfg["batch_size"]),
            "calibration_keypoints": int(lattice_cfg["calibration_keypoints"]),
            "lattice_size": int(lattice_cfg["lattice_size"]),
            "learning_rate": float(lattice_cfg["learning_rate"]),
            "early_stopping_patience": int(lattice_cfg["early_stopping_patience"]),
            "grouping_mode": str(lattice_cfg["grouping_mode"]),
            "max_group_dim": int(lattice_cfg["max_group_dim"]),
            "fixed_groups": [list(g) for g in LATTICE_FIXED_GROUPS],
            "use_tuned": bool(LATTICE_USE_TUNED),
        },
        "counterfactual": {
            "max_cases": int(COUNTERFACTUAL_MAX_CASES),
            "exact_max_cases": int(COUNTERFACTUAL_EXACT_MAX_CASES),
            "exact_time_limit": float(COUNTERFACTUAL_EXACT_TIME_LIMIT),
            "exact_mip_gap": float(COUNTERFACTUAL_EXACT_MIP_GAP),
            "refine_max_rounds": int(COUNTERFACTUAL_REFINE_MAX_ROUNDS),
        },
        "threshold_tuning": threshold_tuning,
    }
    manifest = build_reproducibility_manifest(
        workspace_root=ARTIFACTS_DIR.parent,
        data_path=DATA_PATH,
        artifact_root=dirs["root"],
        config_snapshot=config_snapshot,
    )
    write_reproducibility_manifest(dirs["metrics"], manifest)

    baseline_status = {}
    for row in baselines_df.to_dict(orient="records"):
        model_name = str(row.get("model", "unknown"))
        baseline_status[model_name] = {
            "status": str(row.get("status", "unknown")),
            "error_type": str(row.get("error_type", "")),
            "fallback_used": bool(row.get("fallback_used", False)),
        }

    failure_transparency = {
        "predictor": {
            "model_family": predictor_result.model_family,
            "fallback_used": bool(predictor_result.model_family != "tensorflow_lattice"),
            "fallback_reason": predictor_result.fallback_reason,
        },
        "baselines": baseline_status,
    }

    best_baseline_auc = 0.0
    if not baselines_df.empty and "auc" in baselines_df.columns:
        baseline_auc = pd.to_numeric(baselines_df["auc"], errors="coerce")
        if baseline_auc.notna().any():
            best_baseline_auc = float(baseline_auc.max())

    with open(dirs["metrics"] / "final_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_profile": RUN_PROFILE,
                "config_hash_sha256": manifest.get("config_hash_sha256"),
                "threshold": float(decision_threshold),
                "xcreditscore": {
                    "auc": float(predictor_result.metrics.get("auc", 0.0)),
                    "pr_auc": float(predictor_result.metrics.get("pr_auc", 0.0)),
                    "f1": float(predictor_result.metrics.get("f1", 0.0)),
                    "accuracy": float(predictor_result.metrics.get("accuracy", 0.0)),
                },
                "best_baseline_auc": float(best_baseline_auc),
                "model_competitiveness": model_competitiveness,
                "failure_transparency": failure_transparency,
                "counterfactual": {
                    "cases": int(len(cfs)),
                    "feasible_rate": float(sum(int(r.feasible) for r in cfs) / max(len(cfs), 1)),
                    "near_feasible_count": int(
                        sum(int(r.recourse_band == "near_feasible") for r in cfs)
                    ),
                },
            },
            f,
            indent=2,
        )

    with open(dirs["metrics"] / "lr_vs_xgboost_gap.json", "w", encoding="utf-8") as f:
        json.dump(lr_vs_xgb_gap_record, f, indent=2)

    with open(dirs["config"] / "monotonic_constraints.json", "w", encoding="utf-8") as f:
        json.dump(MONOTONIC_CONSTRAINTS, f, indent=2)

    # Emit a documented sibling registry so it is possible to inspect not just
    # the direction but the domain rationale. Sourced from the active
    # dataset module's MONOTONIC_JUSTIFICATIONS dict (present for HELOC and
    # GMSC; empty dict is tolerated for future datasets without erroring).
    from .config import DATASET_NAME as _DATASET_NAME
    from .config import MONOTONIC_JUSTIFICATIONS as _MONOTONIC_JUSTIFICATIONS
    documented_registry = {
        "dataset": _DATASET_NAME,
        "features": {
            name: {
                "direction": int(direction),
                "justification": _MONOTONIC_JUSTIFICATIONS.get(name, ""),
            }
            for name, direction in MONOTONIC_CONSTRAINTS.items()
        },
    }
    with open(
        dirs["config"] / "monotonic_constraints_documented.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(documented_registry, f, indent=2)

    with open(dirs["config"] / "lattice_groups.json", "w", encoding="utf-8") as f:
        json.dump(predictor_result.lattice_groups, f, indent=2)

    save_imputer(transformers["imputer"], dirs["models"] / "imputer.joblib")
    save_scaler(transformers["scaler"], dirs["models"] / "scaler.joblib")
    save_predictor(predictor_result.model, dirs["models"])

    onnx_status = {"ok": False, "detail": "not_attempted"}
    onnx_parity = {"ok": False, "status": "skipped"}
    if hasattr(predictor_result.model, "model"):
        keras_model = getattr(predictor_result.model, "model")
        ok, detail = export_keras_to_onnx(
            keras_model,
            bundle.X_test[:8],
            dirs["models"] / "xcreditscore_model.onnx",
        )
        onnx_status = {"ok": ok, "detail": detail}
        if ok:
            # A 64-row smoke test understates the claim. Run parity over the
            # FULL test partition; it is computationally trivial and lets the
            # paper report deployment parity on every scored row rather than a
            # sample. The 64-row smoke result is retained alongside it.
            onnx_parity = run_onnx_parity_smoke(
                native_model=predictor_result.model,
                sample_input=bundle.X_test,
                onnx_path=dirs["models"] / "xcreditscore_model.onnx",
            )
            onnx_parity_smoke64 = run_onnx_parity_smoke(
                native_model=predictor_result.model,
                sample_input=bundle.X_test[:64],
                onnx_path=dirs["models"] / "xcreditscore_model.onnx",
            )
            onnx_parity["smoke_64_row_reference"] = onnx_parity_smoke64
            # A parity error only matters if it flips a lending decision;
            # count decision flips at the tuned operating threshold.
            try:
                import onnxruntime as _ort

                _X = np.asarray(bundle.X_test, dtype=np.float32)
                _native = np.asarray(
                    predictor_result.model.predict_proba(_X)[:, 1], dtype=float
                ).reshape(-1)
                _sess = _ort.InferenceSession(
                    str(dirs["models"] / "xcreditscore_model.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                _in = _sess.get_inputs()[0].name
                _raw = np.asarray(_sess.run(None, {_in: _X})[0])
                _onnx = (
                    _raw[:, 1].reshape(-1)
                    if _raw.ndim == 2 and _raw.shape[1] >= 2
                    else _raw.reshape(-1)
                )
                _n = int(min(len(_native), len(_onnx)))
                onnx_parity["decision_flips_at_tau"] = int(
                    (
                        (_native[:_n] >= decision_threshold)
                        != (_onnx[:_n] >= decision_threshold)
                    ).sum()
                )
                onnx_parity["tau"] = float(decision_threshold)
            except Exception as _exc:
                onnx_parity["decision_flips_error"] = type(_exc).__name__
        else:
            onnx_parity = {"ok": False, "status": "skipped_export_failed"}

    with open(dirs["metrics"] / "onnx_export_status.json", "w", encoding="utf-8") as f:
        json.dump(onnx_status, f, indent=2)

    with open(dirs["metrics"] / "onnx_parity_smoke.json", "w", encoding="utf-8") as f:
        json.dump(onnx_parity, f, indent=2)

    failure_transparency["onnx"] = {
        "export_status": onnx_status,
        "parity_status": onnx_parity,
    }
    with open(dirs["metrics"] / "failure_transparency.json", "w", encoding="utf-8") as f:
        json.dump(failure_transparency, f, indent=2)

    print("Build complete. Artifacts written to:", dirs["root"])


if __name__ == "__main__":
    main()
