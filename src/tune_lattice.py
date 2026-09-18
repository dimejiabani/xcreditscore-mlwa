from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import (
    PACK_DIR,
    FEATURE_COLS,
    LATTICE_FIXED_GROUPS,
    RANDOM_STATE,
    SENTINEL_VALUES,
    TARGET_COL,
    prepare_target,
)
from .data_pipeline import load_raw_data
from .predictor import train_and_evaluate


def _profile_score(result: dict, profile: str) -> float:
    if profile == "risk_heavy":
        return float(
            (0.8 * result["auc_mean"])
            + (1.1 * result["pr_auc_mean"])
            + (0.7 * result["f1_mean"])
            - (0.4 * result["log_loss_mean"])
            - (0.2 * result["brier_mean"])
        )
    # balanced
    return float(
        (1.0 * result["auc_mean"])
        + (1.0 * result["pr_auc_mean"])
        + (0.5 * result["f1_mean"])
        - (0.35 * result["log_loss_mean"])
        - (0.2 * result["brier_mean"])
    )


def _evaluate_cv(
    X_df: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    n_splits: int,
    n_repeats: int,
) -> dict:
    fold_rows: list[dict] = []

    for repeat_idx in range(int(n_repeats)):
        skf = StratifiedKFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=RANDOM_STATE + repeat_idx,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_df, y), start=1):
            X_train_df = X_df.iloc[train_idx].copy()
            X_test_df = X_df.iloc[test_idx].copy()
            y_train = y[train_idx]
            y_test = y[test_idx]

            X_fit_df, X_valid_df, y_fit, y_valid = train_test_split(
                X_train_df,
                y_train,
                test_size=0.2,
                random_state=RANDOM_STATE + repeat_idx,
                stratify=y_train,
            )

            imputer = SimpleImputer(strategy="median")
            X_fit_imp = imputer.fit_transform(X_fit_df)
            X_valid_imp = imputer.transform(X_valid_df)
            X_test_imp = imputer.transform(X_test_df)

            scaler = MinMaxScaler()
            X_fit = scaler.fit_transform(X_fit_imp)
            X_valid = scaler.transform(X_valid_imp)
            X_test = scaler.transform(X_test_imp)

            run = train_and_evaluate(
                X_fit,
                y_fit,
                X_valid,
                y_valid,
                X_test,
                y_test,
                list(X_df.columns),
                epochs=int(params["epochs"]),
                batch_size=int(params["batch_size"]),
                calibration_keypoints=int(params["calibration_keypoints"]),
                lattice_size=int(params["lattice_size"]),
                learning_rate=float(params["learning_rate"]),
                early_stopping_patience=int(params["early_stopping_patience"]),
                grouping_mode=str(params["grouping_mode"]),
                max_group_dim=int(params["max_group_dim"]),
                fixed_groups=LATTICE_FIXED_GROUPS,
            )

            fold_rows.append(
                {
                    "repeat": int(repeat_idx + 1),
                    "fold": int(fold_idx),
                    "auc": float(roc_auc_score(y_test, run.y_proba)),
                    "pr_auc": float(average_precision_score(y_test, run.y_proba)),
                    "f1": float(f1_score(y_test, run.y_pred, zero_division=0)),
                    "brier": float(brier_score_loss(y_test, run.y_proba)),
                    "log_loss": float(log_loss(y_test, np.clip(run.y_proba, 1e-6, 1 - 1e-6))),
                    "model_family": run.model_family,
                    "fallback_reason": run.fallback_reason,
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    fallback_count = int((fold_df["model_family"] != "tensorflow_lattice").sum())
    result = {
        "cv_folds": int(len(fold_df)),
        "auc_mean": float(fold_df["auc"].mean()),
        "auc_std": float(fold_df["auc"].std(ddof=1) if len(fold_df) > 1 else 0.0),
        "pr_auc_mean": float(fold_df["pr_auc"].mean()),
        "pr_auc_std": float(fold_df["pr_auc"].std(ddof=1) if len(fold_df) > 1 else 0.0),
        "f1_mean": float(fold_df["f1"].mean()),
        "f1_std": float(fold_df["f1"].std(ddof=1) if len(fold_df) > 1 else 0.0),
        "brier_mean": float(fold_df["brier"].mean()),
        "log_loss_mean": float(fold_df["log_loss"].mean()),
        "fallback_count": fallback_count,
        "all_lattice": bool(fallback_count == 0),
    }
    result["balanced_score"] = _profile_score(result, "balanced")
    result["risk_heavy_score"] = _profile_score(result, "risk_heavy")
    # Backward-compatible alias.
    result["composite_score"] = float(result["balanced_score"])
    return result


def run_tuning() -> None:
    raw = load_raw_data()

    cleaned = raw.copy()
    cleaned[TARGET_COL] = prepare_target(cleaned[TARGET_COL])

    X_df = cleaned[FEATURE_COLS].replace(SENTINEL_VALUES, np.nan)
    y = cleaned[TARGET_COL].to_numpy(dtype=int)

    tune_profile = os.getenv("TUNE_PROFILE", "fast").strip().lower()
    if tune_profile not in {"fast", "final"}:
        tune_profile = "fast"

    cv_splits = int(os.getenv("TUNE_CV_SPLITS", "3" if tune_profile == "fast" else "5"))
    cv_repeats = int(os.getenv("TUNE_CV_REPEATS", "1" if tune_profile == "fast" else "2"))
    objective_profile = os.getenv("TUNE_OBJECTIVE_PROFILE", "balanced").strip().lower()
    if objective_profile not in {"balanced", "risk_heavy"}:
        objective_profile = "balanced"

    objective = os.getenv("TUNE_OBJECTIVE", "auc_mean").strip()
    if objective not in {"auc_mean", "pr_auc_mean", "f1_mean", "balanced_score", "risk_heavy_score", "composite_score"}:
        objective = "balanced_score" if objective_profile == "balanced" else "risk_heavy_score"

    if tune_profile == "fast":
        search_space = {
            "epochs": [12, 18],
            "batch_size": [128],
            "calibration_keypoints": [10, 15],
            "lattice_size": [3],
            "learning_rate": [0.003, 0.005],
            "early_stopping_patience": [3, 4],
            "grouping_mode": ["fixed", "correlation"],
            "max_group_dim": [2, 3],
        }
    else:
        search_space = {
            "epochs": [20, 28, 36],
            "batch_size": [128, 256],
            "calibration_keypoints": [10, 15, 20],
            "lattice_size": [3, 4],
            "learning_rate": [0.002, 0.003, 0.005],
            "early_stopping_patience": [4, 6],
            "grouping_mode": ["fixed", "correlation"],
            "max_group_dim": [2, 3],
        }

    keys = list(search_space.keys())
    all_values = [search_space[k] for k in keys]
    all_trials = list(itertools.product(*all_values))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(all_trials)

    max_trials_env = os.getenv("TUNE_MAX_TRIALS", "")
    if max_trials_env.strip():
        max_trials = max(1, int(max_trials_env))
        all_trials = all_trials[:max_trials]

    parity_total_trials_env = os.getenv("PARITY_TOTAL_TRIALS", "").strip()
    parity_total_trials = 0
    if parity_total_trials_env:
        parity_total_trials = max(1, int(parity_total_trials_env))
        all_trials = all_trials[:parity_total_trials]

    results = []
    for trial_no, values in enumerate(all_trials, start=1):
        params = dict(zip(keys, values))
        cv_metrics = _evaluate_cv(
            X_df=X_df,
            y=y,
            params=params,
            n_splits=cv_splits,
            n_repeats=cv_repeats,
        )

        row = {
            **params,
            "tune_profile": tune_profile,
            "cv_splits": int(cv_splits),
            "cv_repeats": int(cv_repeats),
            **cv_metrics,
        }
        results.append(row)
        print(
            "trial",
            trial_no,
            "of",
            len(all_trials),
            objective,
            round(float(row[objective]), 6),
            "auc",
            round(float(row["auc_mean"]), 6),
            "pr_auc",
            round(float(row["pr_auc_mean"]), 6),
            "f1",
            round(float(row["f1_mean"]), 6),
            "comp",
            round(float(row["composite_score"]), 6),
            "balanced",
            round(float(row["balanced_score"]), 6),
            "risk",
            round(float(row["risk_heavy_score"]), 6),
            "fallbacks",
            int(row["fallback_count"]),
        )

    results_df = pd.DataFrame(results)
    # Prefer stable lattice runs with fewer fallbacks; then optimize chosen objective.
    results_df = results_df.sort_values(
        by=["all_lattice", "fallback_count", objective, "balanced_score", "risk_heavy_score", "auc_mean", "pr_auc_mean", "f1_mean"],
        ascending=[False, True, False, False, False, False, False, False],
    )

    metrics_dir = PACK_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_csv = metrics_dir / "lattice_tuning_results.csv"
    results_df.to_csv(out_csv, index=False)

    best = results_df.iloc[0].to_dict()
    with open(metrics_dir / "lattice_tuning_best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    with open(metrics_dir / "lattice_tuning_run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tune_profile": tune_profile,
                "cv_splits": int(cv_splits),
                "cv_repeats": int(cv_repeats),
                "objective": objective,
                "objective_profile": objective_profile,
                "trial_count": int(len(all_trials)),
                "parity_total_trials": int(parity_total_trials),
            },
            f,
            indent=2,
        )

    print("wrote", str(out_csv))
    print("best_objective", objective, round(float(best[objective]), 6))
    print("best_auc", round(float(best["auc_mean"]), 6))
    print("best_pr_auc", round(float(best["pr_auc_mean"]), 6))
    print("best_f1", round(float(best["f1_mean"]), 6))
    print("best_balanced", round(float(best["balanced_score"]), 6))
    print("best_risk_heavy", round(float(best["risk_heavy_score"]), 6))


if __name__ == "__main__":
    run_tuning()
