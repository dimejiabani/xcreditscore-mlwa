from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from .config import (
    PACK_DIR,
    FEATURE_COLS,
    RANDOM_STATE,
    SENTINEL_VALUES,
    TARGET_COL,
    THRESHOLD,
    prepare_target,
)
from .data_pipeline import load_raw_data


def _prepare_xy(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cleaned = raw.copy()
    cleaned[TARGET_COL] = prepare_target(cleaned[TARGET_COL])
    X = cleaned[FEATURE_COLS].replace(SENTINEL_VALUES, np.nan)
    y = cleaned[TARGET_COL].to_numpy(dtype=int)
    return X, y


def _score(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _evaluate_model_cv(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    threshold: float,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train_df = X.iloc[train_idx]
        X_test_df = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_df)
        X_test_imp = imputer.transform(X_test_df)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_imp)
        X_test = scaler.transform(X_test_imp)

        if model_name == "logistic_regression":
            model = LogisticRegression(
                C=float(params["C"]),
                max_iter=1500,
                class_weight=params["class_weight"],
                solver="lbfgs",
            )
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=None if params["max_depth"] == "none" else int(params["max_depth"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                class_weight=params["class_weight"],
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        elif model_name == "xgboost":
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=float(params["subsample"]),
                colsample_bytree=float(params["colsample_bytree"]),
                objective="binary:logistic",
                eval_metric="auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        row = {"fold": int(fold_idx), **_score(y_test, y_proba, threshold=threshold)}
        fold_scores.append(row)

    df = pd.DataFrame(fold_scores)
    return {
        "auc_mean": float(df["auc"].mean()),
        "pr_auc_mean": float(df["pr_auc"].mean()),
        "f1_mean": float(df["f1"].mean()),
        "auc_std": float(df["auc"].std(ddof=1) if len(df) > 1 else 0.0),
        "pr_auc_std": float(df["pr_auc"].std(ddof=1) if len(df) > 1 else 0.0),
        "f1_std": float(df["f1"].std(ddof=1) if len(df) > 1 else 0.0),
    }


def run_tuning() -> None:
    raw = load_raw_data()
    X, y = _prepare_xy(raw)

    n_splits = int(os.getenv("BASELINE_TUNE_CV_SPLITS", "5"))
    max_trials = int(os.getenv("BASELINE_TUNE_MAX_TRIALS", "0"))
    decision_threshold = float(os.getenv("BASELINE_TUNE_THRESHOLD", str(THRESHOLD)))
    objective = os.getenv("BASELINE_TUNE_OBJECTIVE", "auc_mean").strip().lower()
    if objective not in {"auc_mean", "pr_auc_mean", "f1_mean"}:
        objective = "auc_mean"

    search_spaces: dict[str, dict[str, list]] = {
        "logistic_regression": {
            "C": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
            "class_weight": [None, "balanced"],
        },
        "random_forest": {
            "n_estimators": [220, 320, 420],
            "max_depth": ["none", 8, 12],
            "min_samples_leaf": [1, 5, 15],
            "class_weight": [None, "balanced"],
        },
    }

    try:
        import xgboost  # noqa: F401

        search_spaces["xgboost"] = {
            "n_estimators": [220, 320, 420],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.03, 0.04, 0.06],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
        }
    except Exception:
        pass

    all_rows: list[dict] = []
    best_by_model: dict[str, dict] = {}
    rng = np.random.default_rng(RANDOM_STATE)

    parity_total_trials_env = os.getenv("PARITY_TOTAL_TRIALS", "").strip()
    parity_total_trials = max(0, int(parity_total_trials_env)) if parity_total_trials_env else 0
    model_count = max(len(search_spaces), 1)
    parity_trials_per_model = max(1, int(np.ceil(parity_total_trials / model_count))) if parity_total_trials > 0 else 0

    for model_name, space in search_spaces.items():
        keys = list(space.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*[space[k] for k in keys])]
        rng.shuffle(combos)
        if max_trials > 0:
            combos = combos[:max_trials]
        if parity_trials_per_model > 0:
            combos = combos[:parity_trials_per_model]

        best_row = None
        for trial_idx, params in enumerate(combos, start=1):
            cv = _evaluate_model_cv(
                model_name,
                params,
                X,
                y,
                n_splits=n_splits,
                threshold=decision_threshold,
            )
            row = {
                "model": model_name,
                "trial": int(trial_idx),
                **params,
                **cv,
            }
            all_rows.append(row)
            print(
                "model",
                model_name,
                "trial",
                trial_idx,
                "of",
                len(combos),
                objective,
                round(float(row[objective]), 6),
            )
            if best_row is None or row[objective] > best_row[objective]:
                best_row = row

        if best_row is not None:
            best_by_model[model_name] = best_row

    metrics_dir = PACK_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(metrics_dir / "baseline_tuning_results.csv", index=False)

    with open(metrics_dir / "baseline_tuning_best.json", "w", encoding="utf-8") as f:
        json.dump(best_by_model, f, indent=2, default=str)

    with open(metrics_dir / "baseline_tuning_run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_splits": int(n_splits),
                "objective": objective,
                "max_trials_per_model": int(max_trials),
                "decision_threshold": float(decision_threshold),
                "parity_total_trials": int(parity_total_trials),
                "parity_trials_per_model": int(parity_trials_per_model),
                "models": sorted(list(search_spaces.keys())),
            },
            f,
            indent=2,
        )

    print("wrote", metrics_dir / "baseline_tuning_results.csv")
    for model_name, row in best_by_model.items():
        print("best", model_name, objective, round(float(row[objective]), 6))


if __name__ == "__main__":
    run_tuning()
