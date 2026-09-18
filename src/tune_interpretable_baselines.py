"""40-trial parity tuning for the interpretable-by-design baselines.

Mirrors :mod:`src.tune_baselines` in structure so the same
budget accounting applies. A dedicated cache
(``interpretable_baseline_tuning_best.json``) is written so this run cannot
dilute the per-model budget allocated to the existing LR/RF/XGB baselines.
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from .baselines import build_monotone_constraints_vector
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


def _fit_and_score_ebm(params: dict, feature_names: list[str], constraints: tuple[int, ...],
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       threshold: float) -> dict[str, float]:
    from interpret.glassbox import ExplainableBoostingClassifier

    model = ExplainableBoostingClassifier(
        feature_names=list(feature_names),
        monotone_constraints=list(constraints),
        interactions=int(params["interactions"]),
        learning_rate=float(params["learning_rate"]),
        max_bins=int(params["max_bins"]),
        outer_bags=int(params["outer_bags"]),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return _score(y_test, model.predict_proba(X_test)[:, 1], threshold=threshold)


def _fit_and_score_monotone_gam(params: dict, feature_names: list[str], constraints: tuple[int, ...],
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                threshold: float) -> dict[str, float]:
    from pygam import LogisticGAM

    from .baselines import _build_pygam_terms

    terms = _build_pygam_terms(
        list(feature_names),
        constraints,
        n_splines=int(params["n_splines"]),
        spline_order=int(params["spline_order"]),
        lam=float(params["lam"]),
    )
    model = LogisticGAM(terms, max_iter=int(params.get("max_iter", 200)), tol=1e-4)
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int))
    p1 = np.asarray(model.predict_proba(np.asarray(X_test, dtype=float))).reshape(-1)
    return _score(y_test, np.clip(p1, 1e-6, 1 - 1e-6), threshold=threshold)


def _evaluate_model_cv(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    constraints: tuple[int, ...],
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

        if model_name == "ebm":
            row = _fit_and_score_ebm(
                params, feature_names, constraints,
                X_train, y_train, X_test, y_test, threshold,
            )
        elif model_name == "monotone_gam":
            row = _fit_and_score_monotone_gam(
                params, feature_names, constraints,
                X_train, y_train, X_test, y_test, threshold,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        fold_scores.append({"fold": int(fold_idx), **row})

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
    feature_names = list(FEATURE_COLS)
    constraints = build_monotone_constraints_vector(feature_names, verbose=False)

    n_splits = int(os.getenv("INTERPRETABLE_TUNE_CV_SPLITS", "5"))
    max_trials = int(os.getenv("INTERPRETABLE_TUNE_MAX_TRIALS", "0"))
    decision_threshold = float(os.getenv("INTERPRETABLE_TUNE_THRESHOLD", str(THRESHOLD)))
    objective = os.getenv("INTERPRETABLE_TUNE_OBJECTIVE", "auc_mean").strip().lower()
    if objective not in {"auc_mean", "pr_auc_mean", "f1_mean"}:
        objective = "auc_mean"

    search_spaces: dict[str, dict[str, list]] = {}
    try:
        import interpret  # noqa: F401

        search_spaces["ebm"] = {
            "interactions": [0, 5, 10, 15],
            "learning_rate": [0.01, 0.02, 0.05],
            "max_bins": [128, 256, 512],
            "outer_bags": [8, 14],
        }
    except Exception:
        pass

    try:
        import pygam  # noqa: F401

        search_spaces["monotone_gam"] = {
            "lam": [0.1, 0.6, 1.0, 5.0, 10.0, 25.0],
            "n_splines": [10, 15, 20, 25],
            "spline_order": [3],
            "max_iter": [200],
        }
    except Exception:
        pass

    all_rows: list[dict] = []
    best_by_model: dict[str, dict] = {}
    rng = np.random.default_rng(RANDOM_STATE)

    parity_total_trials_env = os.getenv("INTERPRETABLE_PARITY_TOTAL_TRIALS", "40").strip()
    parity_total_trials = max(0, int(parity_total_trials_env)) if parity_total_trials_env else 0
    model_count = max(len(search_spaces), 1)
    parity_trials_per_model = (
        max(1, int(np.ceil(parity_total_trials / model_count)))
        if parity_total_trials > 0
        else 0
    )

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
                feature_names,
                constraints,
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

    pd.DataFrame(all_rows).to_csv(
        metrics_dir / "interpretable_baseline_tuning_results.csv", index=False
    )
    with open(metrics_dir / "interpretable_baseline_tuning_best.json", "w", encoding="utf-8") as f:
        json.dump(best_by_model, f, indent=2, default=str)

    with open(metrics_dir / "interpretable_baseline_tuning_run_config.json", "w", encoding="utf-8") as f:
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

    print("wrote", metrics_dir / "interpretable_baseline_tuning_results.csv")
    for model_name, row in best_by_model.items():
        print("best", model_name, objective, round(float(row[objective]), 6))


if __name__ == "__main__":
    run_tuning()
