from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	brier_score_loss,
	f1_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from .baselines import run_baselines
from .config import (
	FEATURE_COLS,
	LATTICE_BATCH_SIZE,
	LATTICE_CALIBRATION_KEYPOINTS,
	LATTICE_EARLY_STOPPING_PATIENCE,
	LATTICE_EPOCHS,
	LATTICE_FIXED_GROUPS,
	LATTICE_GROUPING_MODE,
	LATTICE_LEARNING_RATE,
	LATTICE_MAX_GROUP_DIM,
	LATTICE_SIZE,
	RANDOM_STATE,
	SENTINEL_VALUES,
	TARGET_COL,
	THRESHOLD,
	add_engineered_features,
	prepare_target,
)
from .predictor import train_and_evaluate


def _full_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
	y_pred = (y_proba >= threshold).astype(int)
	y_proba_safe = np.clip(y_proba, 1e-6, 1 - 1e-6)
	return {
		"auc": float(roc_auc_score(y_true, y_proba)),
		"pr_auc": float(average_precision_score(y_true, y_proba)),
		"brier": float(brier_score_loss(y_true, y_proba)),
		"log_loss": float(log_loss(y_true, y_proba_safe)),
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
	}


def _aggregate(model_df: pd.DataFrame) -> dict[str, float]:
	metric_cols = [
		"auc",
		"pr_auc",
		"brier",
		"log_loss",
		"accuracy",
		"precision",
		"recall",
		"f1",
	]
	n = max(int(len(model_df)), 1)
	out: dict[str, float] = {"folds": float(n)}
	for c in metric_cols:
		vals = model_df[c].astype(float).to_numpy()
		mean = float(np.mean(vals))
		std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
		sem = float(std / np.sqrt(n)) if n > 1 else 0.0
		ci95 = float(1.96 * sem)
		out[f"{c}_mean"] = mean
		out[f"{c}_std"] = std
		out[f"{c}_ci95"] = ci95
	return out


def run_cross_validation(raw_df: pd.DataFrame, metrics_dir: Path, n_splits: int, n_repeats: int) -> dict:
	cleaned = raw_df.copy()
	cleaned[TARGET_COL] = prepare_target(cleaned[TARGET_COL])

	X_full = cleaned[FEATURE_COLS].replace(SENTINEL_VALUES, np.nan)
	y_full = cleaned[TARGET_COL].to_numpy(dtype=int)

	rows: list[dict] = []
	threshold_rows: list[dict] = []

	for repeat_idx in range(int(n_repeats)):
		skf = StratifiedKFold(
			n_splits=int(n_splits),
			shuffle=True,
			random_state=RANDOM_STATE + repeat_idx,
		)

		for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full), start=1):
			X_train_df = X_full.iloc[train_idx].copy()
			X_test_df = X_full.iloc[test_idx].copy()
			y_train = y_full[train_idx]
			y_test = y_full[test_idx]

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

			xscore = train_and_evaluate(
				X_fit,
				y_fit,
				X_valid,
				y_valid,
				X_test,
				y_test,
				list(X_full.columns),
				epochs=LATTICE_EPOCHS,
				batch_size=LATTICE_BATCH_SIZE,
				calibration_keypoints=LATTICE_CALIBRATION_KEYPOINTS,
				lattice_size=LATTICE_SIZE,
				learning_rate=LATTICE_LEARNING_RATE,
				early_stopping_patience=LATTICE_EARLY_STOPPING_PATIENCE,
				grouping_mode=LATTICE_GROUPING_MODE,
				max_group_dim=LATTICE_MAX_GROUP_DIM,
				fixed_groups=LATTICE_FIXED_GROUPS,
				decision_threshold=float(THRESHOLD),
			)

			rows.append(
				{
					"repeat": int(repeat_idx + 1),
					"fold": int(fold_idx),
					"model": "xcreditscore",
					**_full_metrics(y_test, xscore.y_proba, float(THRESHOLD)),
				}
			)

			baselines = run_baselines(X_fit, y_fit, X_test, y_test, decision_threshold=float(THRESHOLD))
			for name, vals in baselines.items():
				if str(vals.get("status", "ok")) != "ok":
					continue
				rows.append(
					{
						"repeat": int(repeat_idx + 1),
						"fold": int(fold_idx),
						"model": name,
						**{
							k: float(vals[k])
							for k in ["auc", "pr_auc", "brier", "log_loss", "accuracy", "precision", "recall", "f1"]
							if k in vals
						},
					}
				)

			threshold_rows.append(
				{
					"repeat": int(repeat_idx + 1),
					"fold": int(fold_idx),
					"decision_threshold": float(THRESHOLD),
				}
			)

	cv_df = pd.DataFrame(rows)
	cv_df.to_csv(metrics_dir / "cv_model_metrics.csv", index=False)

	threshold_df = pd.DataFrame(threshold_rows)
	threshold_df.to_csv(metrics_dir / "cv_thresholds.csv", index=False)

	summary = {
		"n_splits": int(n_splits),
		"n_repeats": int(n_repeats),
		"models": {},
	}
	for model_name in sorted(cv_df["model"].unique()):
		model_df = cv_df[cv_df["model"] == model_name]
		summary["models"][model_name] = _aggregate(model_df)

	with open(metrics_dir / "cv_summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	return summary
