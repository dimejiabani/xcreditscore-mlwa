from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import (
    DATA_PATH,
    FEATURE_ENGINEERING_ENABLED,
    FEATURE_COLS,
    RANDOM_STATE,
    SENTINEL_VALUES,
    TARGET_COL,
    add_engineered_features,
    prepare_target,
)

# Backward-compat alias for the pre-refactor name (still referenced by
# tests/test_data_pipeline.py). Delegates to the active dataset module.
_add_engineered_features = add_engineered_features


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required = [TARGET_COL, *FEATURE_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[required].copy()


def preprocess_and_split(df: pd.DataFrame) -> Tuple[DataBundle, dict]:
    cleaned = df.copy()
    cleaned[TARGET_COL] = prepare_target(cleaned[TARGET_COL])

    X = cleaned[FEATURE_COLS].replace(SENTINEL_VALUES, np.nan)
    if FEATURE_ENGINEERING_ENABLED:
        X = add_engineered_features(X)
    y = cleaned[TARGET_COL].to_numpy(dtype=int)

    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_valid_df, X_test_df, y_valid, y_test = train_test_split(
        X_temp_df,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_df)
    X_valid_imp = imputer.transform(X_valid_df)
    X_test_imp = imputer.transform(X_test_df)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_valid = scaler.transform(X_valid_imp)
    X_test = scaler.transform(X_test_imp)

    bundle = DataBundle(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        feature_names=list(X.columns),
    )
    transformers = {"imputer": imputer, "scaler": scaler}
    return bundle, transformers


def save_imputer(imputer: SimpleImputer, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, output_path)


def save_scaler(scaler: MinMaxScaler, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
