from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS, TARGET_COL
from src.data_pipeline import _add_engineered_features, load_raw_data, preprocess_and_split


def _synthetic_df(n_rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {col: rng.integers(1, 50, size=n_rows).astype(float) for col in FEATURE_COLS}
    labels = np.array(["Bad", "Good"] * (n_rows // 2), dtype=object)
    if labels.size < n_rows:
        labels = np.concatenate([labels, np.array(["Bad"], dtype=object)])
    data[TARGET_COL] = labels[:n_rows]
    return pd.DataFrame(data)


def test_add_engineered_features_columns_and_basic_formulas() -> None:
    base = pd.DataFrame(
        {
            "NumTrades60Ever2DerogPubRec": [2.0],
            "NumTrades90Ever2DerogPubRec": [1.0],
            "NumInqLast6M": [6.0],
            "NumTotalTrades": [9.0],
            "NetFractionRevolvingBurden": [30.0],
            "NumRevolvingTradesWBalance": [2.0],
            "NetFractionInstallBurden": [20.0],
            "NumInstallTradesWBalance": [4.0],
        }
    )

    out = _add_engineered_features(base)

    assert "TotalDelqEvents" in out.columns
    assert "InquiryToTradeRatio" in out.columns
    assert "RevolvingBurdenPerTrade" in out.columns
    assert "InstallBurdenPerTrade" in out.columns

    assert out.loc[0, "TotalDelqEvents"] == pytest.approx(3.0)
    assert out.loc[0, "InquiryToTradeRatio"] == pytest.approx(6.0 / 10.0)
    assert out.loc[0, "RevolvingBurdenPerTrade"] == pytest.approx(30.0 / 3.0)
    assert out.loc[0, "InstallBurdenPerTrade"] == pytest.approx(20.0 / 5.0)


def test_load_raw_data_missing_columns_raises(tmp_path) -> None:
    df = pd.DataFrame({TARGET_COL: ["Bad", "Good"], FEATURE_COLS[0]: [1.0, 2.0]})
    p = tmp_path / "broken.csv"
    df.to_csv(p, index=False)

    with pytest.raises(ValueError):
        load_raw_data(p)


def test_preprocess_and_split_shapes_and_feature_names() -> None:
    df = _synthetic_df(80)
    bundle, transformers = preprocess_and_split(df)

    assert bundle.X_train.shape[0] > 0
    assert bundle.X_valid.shape[0] > 0
    assert bundle.X_test.shape[0] > 0
    assert bundle.X_train.shape[1] == len(bundle.feature_names)
    assert len(bundle.feature_names) >= len(FEATURE_COLS)
    assert "imputer" in transformers and "scaler" in transformers
