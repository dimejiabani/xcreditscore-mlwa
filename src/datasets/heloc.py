"""HELOC dataset configuration (FICO Explainable ML Challenge, 2018).

All constants and hooks previously in ``src.config``. Kept verbatim so that
``python run_project.py --dataset heloc`` reproduces the pre-refactor
pipeline byte-for-byte on the deterministic artifact set.
"""

from __future__ import annotations

import pandas as pd

NAME = "heloc"
ARTIFACTS_SUBDIR = "heloc_pack"
DATA_FILENAME = "heloc_dataset_v1 (1).csv"

TARGET_COL = "RiskPerformance"
TARGET_MAP: dict[str, int] = {"Bad": 1, "Good": 0}

FEATURE_COLS: list[str] = [
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
]

SENTINEL_VALUES: list[int] = [-9, -8, -7]

MONOTONIC_CONSTRAINTS: dict[str, int] = {
    "ExternalRiskEstimate": -1,
    "MSinceOldestTradeOpen": -1,
    "MSinceMostRecentTradeOpen": 0,
    "AverageMInFile": -1,
    "NumSatisfactoryTrades": -1,
    "NumTrades60Ever2DerogPubRec": 1,
    "NumTrades90Ever2DerogPubRec": 1,
    "PercentTradesNeverDelq": -1,
    "MSinceMostRecentDelq": -1,
    "MaxDelq2PublicRecLast12M": 1,
    "MaxDelqEver": 1,
    "NumTotalTrades": 0,
    "NumTradesOpeninLast12M": 1,
    "PercentInstallTrades": 0,
    "MSinceMostRecentInqexcl7days": -1,
    "NumInqLast6M": 1,
    "NumInqLast6Mexcl7days": 1,
    "NetFractionRevolvingBurden": 1,
    "NetFractionInstallBurden": 1,
    "NumRevolvingTradesWBalance": 1,
    "NumInstallTradesWBalance": 1,
    "NumBank2NatlTradesWHighUtilization": 1,
    "PercentTradesWBalance": 1,
    "TotalDelqEvents": 1,
    "InquiryToTradeRatio": 1,
    "RevolvingBurdenPerTrade": 1,
    "InstallBurdenPerTrade": 1,
}

# One-line justifications so the constraint registry is a documented artifact
# under both datasets. Optional for HELOC (already validated in the paper) but
# emitted alongside GMSC's for consistency.
MONOTONIC_JUSTIFICATIONS: dict[str, str] = {
    "ExternalRiskEstimate": "FICO-family risk score: higher = less risky (protective).",
    "MSinceOldestTradeOpen": "Longer credit history reduces default risk.",
    "MSinceMostRecentTradeOpen": "Recency of newest account is directionally ambiguous.",
    "AverageMInFile": "Longer average tenure indicates stability (protective).",
    "NumSatisfactoryTrades": "More satisfactory accounts = stronger repayment record.",
    "NumTrades60Ever2DerogPubRec": "Any 60d+ delinquency count is directly risk-increasing.",
    "NumTrades90Ever2DerogPubRec": "Severe (90d+) delinquency directly increases risk.",
    "PercentTradesNeverDelq": "Share of clean trades is protective.",
    "MSinceMostRecentDelq": "Longer since last delinquency is protective.",
    "MaxDelq2PublicRecLast12M": "Worse recent delinquency severity increases risk.",
    "MaxDelqEver": "Worst-ever delinquency severity is a lasting risk marker.",
    "NumTotalTrades": "Total accounts is directionally ambiguous.",
    "NumTradesOpeninLast12M": "Rapid credit-seeking is risk-increasing.",
    "PercentInstallTrades": "Installment mix effect is directionally ambiguous.",
    "MSinceMostRecentInqexcl7days": "Longer since last inquiry is protective.",
    "NumInqLast6M": "Recent inquiry volume is risk-increasing.",
    "NumInqLast6Mexcl7days": "Recent inquiry volume (dedup) is risk-increasing.",
    "NetFractionRevolvingBurden": "Higher revolving utilization increases risk.",
    "NetFractionInstallBurden": "Higher installment utilization increases risk.",
    "NumRevolvingTradesWBalance": "More revolving accounts carrying balance = higher risk.",
    "NumInstallTradesWBalance": "More installment accounts carrying balance = higher risk.",
    "NumBank2NatlTradesWHighUtilization": "Prime-bank accounts near-max utilized = higher risk.",
    "PercentTradesWBalance": "Higher share of accounts carrying balance = higher risk.",
    "TotalDelqEvents": "Aggregate delinquency events increase risk (engineered).",
    "InquiryToTradeRatio": "Inquiry intensity relative to trades increases risk (engineered).",
    "RevolvingBurdenPerTrade": "Per-trade revolving burden increases risk (engineered).",
    "InstallBurdenPerTrade": "Per-trade installment burden increases risk (engineered).",
}

IMMUTABLE_FEATURES: set[str] = {
    "MSinceOldestTradeOpen",
    "AverageMInFile",
    "MaxDelqEver",
    "MaxDelq2PublicRecLast12M",
}

LATTICE_FIXED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ExternalRiskEstimate", "AverageMInFile"),
    ("MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen"),
    ("NumTrades60Ever2DerogPubRec", "NumTrades90Ever2DerogPubRec"),
    ("MaxDelq2PublicRecLast12M", "MaxDelqEver"),
    ("NumInqLast6M", "NumInqLast6Mexcl7days"),
    ("NetFractionRevolvingBurden", "PercentTradesWBalance"),
    ("NumRevolvingTradesWBalance", "NumInstallTradesWBalance"),
    ("NumSatisfactoryTrades", "NumTotalTrades"),
)

COUNTERFACTUAL_BASE_COST_WEIGHTS: dict[str, float] = {
    "ExternalRiskEstimate": 4.0,
    "MSinceOldestTradeOpen": 5.0,
    "AverageMInFile": 4.0,
    "NetFractionRevolvingBurden": 1.5,
    "NetFractionInstallBurden": 1.5,
    "NumTrades60Ever2DerogPubRec": 3.0,
    "NumTrades90Ever2DerogPubRec": 3.5,
    "NumInqLast6M": 2.0,
    "NumInqLast6Mexcl7days": 2.0,
}

COUNTERFACTUAL_MAX_STEP_FRACTIONS: dict[str, float] = {
    "ExternalRiskEstimate": 0.08,
    "MSinceOldestTradeOpen": 0.02,
    "AverageMInFile": 0.04,
    "NetFractionRevolvingBurden": 0.35,
    "NetFractionInstallBurden": 0.35,
    "NumInqLast6M": 0.20,
    "NumInqLast6Mexcl7days": 0.20,
    "PercentTradesWBalance": 0.30,
}


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """HELOC feature engineering — moved verbatim from data_pipeline.py."""
    X_ext = X.copy()

    # Aggregate delinquency history for a stronger monotonic risk signal.
    X_ext["TotalDelqEvents"] = (
        X_ext["NumTrades60Ever2DerogPubRec"] + X_ext["NumTrades90Ever2DerogPubRec"]
    )

    # Ratios capture intensity effects while keeping directionality interpretable.
    total_trades_denom = X_ext["NumTotalTrades"].clip(lower=0.0) + 1.0
    revolving_trades_denom = X_ext["NumRevolvingTradesWBalance"].clip(lower=0.0) + 1.0
    install_trades_denom = X_ext["NumInstallTradesWBalance"].clip(lower=0.0) + 1.0

    X_ext["InquiryToTradeRatio"] = X_ext["NumInqLast6M"] / total_trades_denom
    X_ext["RevolvingBurdenPerTrade"] = (
        X_ext["NetFractionRevolvingBurden"] / revolving_trades_denom
    )
    X_ext["InstallBurdenPerTrade"] = (
        X_ext["NetFractionInstallBurden"] / install_trades_denom
    )
    return X_ext


def prepare_target(raw_target: pd.Series) -> pd.Series:
    """Map raw ``RiskPerformance`` strings to {0, 1}, matching the paper."""
    mapped = raw_target.map(TARGET_MAP)
    if mapped.isna().any():
        bad = sorted(raw_target[mapped.isna()].astype(str).unique())
        raise ValueError(
            f"Unexpected {TARGET_COL} label(s) {bad}; expected {list(TARGET_MAP)}."
        )
    return mapped
