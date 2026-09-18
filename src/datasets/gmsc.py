"""Give Me Some Credit (GMSC) dataset configuration.

Kaggle 2011 competition; 150 000 rows, 11 features, binary target
``SeriousDlqin2yrs`` (default within 2 years, class balance ~93/7).

A second benchmark where linear
regression trails gradient boosting by a materially larger AUC gap than
on HELOC, giving the "matches boosting" claim empirical bite.

The three delinquency-count columns encode data-entry errors as 96/98;
those are treated as sentinel missing values so the median-imputer step
in :mod:`src.data_pipeline` (which is shared with HELOC) handles them
identically. ``age == 0`` is a well-known single-row error in the
public file; it is also mapped to NaN via the sentinel list.
"""

from __future__ import annotations

import pandas as pd

NAME = "gmsc"
ARTIFACTS_SUBDIR = "gmsc_pack"
DATA_FILENAME = "data/gmsc/gmsc.csv"

TARGET_COL = "SeriousDlqin2yrs"
# Target is already 0/1 in the file — identity map keeps prepare_target simple.
TARGET_MAP: dict[int, int] = {0: 0, 1: 1}

FEATURE_COLS: list[str] = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# 96 / 98 are documented data-entry codes in the three delinquency counts.
# 0 shows up once in `age`, which is impossible for a credit applicant.
SENTINEL_VALUES: list[int] = [96, 98, 0]

# Monotone directions. Signs follow standard credit-risk domain expectations:
# risk-increasing features get +1, protective features get -1, ambiguous 0.
MONOTONIC_CONSTRAINTS: dict[str, int] = {
    "RevolvingUtilizationOfUnsecuredLines": 1,
    "age": -1,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "DebtRatio": 1,
    "MonthlyIncome": -1,
    "NumberOfOpenCreditLinesAndLoans": 0,
    "NumberOfTimes90DaysLate": 1,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfDependents": 1,
}

# One-line justifications per feature — written to the constraint registry
# artifact so the domain rationale can be inspected directly.
MONOTONIC_JUSTIFICATIONS: dict[str, str] = {
    "RevolvingUtilizationOfUnsecuredLines": (
        "+1: high revolving utilisation is the single strongest sub-score "
        "input in FICO and directly risk-increasing."
    ),
    "age": (
        "-1: older applicants have longer, more stable credit histories and "
        "lower observed default rates in every large credit dataset."
    ),
    "NumberOfTime30-59DaysPastDueNotWorse": (
        "+1: any past-due count is a direct risk signal by definition."
    ),
    "DebtRatio": (
        "+1: higher debt-to-income reduces capacity to service further debt "
        "and is a canonical default predictor."
    ),
    "MonthlyIncome": (
        "-1: higher income raises debt-service capacity; treated as "
        "protective (parallel to HELOC's ExternalRiskEstimate direction)."
    ),
    "NumberOfOpenCreditLinesAndLoans": (
        "0: mix of protective (access) and risk-increasing (exposure) — "
        "leave the sign to the data rather than force a direction."
    ),
    "NumberOfTimes90DaysLate": (
        "+1: severe (90+ day) delinquency is the strongest default-history "
        "signal in this feature set."
    ),
    "NumberRealEstateLoansOrLines": (
        "0: mortgages are secured (reducing risk) but multiple can indicate "
        "over-extension — direction is genuinely ambiguous."
    ),
    "NumberOfTime60-89DaysPastDueNotWorse": (
        "+1: intermediate-severity delinquency, directionally identical to "
        "the 30-59 and 90+ counts."
    ),
    "NumberOfDependents": (
        "+1: more dependents raise fixed household expenditure and "
        "narrow the income buffer available for debt service."
    ),
}

# Applicant cannot lower these on demand — age is biological, past-due
# counts are historical facts recorded at the bureau.
IMMUTABLE_FEATURES: set[str] = {
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
}

# Domain-informed 2D interaction pairs for the lattice ensemble.
LATTICE_FIXED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("RevolvingUtilizationOfUnsecuredLines", "DebtRatio"),
    ("MonthlyIncome", "DebtRatio"),
    ("NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse"),
    ("NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse"),
    ("age", "MonthlyIncome"),
    ("NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"),
)

# Recourse cost weights — higher = harder to move in practice.
COUNTERFACTUAL_BASE_COST_WEIGHTS: dict[str, float] = {
    "RevolvingUtilizationOfUnsecuredLines": 1.5,
    "DebtRatio": 2.0,
    "MonthlyIncome": 5.0,
    "NumberOfOpenCreditLinesAndLoans": 3.0,
    "NumberRealEstateLoansOrLines": 4.0,
    "NumberOfDependents": 6.0,
}

COUNTERFACTUAL_MAX_STEP_FRACTIONS: dict[str, float] = {
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "DebtRatio": 0.20,
    "MonthlyIncome": 0.15,
    "NumberOfOpenCreditLinesAndLoans": 0.20,
}


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """GMSC uses raw features (no engineered columns) — return X unchanged.

    The base 10 features already carry the risk signal; adding engineered
    ratios here would either duplicate the raw ratios (DebtRatio, revolving
    utilisation) or introduce features not covered by the constraint
    registry, both of which weaken the direct GMSC-vs-HELOC comparison.
    """
    return X.copy()


def prepare_target(raw_target: pd.Series) -> pd.Series:
    """GMSC's ``SeriousDlqin2yrs`` is already 0/1; enforce integer typing."""
    numeric = pd.to_numeric(raw_target, errors="coerce")
    if numeric.isna().any():
        bad = sorted(raw_target[numeric.isna()].astype(str).unique())
        raise ValueError(
            f"Unexpected {TARGET_COL} label(s) {bad}; expected 0/1 integers."
        )
    values = set(numeric.unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(
            f"{TARGET_COL} values must be 0/1; found {sorted(values)}."
        )
    return numeric.astype(int)
