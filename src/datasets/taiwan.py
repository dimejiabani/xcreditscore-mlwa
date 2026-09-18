"""Taiwan Default of Credit Card Clients dataset (UCI, Yeh & Lien 2009).

30 000 rows, 23 features, binary target ``default_payment_next_month``
with class balance ~78/22. Chosen as a third benchmark
with a *moderate* linear-vs-boosted AUC gap (empirically 0.02–0.05),
triangulating between HELOC (gap ~0.002, no gap) and GMSC
(gap ~0.091, wide gap).

Feature groups and monotonicity choices follow the Yeh & Lien (2009)
domain description and standard credit-scoring practice:

- ``LIMIT_BAL``   : bank-set credit limit (larger = lower risk exposure ratio) → -1
- ``SEX``, ``EDUCATION``, ``MARRIAGE``: demographic, deliberately unconstrained (0)
  to avoid fair-lending direction claims that this feature set cannot support.
- ``AGE``         : ambiguous in this dataset (middle-aged shows highest default in
  Yeh & Lien 2009); left unconstrained (0).
- ``PAY_0..PAY_6``: repayment-delay status by month. Larger integer =
  more months delayed = more risk → +1 for each. -2 / -1 / 0 encode
  "no consumption" / "pay duly" / "revolving credit"; they are left in place
  (not treated as sentinel) because they are meaningful signal per Yeh & Lien.
- ``BILL_AMT1..6``: monthly bill amounts; direction is ambiguous once
  utilisation is not directly available (larger bill can mean more spending
  OR just higher activity), so left unconstrained (0).
- ``PAY_AMT1..6`` : monthly payment amounts (larger = better repayment
  behaviour) → -1 for each.

Sentinel values (treated as NaN and median-imputed):
- ``EDUCATION`` codes 0, 5, 6 = "unknown / other" per UCI documentation
- ``MARRIAGE`` code 0 = "unknown" per UCI documentation
"""

from __future__ import annotations

import pandas as pd

NAME = "taiwan"
ARTIFACTS_SUBDIR = "taiwan_pack"
DATA_FILENAME = "data/taiwan/taiwan_default.csv"

TARGET_COL = "default_payment_next_month"
# Target is already 0/1 in the CSV.
TARGET_MAP: dict[int, int] = {0: 0, 1: 1}

FEATURE_COLS: list[str] = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

# EDUCATION 5/6 and MARRIAGE 0 are "unknown" per Yeh & Lien 2009; EDUCATION 0
# also appears in the data as an unofficial "unknown". These are the only
# sentinels; PAY_* negative codes are meaningful signal and are NOT sentinel.
SENTINEL_VALUES: list[int] = []  # column-specific sentinels handled below


MONOTONIC_CONSTRAINTS: dict[str, int] = {
    "LIMIT_BAL": -1,
    "SEX": 0,
    "EDUCATION": 0,
    "MARRIAGE": 0,
    "AGE": 0,
    "PAY_0": 1,
    "PAY_2": 1,
    "PAY_3": 1,
    "PAY_4": 1,
    "PAY_5": 1,
    "PAY_6": 1,
    "BILL_AMT1": 0,
    "BILL_AMT2": 0,
    "BILL_AMT3": 0,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": -1,
    "PAY_AMT2": -1,
    "PAY_AMT3": -1,
    "PAY_AMT4": -1,
    "PAY_AMT5": -1,
    "PAY_AMT6": -1,
}

MONOTONIC_JUSTIFICATIONS: dict[str, str] = {
    "LIMIT_BAL": (
        "-1: bank-set credit limit reflects the applicant's assessed capacity; "
        "higher limit correlates with lower default probability at origination."
    ),
    "SEX": (
        "0: demographic proxy — deliberately unconstrained to avoid a "
        "fair-lending direction claim this dataset cannot substantiate."
    ),
    "EDUCATION": (
        "0: demographic proxy — same rationale as SEX."
    ),
    "MARRIAGE": (
        "0: demographic proxy — same rationale as SEX."
    ),
    "AGE": (
        "0: Yeh & Lien (2009) show middle-aged applicants have the highest "
        "default rate in this dataset; a monotone direction is not defensible."
    ),
    "PAY_0": (
        "+1: repayment delay status for the most recent month; larger integer "
        "codes = more months delayed = strictly higher risk."
    ),
    "PAY_2": "+1: repayment delay status two months back; same direction as PAY_0.",
    "PAY_3": "+1: repayment delay status three months back; same direction as PAY_0.",
    "PAY_4": "+1: repayment delay status four months back; same direction as PAY_0.",
    "PAY_5": "+1: repayment delay status five months back; same direction as PAY_0.",
    "PAY_6": "+1: repayment delay status six months back; same direction as PAY_0.",
    "BILL_AMT1": (
        "0: bill amount alone is directionally ambiguous without a utilisation "
        "ratio (higher bill can indicate more spending activity OR higher risk)."
    ),
    "BILL_AMT2": "0: same rationale as BILL_AMT1.",
    "BILL_AMT3": "0: same rationale as BILL_AMT1.",
    "BILL_AMT4": "0: same rationale as BILL_AMT1.",
    "BILL_AMT5": "0: same rationale as BILL_AMT1.",
    "BILL_AMT6": "0: same rationale as BILL_AMT1.",
    "PAY_AMT1": (
        "-1: actual payment amount for the most recent month; larger payment "
        "= better repayment behaviour = strictly lower risk."
    ),
    "PAY_AMT2": "-1: payment amount two months back; same direction as PAY_AMT1.",
    "PAY_AMT3": "-1: payment amount three months back; same direction as PAY_AMT1.",
    "PAY_AMT4": "-1: payment amount four months back; same direction as PAY_AMT1.",
    "PAY_AMT5": "-1: payment amount five months back; same direction as PAY_AMT1.",
    "PAY_AMT6": "-1: payment amount six months back; same direction as PAY_AMT1.",
}


# Demographic features and past-history repayment codes cannot be
# changed on demand by the applicant; only forward-looking payment
# behaviour and (with the bank's consent) the credit limit are movable.
IMMUTABLE_FEATURES: set[str] = {
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
}


# Domain-informed 2-D interaction pairs for the lattice ensemble.
LATTICE_FIXED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("LIMIT_BAL", "AGE"),
    ("PAY_0", "PAY_2"),
    ("PAY_0", "BILL_AMT1"),
    ("BILL_AMT1", "PAY_AMT1"),
    ("BILL_AMT2", "PAY_AMT2"),
    ("BILL_AMT3", "PAY_AMT3"),
    ("EDUCATION", "MARRIAGE"),
)


# Recourse cost weights — higher = harder to move.
COUNTERFACTUAL_BASE_COST_WEIGHTS: dict[str, float] = {
    "LIMIT_BAL": 5.0,
    "BILL_AMT1": 1.5,
    "BILL_AMT2": 1.8,
    "BILL_AMT3": 2.0,
    "PAY_AMT1": 3.0,
    "PAY_AMT2": 3.5,
    "PAY_AMT3": 4.0,
}


COUNTERFACTUAL_MAX_STEP_FRACTIONS: dict[str, float] = {
    "LIMIT_BAL": 0.15,
    "BILL_AMT1": 0.30,
    "BILL_AMT2": 0.25,
    "BILL_AMT3": 0.25,
    "PAY_AMT1": 0.25,
    "PAY_AMT2": 0.25,
    "PAY_AMT3": 0.20,
}


_EDUCATION_SENTINEL = {0, 5, 6}
_MARRIAGE_SENTINEL = {0}


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """No engineered features for Taiwan.

    Same rationale as GMSC: the 23 raw features carry the risk signal, and
    adding ratios would either duplicate what BILL_AMT/PAY_AMT already encode
    or introduce new features not covered by the constraint registry.
    """
    X_new = X.copy()
    # Column-specific sentinel handling — replace with NaN so the median
    # imputer in data_pipeline handles them uniformly.
    if "EDUCATION" in X_new.columns:
        X_new.loc[X_new["EDUCATION"].isin(_EDUCATION_SENTINEL), "EDUCATION"] = None
    if "MARRIAGE" in X_new.columns:
        X_new.loc[X_new["MARRIAGE"].isin(_MARRIAGE_SENTINEL), "MARRIAGE"] = None
    return X_new


def prepare_target(raw_target: pd.Series) -> pd.Series:
    """Target is already 0/1 in the deduplicated CSV; enforce int typing."""
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
