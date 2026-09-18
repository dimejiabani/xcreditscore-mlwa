"""Statistical corrections for multiplicity and fold dependence.

Implements Holm-Bonferroni family-wise adjustment, the Nadeau-Bengio
corrected resampled t-test, paired bootstrap confidence intervals on metric
differences, DeLong's paired AUC test, and matched operating-point
comparisons.

Pure-numpy implementations — no new dependencies. Every function returns a
plain dict so the results serialise straight to JSON alongside the
paired-sign-flip / bootstrap-mean-delta outputs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# Baseline test, retained alongside the corrections rather than replaced     #
# --------------------------------------------------------------------------- #
def paired_sign_flip_pvalue(
    deltas: np.ndarray, *, n_perm: int = 20_000, seed: int = 123
) -> float:
    """Two-sided paired sign-flip permutation p-value.

    Retained as the uncorrected Table 8 test so that uncorrected and
    corrected p-values can be reported side by side rather than the
    uncorrected value being silently replaced.
    """
    d = np.asarray(deltas, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), d.shape[0]))
    perm_means = (signs * d).mean(axis=1)
    obs = abs(float(d.mean()))
    p = (float((np.abs(perm_means) >= obs).sum()) + 1.0) / (int(n_perm) + 1.0)
    return float(p)


def bootstrap_ci_mean(
    deltas: np.ndarray, *, n_boot: int = 20_000, seed: int = 42
) -> tuple[float, float]:
    """Non-parametric 95 % percentile CI on the mean of a paired-delta vector."""
    d = np.asarray(deltas, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    idx = rng.integers(0, n, size=(int(n_boot), n))
    lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Holm-Bonferroni step-down                                            #
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Return Holm-Bonferroni step-down adjusted p-values (family-wise, α-safe).

    Given m raw p-values sorted ascending as p_(1) ≤ ... ≤ p_(m), the adjusted
    value at rank i is ``min(1, max_{j≤i}((m − j + 1) * p_(j)))``. Preserves the
    input order in the returned list.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        multiplier = m - rank + 1
        candidate = float(multiplier) * float(pvalues[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


# --------------------------------------------------------------------------- #
# Nadeau-Bengio corrected resampled t-test                             #
# --------------------------------------------------------------------------- #
def nadeau_bengio_ttest(
    deltas: np.ndarray,
    n_test: int,
    n_train: int,
) -> dict[str, float]:
    """Corrected resampled t-test for repeated-k-fold CV (Nadeau & Bengio, 2003).

    ``deltas`` is the vector of per-fold-pair (model_A − model_B) score
    differences across all K × R fold-pairs. The uncorrected paired t-test
    understates variance because fold-pairs share training data; the
    Nadeau-Bengio correction inflates the variance estimator by
    ``(1/n + n_test / n_train)`` before the t-statistic is computed. That
    matches the formula the review specifies.

    Returns t-statistic, two-sided p-value, degrees of freedom, and the
    correction factor actually applied so it can be audited.
    """
    d = np.asarray(deltas, dtype=float).ravel()
    n = int(d.shape[0])
    if n < 2:
        return {
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "df": float(max(n - 1, 0)),
            "variance_multiplier": float("nan"),
            "n_fold_pairs": n,
            "n_test": int(n_test),
            "n_train": int(n_train),
        }
    mean = float(np.mean(d))
    var = float(np.var(d, ddof=1))
    if n_train <= 0:
        multiplier = 1.0 / n
    else:
        multiplier = (1.0 / n) + (float(n_test) / float(n_train))
    corrected_var = var * multiplier
    if corrected_var <= 0.0 or math.isnan(corrected_var):
        return {
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "df": float(n - 1),
            "variance_multiplier": float(multiplier),
            "n_fold_pairs": n,
            "n_test": int(n_test),
            "n_train": int(n_train),
        }
    t_stat = mean / math.sqrt(corrected_var)
    df = n - 1

    # Two-sided p-value via student's-t survival function (no scipy needed).
    p = 2.0 * _student_t_sf(abs(t_stat), df)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p),
        "df": float(df),
        "variance_multiplier": float(multiplier),
        "n_fold_pairs": n,
        "n_test": int(n_test),
        "n_train": int(n_train),
    }


def _student_t_sf(t: float, df: int) -> float:
    """Right-tail survival function for Student's t, closed form via I_x(a,b).

    ``P(T > t) = I_{df / (df + t^2)}(df/2, 1/2) / 2`` — the incomplete-beta
    identity, avoiding scipy.
    """
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    ib = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return 0.5 * ib


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b) via continued fraction (Numerical Recipes recipe)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float, itmax: int = 500, eps: float = 3e-15) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


# --------------------------------------------------------------------------- #
# Bootstrap CI on paired PR-AUC / AUC difference (test partition)      #
# --------------------------------------------------------------------------- #
def paired_bootstrap_metric_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    *,
    metric: str = "pr_auc",
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of (metric(A) − metric(B)) on the same test partition.

    Bootstraps sample-indices (with replacement) 10 000 times by default, giving
    (i) 95 % percentile CI for the difference and (ii) a two-sided bootstrap
    p-value ``2 * min(P(Δ ≥ 0), P(Δ ≤ 0))``. ``metric`` is ``"pr_auc"`` or
    ``"auc"`` — the two metrics reconciled in §4.1 and Table 8.
    """
    metric = metric.strip().lower()
    if metric not in {"pr_auc", "auc"}:
        raise ValueError(f"metric must be 'pr_auc' or 'auc'; got {metric!r}")
    y_true = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(y_score_a, dtype=float).ravel()
    b = np.asarray(y_score_b, dtype=float).ravel()
    if y_true.shape[0] != a.shape[0] or a.shape[0] != b.shape[0]:
        raise ValueError("y_true / y_score_a / y_score_b must share length")

    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    obs_a = float(fn(y_true, a))
    obs_b = float(fn(y_true, b))
    obs_diff = obs_a - obs_b

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    diffs = np.empty(n_boot, dtype=float)
    skipped = 0
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if yb.min() == yb.max():
            # Degenerate bootstrap sample (single class) — retry once.
            idx = rng.integers(0, n, size=n)
            yb = y_true[idx]
            if yb.min() == yb.max():
                diffs[i] = np.nan
                skipped += 1
                continue
        diffs[i] = float(fn(yb, a[idx])) - float(fn(yb, b[idx]))

    valid = diffs[~np.isnan(diffs)]
    if valid.size == 0:
        return {
            "metric": metric,
            "n_boot": int(n_boot),
            "skipped_bootstraps": int(skipped),
            "observed_metric_a": obs_a,
            "observed_metric_b": obs_b,
            "observed_difference": obs_diff,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "two_sided_bootstrap_pvalue": float("nan"),
        }

    ci_lo, ci_hi = np.quantile(valid, [0.025, 0.975])
    # Two-sided bootstrap p-value (Efron & Tibshirani, §16.4 shift-invariant form).
    p_ge = float(np.mean(valid >= 0.0))
    p_le = float(np.mean(valid <= 0.0))
    p_two_sided = min(1.0, 2.0 * min(p_ge, p_le))

    return {
        "metric": metric,
        "n_boot": int(n_boot),
        "skipped_bootstraps": int(skipped),
        "observed_metric_a": obs_a,
        "observed_metric_b": obs_b,
        "observed_difference": obs_diff,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "two_sided_bootstrap_pvalue": float(p_two_sided),
    }


# --------------------------------------------------------------------------- #
# DeLong exact AUC test (paired)                                       #
# --------------------------------------------------------------------------- #
def delong_paired_auc_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> dict[str, float]:
    """DeLong 1988 paired ROC-AUC test — closed-form Z, two-sided p-value.

    Implementation follows Sun & Xu (2014) fast midrank algorithm; only depends
    on numpy. Returns the two AUCs, their difference, the estimated covariance
    matrix diagonal, the DeLong Z-statistic, and the two-sided p-value.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(y_score_a, dtype=float).ravel()
    b = np.asarray(y_score_b, dtype=float).ravel()
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return {
            "auc_a": float("nan"),
            "auc_b": float("nan"),
            "difference": float("nan"),
            "z_stat": float("nan"),
            "two_sided_pvalue": float("nan"),
            "note": "single_class_in_sample",
        }

    def _fast_delong(scores):
        pos_scores = scores[pos]
        neg_scores = scores[neg]
        n_pos = pos_scores.size
        n_neg = neg_scores.size
        # Midranks of pos-only, neg-only, and combined arrays.
        tx = _midrank(pos_scores)
        ty = _midrank(neg_scores)
        tz = _midrank(np.concatenate([pos_scores, neg_scores]))
        auc = (tz[:n_pos].sum() / (n_pos * n_neg)) - ((n_pos + 1.0) / (2.0 * n_neg))
        v01 = (tz[:n_pos] - tx) / n_neg
        v10 = 1.0 - (tz[n_pos:] - ty) / n_pos
        return auc, v01, v10

    auc_a, v01_a, v10_a = _fast_delong(a)
    auc_b, v01_b, v10_b = _fast_delong(b)

    def _cov(v_a, v_b):
        return float(np.cov(v_a, v_b, ddof=1)[0, 1])

    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    s01 = np.array(
        [
            [_cov(v01_a, v01_a), _cov(v01_a, v01_b)],
            [_cov(v01_a, v01_b), _cov(v01_b, v01_b)],
        ]
    )
    s10 = np.array(
        [
            [_cov(v10_a, v10_a), _cov(v10_a, v10_b)],
            [_cov(v10_a, v10_b), _cov(v10_b, v10_b)],
        ]
    )
    cov = s01 / n_pos + s10 / n_neg
    l = np.array([1.0, -1.0])
    var_diff = float(l @ cov @ l)
    diff = float(auc_a - auc_b)
    if var_diff <= 0.0:
        # Identical predictors give diff == 0 and var == 0. That is not an
        # undefined comparison: there is no difference to detect, so the
        # correct report is z = 0, p = 1. Any other zero-variance case is
        # genuinely degenerate and stays nan so it cannot be read as a result.
        if diff == 0.0:
            return {
                "auc_a": float(auc_a),
                "auc_b": float(auc_b),
                "difference": 0.0,
                "z_stat": 0.0,
                "two_sided_pvalue": 1.0,
                "note": "identical_predictions",
            }
        return {
            "auc_a": float(auc_a),
            "auc_b": float(auc_b),
            "difference": diff,
            "z_stat": float("nan"),
            "two_sided_pvalue": float("nan"),
            "note": "zero_or_negative_variance",
        }
    z = diff / math.sqrt(var_diff)
    p = 2.0 * (1.0 - _standard_normal_cdf(abs(z)))
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "difference": diff,
        "z_stat": float(z),
        "two_sided_pvalue": float(p),
    }


def _midrank(x: np.ndarray) -> np.ndarray:
    """Ties get their average rank — the same convention scipy.stats.rankdata uses."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=float)
    n = x.size
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Matched operating-point comparisons                                  #
# --------------------------------------------------------------------------- #
def find_threshold_for_denial_rate(
    y_score: np.ndarray, target_rate: float
) -> float:
    """Return the smallest τ such that P(score ≥ τ) ≤ target_rate.

    Equivalent to the quantile at (1 − target_rate) of the score distribution.
    """
    scores = np.asarray(y_score, dtype=float).ravel()
    q = 1.0 - float(target_rate)
    q = min(1.0, max(0.0, q))
    return float(np.quantile(scores, q))


def find_threshold_for_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float
) -> tuple[float, float, float]:
    """Sweep thresholds and pick the smallest τ whose precision ≥ target.

    Returns (threshold, achieved_precision, recall_at_threshold).
    Ties broken by preferring the higher recall.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    total_pos = int((y == 1).sum())
    denom = tp + fp
    precision = np.where(denom > 0, tp / denom, 0.0)
    recall = tp / max(total_pos, 1)
    mask = precision >= float(target_precision)
    if not mask.any():
        # Fall back to the highest-precision threshold.
        i_best = int(np.argmax(precision))
    else:
        # Among candidates, pick the one with highest recall.
        candidate_idx = np.where(mask)[0]
        i_best = int(candidate_idx[int(np.argmax(recall[candidate_idx]))])
    threshold = float(s[order[i_best]])
    return threshold, float(precision[i_best]), float(recall[i_best])


def matched_operating_point_row(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    y_pred = (s >= float(threshold)).astype(int)
    tp = int(((y == 1) & (y_pred == 1)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tn = int(((y == 0) & (y_pred == 0)).sum())
    denial_rate = float((y_pred == 1).mean())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "model": model_name,
        "threshold": float(threshold),
        "denial_rate": denial_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
