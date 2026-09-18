"""Full-population recourse evaluation with confidence intervals and a
post-solve validity check.

Evaluating recourse on a bounded 80-case subsample leaves three gaps: 80 is
not the population of denied applicants, no confidence interval accompanies
any rate, and stratified cells can report "100 %" on n = 4. This module
addresses all three. It leaves the 80-case output untouched and writes new
artifacts alongside it, so both views remain readable.

Design decisions:
- ``batch_counterfactuals`` is called with ``max_cases = len(denied_indices)``
  so every denied case is processed. The MIP formulation and time limits are
  the same as the original 80-case run (per the "DO NOT change the MIP"
  constraint).
- The selection rule for the original 80-case sample is documented via a
  metadata field: ``list(denied_indices)[:max_cases]`` — i.e., the denied
  cases with the *lowest test-set row index*, which is a semi-random
  ordering (test split is seed-42 stratified) but not a random or margin-
  based selection.
- Solver identity is parsed out of the existing ``solver_status`` string:
  ``gurobi_exact``, ``gurobi``, ``cbc`` (via PuLP), plus refined variants.
- Wilson 95 % CIs are used everywhere so ``n = 4, feasible = 4`` reports
  ``[0.51, 1.00]`` instead of an unqualified ``100 %``.
- Every returned plan is validated by applying δ to the original profile
  and re-scoring under the actual model — pass-rate should be 100 %; any
  failure is a bug in the recourse code, not this evaluator.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .counterfactual import CounterfactualResult, batch_counterfactuals


ORIGINAL_SAMPLE_SIZE = 80
_SOLVER_STATUS_PATTERN = re.compile(r"^([a-z0-9_]+):", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Wilson score CI for a binomial proportion                                    #
# --------------------------------------------------------------------------- #
def wilson_score_ci(successes: int, trials: int, confidence: float = 0.95) -> dict[str, float]:
    """95 % Wilson score confidence interval on ``successes / trials``.

    Wilson is the interval of choice for small-n
    cells' — normal-approximation CIs collapse at p = 0 / p = 1 and misbehave
    for n < 30, whereas Wilson stays valid and asymmetric at the extremes.
    """
    if trials <= 0:
        return {
            "rate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n_successes": int(successes),
            "n_trials": 0,
        }
    if confidence != 0.95:
        # Only 95 % is supported; every reported interval uses it.
        raise ValueError("wilson_score_ci: only 95% CI supported")
    z = 1.959963984540054  # two-sided 95 %
    n = float(trials)
    p = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return {
        "rate": p,
        "ci95_low": max(0.0, center - half),
        "ci95_high": min(1.0, center + half),
        "n_successes": int(successes),
        "n_trials": int(trials),
    }


# --------------------------------------------------------------------------- #
# Solver identity                                                              #
# --------------------------------------------------------------------------- #
def solver_family(solver_status: str) -> str:
    """Reduce full solver_status into ``gurobi_exact | gurobi | cbc | unknown``."""
    if not solver_status:
        return "unknown"
    m = _SOLVER_STATUS_PATTERN.match(str(solver_status))
    if not m:
        return "unknown"
    prefix = m.group(1).lower()
    if prefix.startswith("gurobi_exact"):
        return "gurobi_exact"
    if prefix.startswith("gurobi"):
        return "gurobi"
    if prefix.startswith("cbc") or prefix.startswith("pulp"):
        return "cbc"
    return "unknown"


# --------------------------------------------------------------------------- #
# Score-band stratification (mirrors the existing _score_band definition)      #
# --------------------------------------------------------------------------- #
def _score_band(original_score: float, threshold: float) -> str:
    delta = float(original_score - threshold)
    if delta < 0.05:
        return "near_threshold"
    if delta < 0.10:
        return "moderate_risk"
    if delta < 0.20:
        return "high_risk"
    return "very_high_risk"


# --------------------------------------------------------------------------- #
# Per-case validity check                                                      #
# --------------------------------------------------------------------------- #
def validate_recourse_plan(
    model: Any,
    x0: np.ndarray,
    changed_features: dict[str, float],
    feature_names: list[str],
    decision_threshold: float,
) -> dict[str, Any]:
    """Apply the returned δ to ``x0`` and confirm the score falls below τ.

    Returns the reconstructed x, the model's re-scored probability, and a
    pass boolean. This is the post-solve validity check — any ``pass = False``
    on a plan the MIP reported as ``feasible = True`` is a bug in the
    upstream solver code, not in this evaluator.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    x_new = np.asarray(x0, dtype=float).copy()
    for name, delta in changed_features.items():
        if name not in name_to_idx:
            continue
        x_new[name_to_idx[name]] += float(delta)
    x_new = np.clip(x_new, 0.0, 1.0)
    proba = np.asarray(model.predict_proba(x_new.reshape(1, -1)), dtype=float).ravel()
    p_new = float(proba[-1])
    return {
        "reconstructed_score": p_new,
        "passes_threshold": bool(p_new < float(decision_threshold)),
        "margin_below_threshold": float(decision_threshold) - p_new,
    }


# --------------------------------------------------------------------------- #
# Main entrypoint                                                              #
# --------------------------------------------------------------------------- #
def run_full_recourse_evaluation(
    *,
    model: Any,
    X_test: np.ndarray,
    denied_indices: np.ndarray,
    feature_names: list[str],
    lattice_groups: list[list[str]] | None,
    decision_threshold: float,
    output_dir: Path,
    original_sample_size: int = ORIGINAL_SAMPLE_SIZE,
    exact_max_cases: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run batch_counterfactuals on every denied test case; emit artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    denied = np.asarray(list(denied_indices), dtype=int).ravel()
    n_denied = int(denied.shape[0])
    if verbose:
        print(
            f"[recourse-full] {n_denied} denied test cases identified; "
            f"first_{original_sample_size} matches the original 80-case sample."
        )

    if n_denied == 0:
        summary = {"n_denied": 0, "note": "no_denied_cases"}
        (output_dir / "counterfactual_full_evaluation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    # ---- Run recourse on every denied case ---------------------------------
    t0 = time.time()
    kwargs: dict[str, Any] = {
        "max_cases": n_denied,
        "decision_threshold": float(decision_threshold),
    }
    if exact_max_cases is not None:
        kwargs["exact_max_cases"] = int(exact_max_cases)
    results: list[CounterfactualResult] = batch_counterfactuals(
        model,
        X_test,
        denied,
        list(feature_names),
        lattice_groups=lattice_groups,
        **kwargs,
    )
    elapsed = time.time() - t0
    if verbose:
        print(
            f"[recourse-full] batch_counterfactuals returned {len(results)} plans "
            f"in {elapsed:.1f}s."
        )

    # ---- Per-case rows: solver identity + validity check --------------------
    rows: list[dict[str, Any]] = []
    for r in results:
        x0 = np.asarray(X_test[int(r.index)], dtype=float)
        validity = validate_recourse_plan(
            model, x0, r.changed_features, list(feature_names), decision_threshold
        )
        rows.append(
            {
                "test_index": int(r.index),
                "feasible": bool(r.feasible),
                "recourse_band": r.recourse_band,
                "original_score_band": _score_band(r.original_score, decision_threshold),
                "solver_status": r.solver_status,
                "solver_family": solver_family(r.solver_status),
                "original_score": float(r.original_score),
                "new_score": float(r.new_score),
                "score_gap_to_threshold": float(r.score_gap_to_threshold),
                "total_cost": float(r.total_cost),
                "changed_features": json.dumps(r.changed_features),
                "validity_reconstructed_score": validity["reconstructed_score"],
                "validity_passes_threshold": validity["passes_threshold"],
                "validity_margin_below_threshold": validity["margin_below_threshold"],
            }
        )
    per_case_df = pd.DataFrame(rows)
    per_case_df.to_csv(output_dir / "counterfactual_full_evaluation.csv", index=False)

    # ---- Overall and per-band Wilson CIs -----------------------------------
    def _band_stats(df: pd.DataFrame, band_col: str) -> pd.DataFrame:
        out_rows: list[dict[str, Any]] = []
        # Include an "OVERALL" row so the reader sees the aggregate CI too.
        overall = df.copy()
        overall[band_col] = "OVERALL"
        stacked = pd.concat([overall, df], ignore_index=True)
        for band, group in stacked.groupby(band_col, sort=False):
            n = int(len(group))
            feasible = int(group["feasible"].sum())
            near_feasible = int(
                (group["recourse_band"].fillna("").astype(str) == "near_feasible").sum()
            )
            median_cost = float(pd.to_numeric(group["total_cost"], errors="coerce").median()) if n else float("nan")
            wilson_feas = wilson_score_ci(feasible, n)
            wilson_near = wilson_score_ci(near_feasible, n)
            small_n_flag = n < 30
            out_rows.append(
                {
                    band_col: band,
                    "n_cases": n,
                    "feasible": feasible,
                    "feasible_rate": wilson_feas["rate"],
                    "feasible_rate_wilson_ci95_low": wilson_feas["ci95_low"],
                    "feasible_rate_wilson_ci95_high": wilson_feas["ci95_high"],
                    "near_feasible": near_feasible,
                    "near_feasible_rate": wilson_near["rate"],
                    "near_feasible_rate_wilson_ci95_low": wilson_near["ci95_low"],
                    "near_feasible_rate_wilson_ci95_high": wilson_near["ci95_high"],
                    "median_total_cost": median_cost,
                    "small_n_flag": bool(small_n_flag),
                    "reliability_note": (
                        "insufficient sample size for reliable inference (n<30)"
                        if small_n_flag
                        else ""
                    ),
                }
            )
        return pd.DataFrame(out_rows)

    stratified_df = _band_stats(per_case_df, "original_score_band")
    stratified_df.to_csv(
        output_dir / "counterfactual_full_evaluation_by_score_band.csv", index=False
    )

    # ---- Restrict-to-first-N sanity metric ---------------------------------
    # The original 80-case sample was
    # ``list(denied_indices)[:80]``; restricting the full results to that
    # same slice must reproduce the original headline 53.75 % feasibility
    # (up to solver stochasticity — for a deterministic MIP, exactly).
    first_n = per_case_df.head(int(original_sample_size))
    first_n_feasible = int(first_n["feasible"].sum())
    first_n_ci = wilson_score_ci(first_n_feasible, int(len(first_n)))

    # ---- Validity roll-up ---------------------------------------------------
    validity_denom = int(per_case_df["feasible"].sum())
    validity_num = int(
        (
            per_case_df["feasible"]
            & per_case_df["validity_passes_threshold"]
        ).sum()
    )
    validity_all_denom = int(len(per_case_df))
    validity_all_num = int(per_case_df["validity_passes_threshold"].sum())

    # ---- Solver breakdown ---------------------------------------------------
    solver_counts = (
        per_case_df.groupby("solver_family").size().rename("count").reset_index()
    )
    solver_counts["share"] = solver_counts["count"] / max(int(len(per_case_df)), 1)
    solver_counts.to_csv(
        output_dir / "counterfactual_full_evaluation_solver_breakdown.csv",
        index=False,
    )

    overall_ci = wilson_score_ci(int(per_case_df["feasible"].sum()), int(len(per_case_df)))
    summary: dict[str, Any] = {
        "n_denied": int(n_denied),
        "n_cases_evaluated": int(len(per_case_df)),
        "wall_seconds": float(elapsed),
        "selection_rule_original_sample": (
            "list(denied_indices)[:max_cases] — i.e. the denied cases with the "
            "lowest test-set row index (semi-random via the seed-42 stratified "
            "train/test split, but neither randomised across denials nor "
            "stratified by score margin). This is documented as a "
            "potential selection bias in the original 80-case results."
        ),
        "overall_feasibility": overall_ci,
        "first_N_sanity_check": {
            "N": int(original_sample_size),
            "n_evaluated": int(len(first_n)),
            **first_n_ci,
        },
        "recourse_validity": {
            "definition": (
                "For every returned plan, apply δ to the original x0, "
                "re-score under the deployed model, and confirm P(bad) < τ."
            ),
            "tau": float(decision_threshold),
            "pass_rate_among_feasible": (
                float(validity_num / max(validity_denom, 1)) if validity_denom else 0.0
            ),
            "pass_count_among_feasible": validity_num,
            "feasible_count": validity_denom,
            "pass_rate_all_cases": (
                float(validity_all_num / max(validity_all_denom, 1))
                if validity_all_denom
                else 0.0
            ),
        },
        "solver_breakdown": [
            {
                "solver_family": r["solver_family"],
                "count": int(r["count"]),
                "share": float(r["share"]),
            }
            for _, r in solver_counts.iterrows()
        ],
        "stratified_by_score_band_rows": stratified_df.to_dict(orient="records"),
        "notes": [
            "Original 80-case artifacts (counterfactual_results.csv, "
            "counterfactual_success_by_score_band.csv, counterfactual_summary.json) "
            "are NOT overwritten — this evaluation writes strictly new files.",
            "Wilson 95% CI is used because normal-approximation CIs misbehave "
            "at p=0/p=1 and for n<30 bands.",
            "Score-band n<30 cells are flagged with reliability_note; report "
            "them as descriptive summaries rather than population estimates.",
        ],
    }
    with open(
        output_dir / "counterfactual_full_evaluation_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    if verbose:
        overall_p = summary["overall_feasibility"]["rate"]
        overall_lo = summary["overall_feasibility"]["ci95_low"]
        overall_hi = summary["overall_feasibility"]["ci95_high"]
        print(
            f"[recourse-full] overall feasibility: {overall_p:.3f} "
            f"[Wilson 95%: {overall_lo:.3f}, {overall_hi:.3f}]  "
            f"(N={summary['n_cases_evaluated']})"
        )
        sanity_p = summary["first_N_sanity_check"]["rate"]
        print(
            f"[recourse-full] first-{summary['first_N_sanity_check']['N']} sanity: "
            f"feasibility={sanity_p:.4f} "
            f"[Wilson 95%: {summary['first_N_sanity_check']['ci95_low']:.3f}, "
            f"{summary['first_N_sanity_check']['ci95_high']:.3f}]  "
            f"(this row should reproduce the paper's headline 53.75%)"
        )
        print(
            f"[recourse-full] validity pass rate among feasible: "
            f"{summary['recourse_validity']['pass_rate_among_feasible']:.4f} "
            f"({summary['recourse_validity']['pass_count_among_feasible']}"
            f" / {summary['recourse_validity']['feasible_count']})"
        )
        print("[recourse-full] solver breakdown:")
        for row in summary["solver_breakdown"]:
            print(f"  {row['solver_family']:<14} count={row['count']:>4}  share={row['share']:.3f}")

    return summary
