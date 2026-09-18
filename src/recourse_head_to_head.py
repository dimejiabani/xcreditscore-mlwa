"""Head-to-head recourse: the existing MIP against XCreditScore's lattice vs
monotone-constrained XGBoost, both fed the identical constraint registry,
identical denied test cases, identical decision threshold τ, and identical
per-plan continuous-model validity check.

The question this answers is whether the recourse feasibility
number (63.8 % on HELOC) is a *lattice-specific* property or a
*shared-registry-specific* property — i.e., would a monotone-XGB predictor
attached to the same recourse engine produce comparable, better, or worse
feasibility on the same denied applicants?

This module answers that question directly. It reuses ``batch_counterfactuals``
from ``src.counterfactual`` unchanged: when called with ``lattice_groups=None``
the engine falls through to the linear-surrogate MIP path (Gurobi first, PuLP
+ CBC on failure), so passing ``monotone_xgb_model`` in place of the lattice
predictor yields "same MIP recourse machinery, different predictor, identical
registry".

Nothing in the existing 80-case or 1023-case recourse artifacts is touched;
this module writes strictly new files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .counterfactual import CounterfactualResult, batch_counterfactuals
from .recourse_full_evaluation import (
    _score_band,
    solver_family,
    validate_recourse_plan,
    wilson_score_ci,
)


def _run_one_model(
    *,
    model: Any,
    label: str,
    X_test: np.ndarray,
    denied_indices: np.ndarray,
    feature_names: list[str],
    decision_threshold: float,
    lattice_groups: list[list[str]] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run batch_counterfactuals against one model and validate every plan."""
    t0 = time.time()
    results: list[CounterfactualResult] = batch_counterfactuals(
        model,
        X_test,
        denied_indices,
        list(feature_names),
        lattice_groups=lattice_groups,
        max_cases=int(denied_indices.shape[0]),
        decision_threshold=float(decision_threshold),
    )
    elapsed = time.time() - t0

    rows: list[dict[str, Any]] = []
    for r in results:
        x0 = np.asarray(X_test[int(r.index)], dtype=float)
        validity = validate_recourse_plan(
            model, x0, r.changed_features, list(feature_names), decision_threshold
        )
        rows.append(
            {
                "predictor_label": label,
                "test_index": int(r.index),
                "feasible": bool(r.feasible),
                "recourse_band": r.recourse_band,
                "original_score_band": _score_band(r.original_score, decision_threshold),
                "solver_status": r.solver_status,
                "solver_family": solver_family(r.solver_status),
                "original_score": float(r.original_score),
                "new_score": float(r.new_score),
                "total_cost": float(r.total_cost),
                "n_changed_features": int(len(r.changed_features)),
                "changed_features": json.dumps(r.changed_features),
                "validity_reconstructed_score": validity["reconstructed_score"],
                "validity_passes_threshold": validity["passes_threshold"],
            }
        )
    df = pd.DataFrame(rows)

    feas_n = int(df["feasible"].sum())
    valid_feas_n = int((df["feasible"] & df["validity_passes_threshold"]).sum())
    ci = wilson_score_ci(feas_n, int(len(df)))
    summary = {
        "predictor_label": label,
        "n_cases_evaluated": int(len(df)),
        "wall_seconds": float(elapsed),
        "overall_feasibility": ci,
        "validity_pass_rate_among_feasible": (
            float(valid_feas_n / feas_n) if feas_n else 0.0
        ),
        "validity_pass_count_among_feasible": valid_feas_n,
        "feasible_count": feas_n,
        "median_total_cost_all_cases": float(pd.to_numeric(df["total_cost"], errors="coerce").median()),
        "median_total_cost_feasible_cases": (
            float(pd.to_numeric(df.loc[df["feasible"], "total_cost"], errors="coerce").median())
            if feas_n else 0.0
        ),
        "median_changed_features_feasible": (
            float(df.loc[df["feasible"], "n_changed_features"].median())
            if feas_n else 0.0
        ),
        "solver_breakdown": (
            df.groupby("solver_family")
              .size()
              .to_frame("count")
              .reset_index()
              .to_dict(orient="records")
        ),
    }
    return df, summary


def _band_stats(df: pd.DataFrame, band_col: str) -> pd.DataFrame:
    """Per-band Wilson-CI feasibility, including an OVERALL row and n<30 flag."""
    out_rows: list[dict[str, Any]] = []
    overall = df.copy()
    overall[band_col] = "OVERALL"
    stacked = pd.concat([overall, df], ignore_index=True)
    for band, group in stacked.groupby(band_col, sort=False):
        n = int(len(group))
        feasible = int(group["feasible"].sum())
        ci = wilson_score_ci(feasible, n)
        out_rows.append(
            {
                band_col: band,
                "n_cases": n,
                "feasible": feasible,
                "feasible_rate": ci["rate"],
                "feasible_rate_wilson_ci95_low": ci["ci95_low"],
                "feasible_rate_wilson_ci95_high": ci["ci95_high"],
                "small_n_flag": bool(n < 30),
                "reliability_note": (
                    "insufficient sample size for reliable inference (n<30)"
                    if n < 30
                    else ""
                ),
            }
        )
    return pd.DataFrame(out_rows)


def run_head_to_head(
    *,
    lattice_model: Any,
    lattice_groups: list[list[str]] | None,
    monotone_xgb_model: Any,
    X_test: np.ndarray,
    denied_indices: np.ndarray,
    feature_names: list[str],
    decision_threshold: float,
    output_dir: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the MIP recourse engine against both predictors on identical inputs.

    Returns a summary dict. Writes:
      - counterfactual_head_to_head.csv          (per-case rows for both models)
      - counterfactual_head_to_head_by_band.csv  (per-band Wilson CIs, both)
      - counterfactual_head_to_head_summary.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    denied = np.asarray(denied_indices, dtype=int).ravel()
    if verbose:
        print(f"[recourse-h2h] evaluating {denied.shape[0]} denied cases against 2 predictors")

    lattice_df, lattice_summary = _run_one_model(
        model=lattice_model,
        label="xcreditscore_lattice",
        X_test=X_test,
        denied_indices=denied,
        feature_names=feature_names,
        decision_threshold=decision_threshold,
        lattice_groups=lattice_groups,
    )
    if verbose:
        lci = lattice_summary["overall_feasibility"]
        print(
            f"[recourse-h2h] lattice:          "
            f"feasibility={lci['rate']:.4f} [Wilson95: {lci['ci95_low']:.4f}, {lci['ci95_high']:.4f}]  "
            f"validity={lattice_summary['validity_pass_rate_among_feasible']:.4f}  "
            f"wall={lattice_summary['wall_seconds']:.1f}s"
        )

    if monotone_xgb_model is None:
        raise ValueError("head-to-head requires a fitted monotone_xgb_model")

    xgb_df, xgb_summary = _run_one_model(
        model=monotone_xgb_model,
        label="monotone_xgboost",
        X_test=X_test,
        denied_indices=denied,
        feature_names=feature_names,
        decision_threshold=decision_threshold,
        # Lattice groups are lattice-specific; the linear-surrogate path is
        # what any non-lattice model should take.
        lattice_groups=None,
    )
    if verbose:
        xci = xgb_summary["overall_feasibility"]
        print(
            f"[recourse-h2h] monotone_xgboost: "
            f"feasibility={xci['rate']:.4f} [Wilson95: {xci['ci95_low']:.4f}, {xci['ci95_high']:.4f}]  "
            f"validity={xgb_summary['validity_pass_rate_among_feasible']:.4f}  "
            f"wall={xgb_summary['wall_seconds']:.1f}s"
        )

    combined_df = pd.concat([lattice_df, xgb_df], ignore_index=True)
    combined_df.to_csv(
        output_dir / "counterfactual_head_to_head.csv", index=False
    )

    # Per-band Wilson CI table with predictor_label as the first column.
    band_lattice = _band_stats(lattice_df, "original_score_band").assign(
        predictor_label="xcreditscore_lattice"
    )
    band_xgb = _band_stats(xgb_df, "original_score_band").assign(
        predictor_label="monotone_xgboost"
    )
    band = pd.concat([band_lattice, band_xgb], ignore_index=True)[
        [
            "predictor_label",
            "original_score_band",
            "n_cases",
            "feasible",
            "feasible_rate",
            "feasible_rate_wilson_ci95_low",
            "feasible_rate_wilson_ci95_high",
            "small_n_flag",
            "reliability_note",
        ]
    ]
    band.to_csv(
        output_dir / "counterfactual_head_to_head_by_band.csv", index=False
    )

    # Same-case comparison — for each denied applicant, do the two predictors
    # agree on feasibility? Where they disagree, which one is easier to
    # recourse against?
    merged = lattice_df.merge(
        xgb_df, on="test_index", suffixes=("_lat", "_xgb")
    )
    agreement_matrix = pd.crosstab(
        merged["feasible_lat"], merged["feasible_xgb"],
        margins=True, margins_name="TOTAL"
    ).to_dict()

    lattice_only_feasible = int(((merged["feasible_lat"]) & (~merged["feasible_xgb"])).sum())
    xgb_only_feasible = int(((~merged["feasible_lat"]) & (merged["feasible_xgb"])).sum())
    both_feasible = int(((merged["feasible_lat"]) & (merged["feasible_xgb"])).sum())
    neither_feasible = int(((~merged["feasible_lat"]) & (~merged["feasible_xgb"])).sum())

    summary = {
        "n_denied": int(denied.shape[0]),
        "decision_threshold": float(decision_threshold),
        "protocol": (
            "Same denied cases, same constraint registry, same τ, same MIP "
            "engine (Gurobi surrogate + CBC fallback); only the fitted "
            "predictor differs. Post-solve validity is checked against each "
            "predictor's own predict_proba."
        ),
        "lattice": lattice_summary,
        "monotone_xgboost": xgb_summary,
        "same_case_agreement": {
            "both_feasible": both_feasible,
            "neither_feasible": neither_feasible,
            "lattice_only_feasible": lattice_only_feasible,
            "monotone_xgb_only_feasible": xgb_only_feasible,
            "agreement_matrix_feasible_lattice_x_feasible_xgb": agreement_matrix,
        },
    }
    with open(
        output_dir / "counterfactual_head_to_head_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(
            "[recourse-h2h] same-case agreement: "
            f"both_feasible={both_feasible} neither={neither_feasible} "
            f"lattice_only={lattice_only_feasible} xgb_only={xgb_only_feasible}"
        )
    return summary
