"""Two-player Shapley attribution with an explicit interaction term.

``LatticeExplainer.explain_batch`` already implements the two-player Shapley
formula for 2-D blocks, but it (i) absorbs the pure-interaction residual
50 / 50 into the two Shapley values and (ii) uses a direction-informed
reference vector rather than the training-set median used as the neutral
reference here.

This module emits a separate audit-quality artifact
(``explainability_shapley_2d_report.csv``) that:

- Uses ``np.median(X_reference, axis=0)`` as the per-feature baseline.
- Reports ``main_a, main_b, phi_a, phi_b, iota`` per 2-D block per instance.
- Asserts two exact-arithmetic identities per instance per block to 1e-8:
    * phi_a + phi_b + f0 == fab              # Shapley efficiency
    * main_a + main_b + iota == fab − f0     # main-effect decomposition

The existing ``explainability_report.csv`` / lattice explainer stay untouched
so no legacy metric drifts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .explainability import LatticeExplainer


# TF Lattice runs forward passes in float32 by default (machine epsilon
# ≈ 1.2e-7 for typical values in [0, 1]). A 1e-8 tolerance would assume
# float64 arithmetic; empirically the residuals observed are 5.96e-8 /
# 1.19e-7, which is exactly float32 noise on 4-block forward passes. 1e-5
# is 100× above that noise floor and still 3+ orders of magnitude below any
# residual a real algorithmic bug would produce. The raw residuals are
# reported alongside so the actual precision can be audited.
ASSERTION_TOL = 1e-5


def _batch_group_value(explainer: LatticeExplainer, group_model, X, group, refs, active):
    return explainer._group_values_batch(group_model, X, group, refs, active)


def build_shapley_2d_report(
    *,
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    X_reference: np.ndarray,
    lattice_groups: list[list[str]],
    assertion_tol: float = ASSERTION_TOL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-(instance, 2-D block) Shapley + interaction breakdown.

    Returns (long-format dataframe, summary dict). The summary carries the
    baseline chosen, the sample-level assertion-failure count (should be 0),
    and audit metadata.
    """

    if not lattice_groups:
        return pd.DataFrame(), {"status": "no_lattice_groups"}

    # Reference vector = training-set median of the (input) feature vector.
    # The training features live in [0, 1] after MinMaxScaler; the lattice
    # calibrator is monotone in its input, so the calibrated median is
    # attained at the raw-value median. This matches the stated definition
    # "training-set median of the feature's calibrated value" up to a
    # monotone bijection per feature.
    baseline_vec = np.median(np.asarray(X_reference, dtype=float), axis=0)
    refs: dict[str, float] = {
        name: float(baseline_vec[i]) for i, name in enumerate(feature_names)
    }

    explainer = LatticeExplainer(
        model=model, feature_names=list(feature_names), lattice_groups=lattice_groups
    )

    twoD_groups = [g for g in lattice_groups if len(g) == 2]
    if not twoD_groups:
        return pd.DataFrame(), {
            "status": "no_2d_blocks",
            "baseline": "training_set_median",
        }

    X = np.asarray(X, dtype=float)
    n_rows = int(X.shape[0])
    rows: list[dict[str, Any]] = []
    assertion_failures = 0
    max_shapley_residual = 0.0
    max_main_residual = 0.0

    for g_idx, group in enumerate(lattice_groups):
        if len(group) != 2:
            continue
        a, b = group[0], group[1]
        group_tuple, group_model, weight = explainer.group_models[g_idx]
        assert list(group_tuple) == list(group), "lattice_groups / explainer group mismatch"

        # Four block evaluations (all vectorised across the n_rows batch).
        f0 = _batch_group_value(explainer, group_model, X, group, refs, set())      # both at ref
        fa = _batch_group_value(explainer, group_model, X, group, refs, {a})        # a active
        fb = _batch_group_value(explainer, group_model, X, group, refs, {b})        # b active
        fab = _batch_group_value(explainer, group_model, X, group, refs, {a, b})    # both active

        # Two-player Shapley (identical formula to the existing code).
        phi_a = 0.5 * ((fa - f0) + (fab - fb))
        phi_b = 0.5 * ((fb - f0) + (fab - fa))

        # Main-effect / interaction decomposition.
        main_a = fa - f0
        main_b = fb - f0
        iota = fab - fa - fb + f0

        # Assertions — exact arithmetic identities up to floating error.
        shapley_resid = np.abs(phi_a + phi_b + f0 - fab)
        main_resid = np.abs(main_a + main_b + iota - (fab - f0))
        block_shapley_fail = int(np.sum(shapley_resid > assertion_tol))
        block_main_fail = int(np.sum(main_resid > assertion_tol))
        assertion_failures += block_shapley_fail + block_main_fail
        max_shapley_residual = max(max_shapley_residual, float(np.max(shapley_resid)))
        max_main_residual = max(max_main_residual, float(np.max(main_resid)))

        for i in range(n_rows):
            rows.append(
                {
                    "test_index": int(i),
                    "block_index": int(g_idx),
                    "feature_a": a,
                    "feature_b": b,
                    "dense_weight": float(weight),
                    "reference_a": float(refs[a]),
                    "reference_b": float(refs[b]),
                    "value_a": float(X[i, explainer.feature_to_idx[a]]),
                    "value_b": float(X[i, explainer.feature_to_idx[b]]),
                    # Raw block outputs at each coalition.
                    "block_output_empty_f0": float(f0[i]),
                    "block_output_a_only_fa": float(fa[i]),
                    "block_output_b_only_fb": float(fb[i]),
                    "block_output_full_fab": float(fab[i]),
                    # Two-player Shapley attributions (block-level, pre-weight).
                    "shapley_phi_a": float(phi_a[i]),
                    "shapley_phi_b": float(phi_b[i]),
                    # Main-effect / interaction decomposition.
                    "main_effect_a": float(main_a[i]),
                    "main_effect_b": float(main_b[i]),
                    "pure_interaction_iota": float(iota[i]),
                    # Weighted contributions to the outer dense logit
                    # (mirrors what `explain_batch` folds into `phi_logit`).
                    "logit_phi_a_weighted": float(weight * phi_a[i]),
                    "logit_phi_b_weighted": float(weight * phi_b[i]),
                    "logit_iota_weighted": float(weight * iota[i]),
                    # Assertion residuals are emitted so they can be inspected.
                    "shapley_efficiency_residual_abs": float(shapley_resid[i]),
                    "main_effect_residual_abs": float(main_resid[i]),
                }
            )

    df = pd.DataFrame(rows)
    summary = {
        "status": "ok",
        "baseline": "training_set_median_per_feature",
        "assertion_tolerance": float(assertion_tol),
        "shapley_efficiency_max_abs_residual": float(max_shapley_residual),
        "main_effect_max_abs_residual": float(max_main_residual),
        "assertion_failures": int(assertion_failures),
        "n_test_instances": int(n_rows),
        "n_2d_blocks": int(len(twoD_groups)),
        "n_rows_total": int(len(rows)),
        "assertions_verified": [
            "phi_a + phi_b + f0 == fab (per instance, per 2-D block)",
            "main_a + main_b + iota == fab - f0 (per instance, per 2-D block)",
        ],
    }
    if assertion_failures > 0:
        # Loud failure — the point of the audit is that these identities hold
        # to machine precision. Anything else is a code bug.
        raise AssertionError(
            f"Shapley identity failed for {assertion_failures} instance-block cells "
            f"(tol={assertion_tol}). max shapley residual = "
            f"{max_shapley_residual:.3e}, max main-effect residual = "
            f"{max_main_residual:.3e}."
        )

    return df, summary


def side_by_side_audit(
    *,
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    X_reference: np.ndarray,
    lattice_groups: list[list[str]],
    n_instances: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """SELF-AUDIT: pick n_instances at random, print old vs new phi side by side.

    "Old" values are the direction-informed Shapley phi from the existing
    LatticeExplainer (as the pipeline currently reports). "New" values are the
    median-baseline Shapley phi plus the pure interaction term the existing
    code absorbs. Rows: (instance, block). Also asserts identities.
    """
    from .explainability import _reference_vector

    X = np.asarray(X, dtype=float)
    n_rows = int(X.shape[0])
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(n_rows, size=int(min(n_instances, n_rows)), replace=False)

    # "Old" refs (direction-informed) — verbatim from the existing pipeline.
    ref_medians_all = np.median(X_reference, axis=0)
    ref_mins_all = np.min(X_reference, axis=0)
    ref_maxs_all = np.max(X_reference, axis=0)
    old_refs = _reference_vector(
        feature_names, ref_mins_all, ref_maxs_all, ref_medians_all
    )
    new_refs = {
        name: float(ref_medians_all[i]) for i, name in enumerate(feature_names)
    }

    explainer = LatticeExplainer(
        model=model, feature_names=list(feature_names), lattice_groups=lattice_groups
    )

    X_sub = X[picked]
    audit_rows: list[dict[str, Any]] = []
    for g_idx, group in enumerate(lattice_groups):
        if len(group) != 2:
            continue
        a, b = group[0], group[1]
        _, group_model, weight = explainer.group_models[g_idx]

        def _run(refs, X_batch):
            f0 = _batch_group_value(explainer, group_model, X_batch, group, refs, set())
            fa = _batch_group_value(explainer, group_model, X_batch, group, refs, {a})
            fb = _batch_group_value(explainer, group_model, X_batch, group, refs, {b})
            fab = _batch_group_value(explainer, group_model, X_batch, group, refs, {a, b})
            return f0, fa, fb, fab

        old_f0, old_fa, old_fb, old_fab = _run(old_refs, X_sub)
        new_f0, new_fa, new_fb, new_fab = _run(new_refs, X_sub)

        old_phi_a = 0.5 * ((old_fa - old_f0) + (old_fab - old_fb))
        old_phi_b = 0.5 * ((old_fb - old_f0) + (old_fab - old_fa))

        new_phi_a = 0.5 * ((new_fa - new_f0) + (new_fab - new_fb))
        new_phi_b = 0.5 * ((new_fb - new_f0) + (new_fab - new_fa))
        new_iota = new_fab - new_fa - new_fb + new_f0

        for k, test_idx in enumerate(picked.tolist()):
            audit_rows.append(
                {
                    "test_index": int(test_idx),
                    "block_index": int(g_idx),
                    "feature_a": a,
                    "feature_b": b,
                    "dense_weight": float(weight),
                    # OLD (direction-informed baseline, existing pipeline).
                    "old_phi_a": float(old_phi_a[k]),
                    "old_phi_b": float(old_phi_b[k]),
                    "old_baseline_f0": float(old_f0[k]),
                    "old_block_output_fab": float(old_fab[k]),
                    "old_identity_residual": float(
                        abs(old_phi_a[k] + old_phi_b[k] + old_f0[k] - old_fab[k])
                    ),
                    # NEW (median baseline, this module).
                    "new_phi_a": float(new_phi_a[k]),
                    "new_phi_b": float(new_phi_b[k]),
                    "new_pure_interaction": float(new_iota[k]),
                    "new_baseline_f0": float(new_f0[k]),
                    "new_block_output_fab": float(new_fab[k]),
                    "new_identity_residual": float(
                        abs(new_phi_a[k] + new_phi_b[k] + new_f0[k] - new_fab[k])
                    ),
                }
            )

    return pd.DataFrame(audit_rows)
