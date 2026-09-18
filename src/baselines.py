from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

from .config import (
    PACK_DIR,
    BASELINE_RF_N_ESTIMATORS,
    BASELINE_XGB_N_ESTIMATORS,
    MONOTONIC_CONSTRAINTS,
    SKIP_XGBOOST,
    THRESHOLD,
)


def _load_tuned_baseline_params() -> dict:
    if os.getenv("BASELINE_USE_TUNED", "0").strip() != "1":
        return {}
    path = PACK_DIR / "metrics" / "baseline_tuning_best.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    decision_threshold: float = THRESHOLD,
) -> Dict[str, float]:
    y_proba_safe = np.clip(y_proba, 1e-6, 1 - 1e-6)
    y_pred = (y_proba >= decision_threshold).astype(int)
    return {
        "status": "ok",
        "error_type": "",
        "fallback_used": False,
        "auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba_safe)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    decision_threshold: float = THRESHOLD,
    *,
    fitted_probas_out: Optional[Dict[str, np.ndarray]] = None,
    fitted_models_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit the three original baselines (LR, RF, XGBoost) and return metrics.

    ``fitted_probas_out``: if a dict is passed, it is populated with the
    positive-class probabilities on ``X_test`` for each successful fit
    (``{"logistic_regression": np.ndarray, ...}``). No effect on returned
    metrics. Used by the matched-operating-point analysis so the exact
    same fits produce both the reported metrics and the recomputed
    threshold-swept operating points.

    ``fitted_models_out``: if a dict is passed, it is populated with the
    fitted sklearn / xgboost estimator itself, so downstream stability
    benchmarks can call ``predict_proba`` on perturbed inputs
    without re-training. No effect on returned metrics.
    """
    outputs: Dict[str, Dict[str, float]] = {}
    tuned = _load_tuned_baseline_params()

    def _stash(name: str, proba: np.ndarray) -> None:
        if fitted_probas_out is not None:
            fitted_probas_out[name] = np.asarray(proba, dtype=float)

    def _stash_model(name: str, model: Any) -> None:
        if fitted_models_out is not None:
            fitted_models_out[name] = model

    def _empty_status(status: str, error_type: str = "") -> Dict[str, float]:
        return {
            "status": status,
            "error_type": error_type,
            "fallback_used": True,
            "auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    lr_cfg = tuned.get("logistic_regression", {})
    lr = LogisticRegression(
        C=float(lr_cfg.get("C", 1.0)),
        class_weight=lr_cfg.get("class_weight", None),
        max_iter=1500,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    _stash("logistic_regression", lr_proba)
    _stash_model("logistic_regression", lr)
    outputs["logistic_regression"] = _metrics(
        y_test,
        lr_proba,
        decision_threshold=decision_threshold,
    )

    rf_cfg = tuned.get("random_forest", {})
    rf_max_depth = rf_cfg.get("max_depth", None)
    if isinstance(rf_max_depth, str) and rf_max_depth.lower() == "none":
        rf_max_depth = None
    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", BASELINE_RF_N_ESTIMATORS)),
        max_depth=None if rf_max_depth is None else int(rf_max_depth),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
        class_weight=rf_cfg.get("class_weight", None),
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    _stash("random_forest", rf_proba)
    _stash_model("random_forest", rf)
    outputs["random_forest"] = _metrics(
        y_test,
        rf_proba,
        decision_threshold=decision_threshold,
    )

    if SKIP_XGBOOST:
        outputs["xgboost"] = _empty_status("skipped", "SKIP_XGBOOST=1")
        return outputs

    try:
        from xgboost import XGBClassifier

        xgb_cfg = tuned.get("xgboost", {})
        xgb = XGBClassifier(
            n_estimators=int(xgb_cfg.get("n_estimators", BASELINE_XGB_N_ESTIMATORS)),
            max_depth=int(xgb_cfg.get("max_depth", 5)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.04)),
            subsample=float(xgb_cfg.get("subsample", 0.85)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.85)),
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb.fit(X_train, y_train)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        _stash("xgboost", xgb_proba)
        _stash_model("xgboost", xgb)
        outputs["xgboost"] = _metrics(
            y_test,
            xgb_proba,
            decision_threshold=decision_threshold,
        )
    except Exception as exc:
        outputs["xgboost"] = _empty_status("error", type(exc).__name__)

    return outputs


def build_monotone_constraints_vector(
    feature_names: List[str],
    constraint_map: Optional[Dict[str, int]] = None,
    *,
    verbose: bool = True,
) -> Tuple[int, ...]:
    """Map MONOTONIC_CONSTRAINTS onto training-column order for XGBoost.

    Returns a tuple of {-1, 0, +1} whose length equals len(feature_names).
    Unknown features fall back to 0 (unconstrained) and are called out in the
    verification printout recording the exact mapping used.
    """
    src = MONOTONIC_CONSTRAINTS if constraint_map is None else constraint_map
    vec: List[int] = []
    unknowns: List[str] = []
    for name in feature_names:
        raw = src.get(name, 0)
        c = int(raw)
        if c not in (-1, 0, 1):
            raise ValueError(
                f"monotone_xgboost: constraint for {name!r} must be -1/0/+1, got {raw!r}"
            )
        if name not in src:
            unknowns.append(name)
        vec.append(c)

    if len(vec) != len(feature_names):
        raise AssertionError(
            "monotone_xgboost: constraint vector length "
            f"{len(vec)} != feature count {len(feature_names)}"
        )

    if verbose:
        print(
            f"[monotone_xgboost] feature -> monotone_constraints mapping "
            f"(n={len(feature_names)}):"
        )
        for i, (name, c) in enumerate(zip(feature_names, vec)):
            print(f"  [{i:>2}] {name:<44} {c:+d}")
        pos = sum(1 for c in vec if c > 0)
        neg = sum(1 for c in vec if c < 0)
        zer = sum(1 for c in vec if c == 0)
        print(
            f"[monotone_xgboost] summary: +1={pos}, -1={neg}, 0={zer}, "
            f"total={len(vec)}"
        )
        if unknowns:
            print(
                "[monotone_xgboost] WARNING: features absent from "
                f"MONOTONIC_CONSTRAINTS defaulted to 0: {unknowns}"
            )

    return tuple(vec)


def train_monotone_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    *,
    decision_threshold: float = THRESHOLD,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Train an XGBoost baseline with monotone_constraints from the registry.

    Uses the same 40-trial-tuned hyperparameters as the unconstrained xgboost
    baseline (or the same fixed defaults when tuning is disabled), so the only
    controlled difference between the two models is the monotone constraint.
    Returns (metrics_dict, fitted_model). fitted_model is None when xgboost is
    skipped or fails to train.
    """

    def _empty(status: str, error_type: str = "") -> Dict[str, Any]:
        return {
            "status": status,
            "error_type": error_type,
            "fallback_used": True,
            "auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    if SKIP_XGBOOST:
        return _empty("skipped", "SKIP_XGBOOST=1"), None

    if not feature_names:
        return _empty("error", "no_feature_names"), None

    if X_train.shape[1] != len(feature_names):
        return _empty(
            "error",
            f"feature_name_length_mismatch:{X_train.shape[1]}_vs_{len(feature_names)}",
        ), None

    try:
        constraints = build_monotone_constraints_vector(
            list(feature_names), verbose=verbose
        )
    except Exception as exc:
        return _empty("error", type(exc).__name__), None

    try:
        from xgboost import XGBClassifier

        tuned = _load_tuned_baseline_params()
        # Prefer a monotone-specific tuned config if the tuner has produced one;
        # otherwise reuse the unconstrained xgboost tuned config so both models
        # share the exact 40-trial-tuned hyperparameters and differ only in the
        # monotone constraint. This is the parity contract.
        cfg = tuned.get("monotone_xgboost") or tuned.get("xgboost", {})
        model = XGBClassifier(
            n_estimators=int(cfg.get("n_estimators", BASELINE_XGB_N_ESTIMATORS)),
            max_depth=int(cfg.get("max_depth", 5)),
            learning_rate=float(cfg.get("learning_rate", 0.04)),
            subsample=float(cfg.get("subsample", 0.85)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.85)),
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            monotone_constraints=constraints,
        )
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        return _metrics(y_test, y_proba, decision_threshold=decision_threshold), model
    except Exception as exc:
        return _empty("error", type(exc).__name__), None


def _load_interpretable_tuned_params() -> dict:
    """Load tuned hyperparameters for the EBM and monotone-GAM baselines."""
    path = PACK_DIR / "metrics" / "interpretable_baseline_tuning_best.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class _GamProbaWrapper:
    """Give pygam.LogisticGAM a scikit-learn-shaped predict_proba(X)->[n,2]."""

    def __init__(self, gam) -> None:
        self.gam = gam

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        p1 = np.asarray(self.gam.predict_proba(arr), dtype=float).reshape(-1)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p1, p1])


def train_ebm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    *,
    decision_threshold: float = THRESHOLD,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Explainable Boosting Machine with monotone constraints.

    Interpretable-by-design competitor. Uses the 40-trial-
    parity-tuned hyperparameters from ``interpretable_baseline_tuning_best.json``
    when present; otherwise falls back to the library defaults. Returns the
    (metrics_dict, fitted_model) tuple. fitted_model is None on failure so the
    perturbation protocol can be skipped gracefully.
    """

    def _empty(status: str, error_type: str = "") -> Dict[str, Any]:
        return {
            "status": status,
            "error_type": error_type,
            "fallback_used": True,
            "auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    if not feature_names:
        return _empty("error", "no_feature_names"), None
    if X_train.shape[1] != len(feature_names):
        return _empty(
            "error",
            f"feature_name_length_mismatch:{X_train.shape[1]}_vs_{len(feature_names)}",
        ), None

    try:
        constraints = build_monotone_constraints_vector(
            list(feature_names), verbose=verbose
        )
    except Exception as exc:
        return _empty("error", type(exc).__name__), None

    try:
        from interpret.glassbox import ExplainableBoostingClassifier

        tuned = _load_interpretable_tuned_params()
        cfg = tuned.get("ebm", {}) if isinstance(tuned, dict) else {}

        model = ExplainableBoostingClassifier(
            feature_names=list(feature_names),
            monotone_constraints=list(constraints),
            interactions=int(cfg.get("interactions", 10)),
            learning_rate=float(cfg.get("learning_rate", 0.02)),
            max_bins=int(cfg.get("max_bins", 256)),
            outer_bags=int(cfg.get("outer_bags", 14)),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        return _metrics(y_test, y_proba, decision_threshold=decision_threshold), model
    except Exception as exc:
        return _empty("error", type(exc).__name__), None


def _build_pygam_terms(
    feature_names: List[str],
    constraints: Tuple[int, ...],
    n_splines: int,
    spline_order: int,
    lam: float,
):
    """Build a pygam TermList honouring MONOTONIC_CONSTRAINTS per column."""
    from pygam import s

    terms = None
    for idx, direction in enumerate(constraints):
        if direction > 0:
            cstr = "monotonic_inc"
        elif direction < 0:
            cstr = "monotonic_dec"
        else:
            cstr = None
        term = s(
            idx,
            n_splines=int(n_splines),
            spline_order=int(spline_order),
            lam=float(lam),
            constraints=cstr,
        )
        terms = term if terms is None else terms + term
    return terms


def train_monotone_gam(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    *,
    decision_threshold: float = THRESHOLD,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Monotone-constrained logistic GAM (pyGAM).

    Per-feature spline smoothers with ``monotonic_inc``/``monotonic_dec``
    constraints mirroring the ``MONOTONIC_CONSTRAINTS`` registry so the
    directional structure matches XCreditScore, monotone-XGBoost, and EBM.
    Returns (metrics_dict, wrapped_model). The model is wrapped to expose a
    scikit-learn-shaped ``predict_proba`` for the perturbation protocol.
    """

    def _empty(status: str, error_type: str = "") -> Dict[str, Any]:
        return {
            "status": status,
            "error_type": error_type,
            "fallback_used": True,
            "auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    if not feature_names:
        return _empty("error", "no_feature_names"), None
    if X_train.shape[1] != len(feature_names):
        return _empty(
            "error",
            f"feature_name_length_mismatch:{X_train.shape[1]}_vs_{len(feature_names)}",
        ), None

    try:
        constraints = build_monotone_constraints_vector(
            list(feature_names), verbose=verbose
        )
    except Exception as exc:
        return _empty("error", type(exc).__name__), None

    try:
        from pygam import LogisticGAM

        tuned = _load_interpretable_tuned_params()
        cfg = tuned.get("monotone_gam", {}) if isinstance(tuned, dict) else {}
        n_splines = int(cfg.get("n_splines", 20))
        spline_order = int(cfg.get("spline_order", 3))
        lam = float(cfg.get("lam", 0.6))
        max_iter = int(cfg.get("max_iter", 200))

        terms = _build_pygam_terms(
            list(feature_names), constraints, n_splines, spline_order, lam
        )
        gam = LogisticGAM(terms, max_iter=max_iter, tol=1e-4)
        gam.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int))
        wrapped = _GamProbaWrapper(gam)
        y_proba = wrapped.predict_proba(X_test)[:, 1]
        return _metrics(y_test, y_proba, decision_threshold=decision_threshold), wrapped
    except Exception as exc:
        return _empty("error", type(exc).__name__), None
