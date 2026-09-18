from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import tf_keras as keras
except Exception:
    keras = None

from .config import MONOTONIC_CONSTRAINTS, THRESHOLD


def _reference_vector(
    feature_names: List[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    medians: np.ndarray,
) -> Dict[str, float]:
    refs: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        direction = MONOTONIC_CONSTRAINTS.get(name, 0)
        if direction > 0:
            refs[name] = float(mins[i])
        elif direction < 0:
            refs[name] = float(maxs[i])
        else:
            refs[name] = float(medians[i])
    return refs


def local_feature_effects(
    model,
    x: np.ndarray,
    feature_names: List[str],
    refs: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    base_score = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    effects: Dict[str, float] = {}

    for i, name in enumerate(feature_names):
        x_ref = x.copy()
        x_ref[i] = refs[name]
        ref_score = float(model.predict_proba(x_ref.reshape(1, -1))[0, 1])
        effects[name] = base_score - ref_score

    return base_score, effects


@dataclass
class LatticeExplainer:
    model: object
    feature_names: List[str]
    lattice_groups: List[List[str]]

    def __post_init__(self) -> None:
        if keras is None or not hasattr(self.model, "model"):
            raise ValueError("LatticeExplainer requires a Keras lattice model.")

        self.keras_model = getattr(self.model, "model")
        self.feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}

        dense = self.keras_model.get_layer("pd_output")
        dense_w, dense_b = dense.get_weights()
        self.dense_w = np.array(dense_w).reshape(-1)
        self.dense_b = float(np.array(dense_b).reshape(-1)[0])

        self.group_models = []
        for g_idx, group in enumerate(self.lattice_groups):
            layer = self.keras_model.get_layer(f"lattice_{g_idx}")
            gm = keras.Model(self.keras_model.inputs, layer.output)
            self.group_models.append((group, gm, float(self.dense_w[g_idx])))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _group_value(
        self,
        group_model,
        x: np.ndarray,
        group: List[str],
        refs: Dict[str, float],
        active_subset: set[str],
    ) -> float:
        x_mod = x.copy()
        for name in group:
            if name not in active_subset:
                x_mod[self.feature_to_idx[name]] = refs[name]
        out = group_model.predict(x_mod.reshape(1, -1), verbose=0)
        return float(np.array(out).reshape(-1)[0])

    def _group_values_batch(
        self,
        group_model,
        X: np.ndarray,
        group: List[str],
        refs: Dict[str, float],
        active_subset: set[str],
    ) -> np.ndarray:
        X_mod = X.copy()
        for name in group:
            if name not in active_subset:
                X_mod[:, self.feature_to_idx[name]] = refs[name]
        out = group_model.predict(X_mod, verbose=0)
        return np.array(out).reshape(-1)

    def explain_batch(self, X: np.ndarray, refs: Dict[str, float]) -> Dict[str, Any]:
        n_rows = int(X.shape[0])
        n_features = len(self.feature_names)
        phi_logit = np.zeros((n_rows, n_features), dtype=float)

        base_logit = np.full(n_rows, self.dense_b, dtype=float)
        full_logit = np.full(n_rows, self.dense_b, dtype=float)

        for group, group_model, weight in self.group_models:
            group_set = set(group)
            empty_val = self._group_values_batch(group_model, X, group, refs, set())
            full_val = self._group_values_batch(group_model, X, group, refs, group_set)

            base_logit += weight * empty_val
            full_logit += weight * full_val

            if len(group) == 1:
                name = group[0]
                idx = self.feature_to_idx[name]
                phi_logit[:, idx] += weight * (full_val - empty_val)
                continue

            if len(group) == 2:
                a, b = group[0], group[1]
                a_idx = self.feature_to_idx[a]
                b_idx = self.feature_to_idx[b]

                fa = self._group_values_batch(group_model, X, group, refs, {a})
                fb = self._group_values_batch(group_model, X, group, refs, {b})

                phi_a = 0.5 * ((fa - empty_val) + (full_val - fb))
                phi_b = 0.5 * ((fb - empty_val) + (full_val - fa))

                phi_logit[:, a_idx] += weight * phi_a
                phi_logit[:, b_idx] += weight * phi_b
                continue

            # Defensive fallback for unexpected group sizes (>2).
            for name in group:
                idx = self.feature_to_idx[name]
                val = self._group_values_batch(group_model, X, group, refs, {name})
                phi_logit[:, idx] += weight * (val - empty_val) / max(len(group), 1)

        base_score = 1.0 / (1.0 + np.exp(-base_logit))
        pd_score = 1.0 / (1.0 + np.exp(-full_logit))

        return {
            "base_logit": base_logit,
            "logit": full_logit,
            "base_score": base_score,
            "pd_score": pd_score,
            "phi_logit": phi_logit,
        }

    def explain_one(self, x: np.ndarray, refs: Dict[str, float]) -> Dict[str, Any]:
        phi_logit = {name: 0.0 for name in self.feature_names}

        base_logit = self.dense_b
        full_logit = self.dense_b

        for group, group_model, weight in self.group_models:
            group_set = set(group)
            empty_val = self._group_value(group_model, x, group, refs, set())
            full_val = self._group_value(group_model, x, group, refs, group_set)
            base_logit += weight * empty_val
            full_logit += weight * full_val

            if len(group) == 1:
                name = group[0]
                phi = full_val - empty_val
                phi_logit[name] += weight * phi
                continue

            if len(group) == 2:
                a, b = group[0], group[1]
                fa = self._group_value(group_model, x, group, refs, {a})
                fb = self._group_value(group_model, x, group, refs, {b})
                fab = full_val
                f0 = empty_val

                phi_a = 0.5 * ((fa - f0) + (fab - fb))
                phi_b = 0.5 * ((fb - f0) + (fab - fa))

                phi_logit[a] += weight * phi_a
                phi_logit[b] += weight * phi_b
                continue

            # Defensive fallback for unexpected group sizes (>2).
            for name in group:
                val = self._group_value(group_model, x, group, refs, {name})
                phi_logit[name] += weight * (val - empty_val) / max(len(group), 1)

        return {
            "base_logit": float(base_logit),
            "logit": float(full_logit),
            "base_score": self._sigmoid(base_logit),
            "pd_score": self._sigmoid(full_logit),
            "phi_logit": {k: float(v) for k, v in phi_logit.items()},
        }


def build_explainability_report(
    model,
    X: np.ndarray,
    feature_names: List[str],
    X_reference: np.ndarray,
    lattice_groups: List[List[str]] | None = None,
    top_k: int = 4,
    decision_threshold: float = THRESHOLD,
) -> pd.DataFrame:
    mins = np.min(X_reference, axis=0)
    maxs = np.max(X_reference, axis=0)
    medians = np.median(X_reference, axis=0)
    refs = _reference_vector(feature_names, mins, maxs, medians)

    use_exact_lattice = bool(lattice_groups) and hasattr(model, "model") and keras is not None
    explainer: LatticeExplainer | None = None
    if use_exact_lattice:
        try:
            explainer = LatticeExplainer(model=model, feature_names=feature_names, lattice_groups=lattice_groups or [])
        except Exception:
            explainer = None

    rows = []
    if explainer is not None:
        batch_result = explainer.explain_batch(X, refs)
        pd_scores = batch_result["pd_score"]
        base_scores = batch_result["base_score"]
        phi_logit = batch_result["phi_logit"]

        for idx in range(len(X)):
            effects = {
                feature_names[j]: float(phi_logit[idx, j])
                for j in range(len(feature_names))
            }
            sorted_effects = sorted(effects.items(), key=lambda kv: kv[1], reverse=True)
            reason_features = [name for name, value in sorted_effects if value > 0][:top_k]

            rows.append(
                {
                    "test_index": int(idx),
                    "pd_score": float(pd_scores[idx]),
                    "base_score": float(base_scores[idx]),
                    "decision": "Deny" if float(pd_scores[idx]) >= decision_threshold else "Approve",
                    "reason_codes": json.dumps(reason_features),
                    "explain_mode": "exact_lattice_logit",
                    "effect_type": "phi_logit",
                    "feature_effects": json.dumps({k: float(v) for k, v in sorted_effects}),
                }
            )
    else:
        baseline_x = np.array(list(refs.values()), dtype=float).reshape(1, -1)
        base_score_ref = float(model.predict_proba(baseline_x)[0, 1])

        for idx, x in enumerate(X):
            score, effects = local_feature_effects(model, x, feature_names, refs)
            sorted_effects = sorted(effects.items(), key=lambda kv: kv[1], reverse=True)
            reason_features = [name for name, value in sorted_effects if value > 0][:top_k]

            rows.append(
                {
                    "test_index": int(idx),
                    "pd_score": float(score),
                    "base_score": float(base_score_ref),
                    "decision": "Deny" if score >= decision_threshold else "Approve",
                    "reason_codes": json.dumps(reason_features),
                    "explain_mode": "perturbation_score",
                    "effect_type": "score_effect",
                    "feature_effects": json.dumps({k: float(v) for k, v in sorted_effects}),
                }
            )

    return pd.DataFrame(rows)
