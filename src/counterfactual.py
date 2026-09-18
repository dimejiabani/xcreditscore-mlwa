from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pulp
from sklearn.linear_model import LinearRegression

from .config import IMMUTABLE_FEATURES, MONOTONIC_CONSTRAINTS, THRESHOLD
from .config import (
    COUNTERFACTUAL_BASE_COST_WEIGHTS,
    COUNTERFACTUAL_DEFAULT_MAX_STEP_FRACTION,
    COUNTERFACTUAL_EXACT_MAX_CASES,
    COUNTERFACTUAL_EXACT_MIP_GAP,
    COUNTERFACTUAL_EXACT_TIME_LIMIT,
    COUNTERFACTUAL_MAX_STEP_FRACTIONS,
    COUNTERFACTUAL_NEAR_MARGIN,
    COUNTERFACTUAL_REFINE_MAX_ROUNDS,
)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1.0 - p)))


@dataclass
class CounterfactualResult:
    index: int
    feasible: bool
    original_score: float
    new_score: float
    total_cost: float
    changed_features: Dict[str, float]
    solver_status: str
    score_gap_to_threshold: float
    recourse_band: str


def _classify_recourse_band(
    new_score: float,
    decision_threshold: float,
    near_margin: float = COUNTERFACTUAL_NEAR_MARGIN,
) -> str:
    if new_score < decision_threshold:
        return "feasible"
    if new_score < (decision_threshold + near_margin):
        return "near_feasible"
    return "infeasible"


def _build_feature_costs(feature_names: List[str], X_reference: np.ndarray | None = None) -> np.ndarray:
    costs = np.ones(len(feature_names), dtype=float)
    for i, name in enumerate(feature_names):
        costs[i] = float(COUNTERFACTUAL_BASE_COST_WEIGHTS.get(name, 1.0))

    # Policy-calibration from observed volatility: lower-volatility features are
    # treated as harder to change in a one-step intervention.
    if X_reference is not None and X_reference.size > 0:
        stds = np.std(X_reference, axis=0).astype(float)
        nonzero = stds[stds > 1e-8]
        if nonzero.size > 0:
            median_std = float(np.median(nonzero))
            for i in range(len(feature_names)):
                s = float(max(stds[i], 1e-8))
                volatility_scale = float(np.clip(median_std / s, 0.6, 2.5))
                costs[i] = float(costs[i] * volatility_scale)
    return costs


def _build_step_limits(feature_names: List[str], mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    limits = np.zeros(len(feature_names), dtype=float)
    for i, name in enumerate(feature_names):
        span = float(max(maxs[i] - mins[i], 1e-8))
        frac = float(
            COUNTERFACTUAL_MAX_STEP_FRACTIONS.get(
                name,
                COUNTERFACTUAL_DEFAULT_MAX_STEP_FRACTION,
            )
        )
        limits[i] = max(1e-8, span * max(0.0, frac))
    return limits


def _construct_bounds(
    x0: np.ndarray,
    feature_names: List[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    step_limits: np.ndarray | None = None,
) -> List[tuple[float, float]]:
    bounds: List[tuple[float, float]] = []
    for i, name in enumerate(feature_names):
        if name in IMMUTABLE_FEATURES:
            bounds.append((float(x0[i]), float(x0[i])))
            continue

        direction = MONOTONIC_CONSTRAINTS.get(name, 0)
        if direction > 0:
            low, high = float(mins[i]), float(x0[i])
        elif direction < 0:
            low, high = float(x0[i]), float(maxs[i])
        else:
            low, high = float(mins[i]), float(maxs[i])

        if step_limits is not None:
            step = float(max(step_limits[i], 1e-8))
            low = max(low, float(x0[i]) - step)
            high = min(high, float(x0[i]) + step)

        if high < low:
            low = high = float(x0[i])
        bounds.append((low, high))
    return bounds


def _fit_linear_surrogate(model, X_reference: np.ndarray) -> LinearRegression:
    surrogate_target = model.predict_proba(X_reference)[:, 1]
    surrogate = LinearRegression()
    surrogate.fit(X_reference, surrogate_target)
    return surrogate


def _extract_lattice_artifacts(model, feature_names: List[str], lattice_groups: List[List[str]]):
    if not hasattr(model, "model"):
        raise ValueError("Exact lattice solver requires Keras lattice model wrapper")

    keras_model = getattr(model, "model")

    calibrators: Dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for name in feature_names:
        cal = keras_model.get_layer(f"cal_{name}")
        keypoints = np.array(cal.input_keypoints, dtype=float).reshape(-1)
        cal_w = cal.get_weights()
        if not cal_w:
            raise ValueError(f"Missing calibrator weights for feature {name}")
        outputs = np.array(cal_w[0], dtype=float).reshape(-1)
        direction = MONOTONIC_CONSTRAINTS.get(name, 0)
        calibrators[name] = (keypoints, outputs, direction)

    lattices = []
    for g_idx, group in enumerate(lattice_groups):
        lat_layer = keras_model.get_layer(f"lattice_{g_idx}")
        cfg = lat_layer.get_config()
        sizes = [int(v) for v in cfg.get("lattice_sizes", [])]
        lat_w = lat_layer.get_weights()
        if not lat_w:
            raise ValueError(f"Missing lattice weights for group {g_idx}")
        vertices = np.array(lat_w[0], dtype=float).reshape(tuple(sizes))
        lattices.append((group, sizes, vertices))

    dense = keras_model.get_layer("pd_output")
    dense_w, dense_b = dense.get_weights()
    dense_w = np.array(dense_w, dtype=float).reshape(-1)
    dense_b = float(np.array(dense_b, dtype=float).reshape(-1)[0])

    return calibrators, lattices, dense_w, dense_b


def _generate_counterfactual_exact_gurobi(
    model,
    x0: np.ndarray,
    feature_names: List[str],
    lattice_groups: List[List[str]],
    bounds: List[tuple[float, float]],
    costs: np.ndarray,
    decision_threshold: float,
) -> tuple[bool, np.ndarray, float, str, float]:
    import gurobipy as gp  # type: ignore[import-not-found]

    x0 = x0.astype(float)
    n = len(feature_names)
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    calibrators, lattices, dense_w, dense_b = _extract_lattice_artifacts(
        model,
        feature_names,
        lattice_groups,
    )

    m = gp.Model("counterfactual_exact_lattice")
    m.Params.OutputFlag = 0
    # Keep exact formulation but cap per-instance runtime for batch completion.
    m.Params.TimeLimit = float(COUNTERFACTUAL_EXACT_TIME_LIMIT)
    m.Params.MIPGap = float(COUNTERFACTUAL_EXACT_MIP_GAP)

    x_vars = [
        m.addVar(lb=bounds[i][0], ub=bounds[i][1], vtype=gp.GRB.CONTINUOUS, name=f"x_{i}")
        for i in range(n)
    ]
    d_plus = [m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"dplus_{i}") for i in range(n)]
    d_minus = [m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"dminus_{i}") for i in range(n)]
    z = [m.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY, name=f"z_{i}") for i in range(n)]

    big_m = [max(abs(bounds[i][1] - bounds[i][0]), 1e-6) for i in range(n)]
    for i in range(n):
        m.addConstr(x_vars[i] - float(x0[i]) == d_plus[i] - d_minus[i], name=f"delta_{i}")
        m.addConstr(d_plus[i] + d_minus[i] <= big_m[i] * z[i], name=f"link_{i}")

    # Calibrated feature variables (exact PWL constraints).
    t_vars = {}
    for name in feature_names:
        i = feature_to_idx[name]
        keypoints, outputs, direction = calibrators[name]
        s_var = m.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name=f"s_{name}")
        t_var = m.addVar(lb=min(outputs), ub=max(outputs), vtype=gp.GRB.CONTINUOUS, name=f"t_{name}")

        if direction < 0:
            m.addConstr(s_var == 1.0 - x_vars[i], name=f"invert_{name}")
        else:
            m.addConstr(s_var == x_vars[i], name=f"direct_{name}")

        m.addGenConstrPWL(s_var, t_var, keypoints.tolist(), outputs.tolist(), name=f"pwl_{name}")
        t_vars[name] = t_var

    lattice_out_vars = []
    for g_idx, (group, sizes, vertices) in enumerate(lattices):
        if len(group) == 1:
            size = int(sizes[0])
            kps = np.linspace(0.0, 1.0, num=size)
            lam = [m.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name=f"lam_{g_idx}_{k}") for k in range(size)]
            m.addConstr(gp.quicksum(lam) == 1.0, name=f"lam_sum_{g_idx}")
            m.addSOS(gp.GRB.SOS_TYPE2, lam)

            t_name = group[0]
            m.addConstr(t_vars[t_name] == gp.quicksum(float(kps[k]) * lam[k] for k in range(size)), name=f"interp1d_{g_idx}")

            l_out = m.addVar(lb=float(np.min(vertices)), ub=float(np.max(vertices)), vtype=gp.GRB.CONTINUOUS, name=f"lout_{g_idx}")
            m.addConstr(l_out == gp.quicksum(float(vertices[k]) * lam[k] for k in range(size)), name=f"lout_def_{g_idx}")
            lattice_out_vars.append(l_out)
            continue

        if len(group) != 2:
            raise ValueError("Exact lattice solver currently supports 1D/2D lattices only")

        s1, s2 = int(sizes[0]), int(sizes[1])
        k1 = np.linspace(0.0, 1.0, num=s1)
        k2 = np.linspace(0.0, 1.0, num=s2)

        lam = {}
        for i1 in range(s1):
            for i2 in range(s2):
                lam[(i1, i2)] = m.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name=f"lam_{g_idx}_{i1}_{i2}")

        m.addConstr(gp.quicksum(lam[(i1, i2)] for i1 in range(s1) for i2 in range(s2)) == 1.0, name=f"lam2_sum_{g_idx}")

        row = [m.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name=f"row_{g_idx}_{i1}") for i1 in range(s1)]
        col = [m.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name=f"col_{g_idx}_{i2}") for i2 in range(s2)]

        for i1 in range(s1):
            m.addConstr(row[i1] == gp.quicksum(lam[(i1, i2)] for i2 in range(s2)), name=f"row_link_{g_idx}_{i1}")
        for i2 in range(s2):
            m.addConstr(col[i2] == gp.quicksum(lam[(i1, i2)] for i1 in range(s1)), name=f"col_link_{g_idx}_{i2}")

        m.addSOS(gp.GRB.SOS_TYPE2, row)
        m.addSOS(gp.GRB.SOS_TYPE2, col)

        m.addConstr(t_vars[group[0]] == gp.quicksum(float(k1[i1]) * row[i1] for i1 in range(s1)), name=f"t1_interp_{g_idx}")
        m.addConstr(t_vars[group[1]] == gp.quicksum(float(k2[i2]) * col[i2] for i2 in range(s2)), name=f"t2_interp_{g_idx}")

        l_out = m.addVar(lb=float(np.min(vertices)), ub=float(np.max(vertices)), vtype=gp.GRB.CONTINUOUS, name=f"lout_{g_idx}")
        m.addConstr(
            l_out == gp.quicksum(float(vertices[i1, i2]) * lam[(i1, i2)] for i1 in range(s1) for i2 in range(s2)),
            name=f"lout2_def_{g_idx}",
        )
        lattice_out_vars.append(l_out)

    # Enforce approve-side decision boundary under the chosen threshold.
    margin = 1e-3
    logit = dense_b + gp.quicksum(float(dense_w[i]) * lattice_out_vars[i] for i in range(len(lattice_out_vars)))
    m.addConstr(logit <= (_logit(decision_threshold) - margin), name="approval_exact")

    sparsity_lambda = 0.05
    m.setObjective(
        gp.quicksum(float(costs[i]) * (d_plus[i] + d_minus[i]) for i in range(n)) + sparsity_lambda * gp.quicksum(z),
        gp.GRB.MINIMIZE,
    )
    m.optimize()

    status_map = {
        gp.GRB.OPTIMAL: "Optimal",
        gp.GRB.TIME_LIMIT: "TimeLimit",
        gp.GRB.SUBOPTIMAL: "Suboptimal",
        gp.GRB.INFEASIBLE: "Infeasible",
    }
    status = status_map.get(m.Status, f"Status_{m.Status}")

    if m.Status in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT} and m.SolCount > 0:
        x_cf = np.array([float(v.X) for v in x_vars], dtype=float)
    else:
        x_cf = x0.copy()

    new_score = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
    feasible = bool(new_score < decision_threshold)
    total_cost = float(np.sum(costs * np.abs(x_cf - x0)))
    return feasible, x_cf, new_score, status, total_cost


def _refine_with_true_model(
    model,
    x_start: np.ndarray,
    x0: np.ndarray,
    feature_names: List[str],
    bounds: List[tuple[float, float]],
    costs: np.ndarray,
    decision_threshold: float,
    max_rounds: int = COUNTERFACTUAL_REFINE_MAX_ROUNDS,
) -> tuple[np.ndarray, float, int]:
    x_cf = x_start.copy()
    current_score = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
    rounds = 0

    if current_score < decision_threshold:
        return x_cf, current_score, rounds

    ranges = np.array([max(1e-6, b[1] - b[0]) for b in bounds], dtype=float)

    candidate_indices = [
        i
        for i, (low, high) in enumerate(bounds)
        if abs(high - low) > 1e-12 and costs[i] <= 2.0
    ]
    if not candidate_indices:
        candidate_indices = [i for i, (low, high) in enumerate(bounds) if abs(high - low) > 1e-12]

    # Prioritize features by local expected risk reduction per unit intervention cost.
    ranked_candidates = []
    for i in candidate_indices:
        name = feature_names[i]
        low, high = bounds[i]
        direction = MONOTONIC_CONSTRAINTS.get(name, 0)
        step = 0.05 * ranges[i]
        if direction > 0:
            cval = max(low, x_cf[i] - step)
        elif direction < 0:
            cval = min(high, x_cf[i] + step)
        else:
            cval = max(low, x_cf[i] - step)
        if abs(cval - x_cf[i]) < 1e-12:
            continue
        x_probe = x_cf.copy()
        x_probe[i] = cval
        probe_score = float(model.predict_proba(x_probe.reshape(1, -1))[0, 1])
        gain = max(0.0, current_score - probe_score)
        move_cost = float(costs[i] * abs(cval - x_cf[i])) + 1e-8
        ranked_candidates.append((gain / move_cost, i))

    if ranked_candidates:
        candidate_indices = [i for _, i in sorted(ranked_candidates, key=lambda x: x[0], reverse=True)]

    for _ in range(max_rounds):
        rounds += 1
        best_candidate = None
        best_score = current_score
        best_gain_ratio = 0.0

        for i in candidate_indices:
            name = feature_names[i]
            low, high = bounds[i]
            if abs(high - low) < 1e-12:
                continue

            direction = MONOTONIC_CONSTRAINTS.get(name, 0)
            step = 0.05 * ranges[i]

            candidates = []
            if direction > 0:
                candidates.append(max(low, x_cf[i] - step))
            elif direction < 0:
                candidates.append(min(high, x_cf[i] + step))
            else:
                candidates.append(max(low, x_cf[i] - step))
                candidates.append(min(high, x_cf[i] + step))

            for cval in candidates:
                if abs(cval - x_cf[i]) < 1e-12:
                    continue
                x_try = x_cf.copy()
                x_try[i] = cval
                score_try = float(model.predict_proba(x_try.reshape(1, -1))[0, 1])
                gain = current_score - score_try
                if gain <= 1e-6:
                    continue
                move_cost = float(costs[i] * abs(cval - x_cf[i])) + 1e-8
                gain_ratio = gain / move_cost
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_candidate = x_try
                    best_score = score_try

        if best_candidate is None:
            break

        x_cf = best_candidate
        current_score = best_score
        if current_score < decision_threshold:
            break

    return x_cf, current_score, rounds


def generate_counterfactual_mip(
    model,
    surrogate: LinearRegression,
    x0: np.ndarray,
    feature_names: List[str],
    feature_mins: np.ndarray,
    feature_maxs: np.ndarray,
    costs: np.ndarray | None = None,
    step_limits: np.ndarray | None = None,
    lattice_groups: List[List[str]] | None = None,
    decision_threshold: float = THRESHOLD,
) -> tuple[bool, np.ndarray, float, str, float]:
    x0 = x0.astype(float)
    if costs is None:
        costs = _build_feature_costs(feature_names)
    bounds = _construct_bounds(
        x0,
        feature_names,
        feature_mins,
        feature_maxs,
        step_limits=step_limits,
    )

    feasible = False
    x_cf = x0.copy()
    new_score = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
    status = "unknown"
    total_cost = float(np.sum(costs * np.abs(x_cf - x0)))

    exact_ready = bool(lattice_groups) and hasattr(model, "model")
    exact_failed = False
    if exact_ready:
        try:
            feasible, x_cf, new_score, status, total_cost = _generate_counterfactual_exact_gurobi(
                model,
                x0,
                feature_names,
                lattice_groups or [],
                bounds=bounds,
                costs=costs,
                decision_threshold=decision_threshold,
            )
            status = f"gurobi_exact:{status}"
        except Exception as exc:
            status = f"gurobi_exact_failed:{type(exc).__name__}"
            exact_failed = True

    if (not exact_ready) or exact_failed:
        try:
            feasible, x_cf, new_score, status, total_cost = _generate_counterfactual_mip_gurobi(
                model,
                surrogate,
                x0,
                feature_names,
                bounds=bounds,
                costs=costs,
                decision_threshold=decision_threshold,
            )
            status = f"gurobi:{status}"
        except Exception:
            feasible, x_cf, new_score, status, total_cost = _generate_counterfactual_mip_pulp(
                model,
                surrogate,
                x0,
                feature_names,
                bounds=bounds,
                costs=costs,
                decision_threshold=decision_threshold,
            )
            status = f"cbc:{status}"

    if not feasible:
        refined_x, refined_score, refine_rounds = _refine_with_true_model(
            model=model,
            x_start=x_cf,
            x0=x0,
            feature_names=feature_names,
            bounds=bounds,
            costs=costs,
            decision_threshold=decision_threshold,
        )
        refined_feasible = bool(refined_score < decision_threshold)
        if refined_feasible or refined_score < new_score:
            x_cf = refined_x
            new_score = refined_score
            feasible = refined_feasible
            total_cost = float(np.sum(costs * np.abs(x_cf - x0)))
            status = f"{status}|refined_{refine_rounds}"

    return feasible, x_cf, new_score, status, total_cost


def _generate_counterfactual_mip_pulp(
    model,
    surrogate: LinearRegression,
    x0: np.ndarray,
    feature_names: List[str],
    bounds: List[tuple[float, float]],
    costs: np.ndarray,
    decision_threshold: float,
) -> tuple[bool, np.ndarray, float, str, float]:
    x0 = x0.astype(float)

    n = len(feature_names)
    prob = pulp.LpProblem("counterfactual_mip", pulp.LpMinimize)

    x_vars = [
        pulp.LpVariable(f"x_{i}", lowBound=bounds[i][0], upBound=bounds[i][1], cat="Continuous")
        for i in range(n)
    ]
    d_plus = [pulp.LpVariable(f"dplus_{i}", lowBound=0.0, cat="Continuous") for i in range(n)]
    d_minus = [pulp.LpVariable(f"dminus_{i}", lowBound=0.0, cat="Continuous") for i in range(n)]
    z = [pulp.LpVariable(f"z_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(n)]

    big_m = [max(abs(bounds[i][1] - bounds[i][0]), 1e-6) for i in range(n)]

    for i in range(n):
        prob += x_vars[i] - float(x0[i]) == d_plus[i] - d_minus[i]
        prob += d_plus[i] + d_minus[i] <= big_m[i] * z[i]

    coef = surrogate.coef_.astype(float)
    intercept = float(surrogate.intercept_)
    margin = 0.01
    prob += (
        intercept + pulp.lpSum(float(coef[i]) * x_vars[i] for i in range(n)) <= (decision_threshold - margin)
    )

    sparsity_lambda = 0.05
    prob += pulp.lpSum(float(costs[i]) * (d_plus[i] + d_minus[i]) for i in range(n)) + (
        sparsity_lambda * pulp.lpSum(z)
    )

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=8)
    prob.solve(solver)

    status = pulp.LpStatus.get(prob.status, "Unknown")
    if status in {"Optimal", "Feasible"}:
        values: List[float] = []
        for i, v in enumerate(x_vars):
            val = v.value()
            values.append(float(val) if val is not None else float(x0[i]))
        x_cf = np.array(values, dtype=float)
    else:
        x_cf = x0.copy()

    new_score = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
    feasible = bool(new_score < decision_threshold)
    total_cost = float(np.sum(costs * np.abs(x_cf - x0)))
    return feasible, x_cf, new_score, status, total_cost


def _generate_counterfactual_mip_gurobi(
    model,
    surrogate: LinearRegression,
    x0: np.ndarray,
    feature_names: List[str],
    bounds: List[tuple[float, float]],
    costs: np.ndarray,
    decision_threshold: float,
) -> tuple[bool, np.ndarray, float, str, float]:
    import gurobipy as gp  # type: ignore[import-not-found]

    x0 = x0.astype(float)
    n = len(feature_names)

    m = gp.Model("counterfactual_mip")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = 8.0
    m.Params.MIPGap = 0.02

    x_vars = [m.addVar(lb=bounds[i][0], ub=bounds[i][1], vtype=gp.GRB.CONTINUOUS, name=f"x_{i}") for i in range(n)]
    d_plus = [m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"dplus_{i}") for i in range(n)]
    d_minus = [m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"dminus_{i}") for i in range(n)]
    z = [m.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY, name=f"z_{i}") for i in range(n)]

    big_m = [max(abs(bounds[i][1] - bounds[i][0]), 1e-6) for i in range(n)]
    for i in range(n):
        m.addConstr(x_vars[i] - float(x0[i]) == d_plus[i] - d_minus[i], name=f"delta_{i}")
        m.addConstr(d_plus[i] + d_minus[i] <= big_m[i] * z[i], name=f"link_{i}")

    coef = surrogate.coef_.astype(float)
    intercept = float(surrogate.intercept_)
    margin = 0.01
    m.addConstr(
        intercept + gp.quicksum(float(coef[i]) * x_vars[i] for i in range(n)) <= (decision_threshold - margin),
        name="approval",
    )

    sparsity_lambda = 0.05
    m.setObjective(
        gp.quicksum(float(costs[i]) * (d_plus[i] + d_minus[i]) for i in range(n)) + sparsity_lambda * gp.quicksum(z),
        gp.GRB.MINIMIZE,
    )
    m.optimize()

    status_map = {
        gp.GRB.OPTIMAL: "Optimal",
        gp.GRB.TIME_LIMIT: "TimeLimit",
        gp.GRB.SUBOPTIMAL: "Suboptimal",
        gp.GRB.INFEASIBLE: "Infeasible",
    }
    status = status_map.get(m.Status, f"Status_{m.Status}")

    if m.Status in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT} and m.SolCount > 0:
        x_cf = np.array([float(v.X) for v in x_vars])
    else:
        x_cf = x0.copy()

    new_score = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
    feasible = bool(new_score < decision_threshold)
    total_cost = float(np.sum(costs * np.abs(x_cf - x0)))
    return feasible, x_cf, new_score, status, total_cost


def batch_counterfactuals(
    model,
    X: np.ndarray,
    denied_indices: Iterable[int],
    feature_names: List[str],
    lattice_groups: List[List[str]] | None = None,
    max_cases: int = 200,
    exact_max_cases: int = COUNTERFACTUAL_EXACT_MAX_CASES,
    decision_threshold: float = THRESHOLD,
) -> List[CounterfactualResult]:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    surrogate = _fit_linear_surrogate(model, X)
    costs = _build_feature_costs(feature_names, X_reference=X)
    step_limits = _build_step_limits(feature_names, mins, maxs)

    results: List[CounterfactualResult] = []
    for case_no, idx in enumerate(list(denied_indices)[:max_cases]):
        x0 = X[idx]
        original_score = float(model.predict_proba(x0.reshape(1, -1))[0, 1])

        case_lattice_groups = lattice_groups
        if exact_max_cases >= 0 and case_no >= exact_max_cases:
            case_lattice_groups = None

        feasible, x_cf, new_score, solver_status, total_cost = generate_counterfactual_mip(
            model,
            surrogate,
            x0,
            feature_names,
            mins,
            maxs,
            costs=costs,
            step_limits=step_limits,
            lattice_groups=case_lattice_groups,
            decision_threshold=decision_threshold,
        )

        deltas = x_cf - x0
        changed_features = {
            feature_names[i]: float(deltas[i])
            for i in range(len(feature_names))
            if abs(deltas[i]) > 1e-8
        }
        score_gap = float(new_score - decision_threshold)
        recourse_band = _classify_recourse_band(new_score, decision_threshold)
        results.append(
            CounterfactualResult(
                index=int(idx),
                feasible=feasible,
                original_score=original_score,
                new_score=new_score,
                total_cost=total_cost,
                changed_features=changed_features,
                solver_status=solver_status,
                score_gap_to_threshold=score_gap,
                recourse_band=recourse_band,
            )
        )
    return results
