"""Head-to-head reason-code stability benchmark.

Runs KernelSHAP and LIME on XCreditScore and every baseline against the same
400 test instances / 8 perturbations / 5-sigma sweep. XCreditScore's own
exact-lattice-logit method is included as the third row so all three
attribution methods sit in the same table on identical inputs, rather than
being compared against stability figures imported from other studies.

Nothing in ``src/train_all.py``'s existing stability diagnostics is touched;
this module writes a distinct set of artifacts alongside them.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


DEFAULT_SIGMAS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10)


# --------------------------------------------------------------------------- #
# Predict-function adapters                                                   #
# --------------------------------------------------------------------------- #
def make_positive_class_predict(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap any of our models so SHAP sees ``f(X) -> P(y=1) ∈ ℝ^n``.

    The task calls this out explicitly for the TF-Lattice model. Every model
    in the pipeline (LR / RF / XGB / mono-XGB / EBM / GAM-wrapper /
    KerasLatticeWrapper) exposes ``predict_proba(X) -> [n, 2]``, so the
    wrapper is uniform.
    """

    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        proba = model.predict_proba(X)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim == 1:
            return arr
        return arr[:, 1]

    return _predict


def make_predict_proba_two_column(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """LIME expects the sklearn-shaped ``predict_proba(X) -> [n, 2]``."""

    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        proba = model.predict_proba(X)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim == 1:
            return np.column_stack([1.0 - arr, arr])
        return arr

    return _predict


def verify_shap_wrapper(
    model: Any,
    X_sample: np.ndarray,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """SELF-AUDIT step: max abs diff between wrapper and native predict.

    Returns the diff and both prediction vectors so a hard fail is loud.
    """
    wrap = make_positive_class_predict(model)
    wrap_out = wrap(X_sample)
    native = np.asarray(model.predict_proba(X_sample), dtype=float)
    if native.ndim == 1:
        native_pos = native
    else:
        native_pos = native[:, 1]
    diff = float(np.max(np.abs(wrap_out - native_pos)))
    return {
        "n_rows": int(X_sample.shape[0]),
        "max_abs_diff": diff,
        "tol": float(tol),
        "passes_tol": bool(diff <= tol),
        "wrapper_head": [float(v) for v in wrap_out[:5]],
        "native_head": [float(v) for v in native_pos[:5]],
    }


# --------------------------------------------------------------------------- #
# Reason-code extraction (top-K by absolute contribution)                     #
# --------------------------------------------------------------------------- #
def _top_k_reasons(
    contributions: np.ndarray, feature_names: list[str], top_k: int
) -> list[str]:
    """Return the top-K feature names ranked by |contribution|.

    Ties broken by original feature order (stable sort). Contributions
    length must equal len(feature_names).
    """
    c = np.abs(np.asarray(contributions, dtype=float)).ravel()
    if c.shape[0] != len(feature_names):
        raise ValueError(
            f"contributions length {c.shape[0]} != feature count {len(feature_names)}"
        )
    order = np.argsort(-c, kind="mergesort")
    return [feature_names[int(i)] for i in order[: int(top_k)]]


def _pair_scores(base_top: list[str], pert_top: list[str]) -> tuple[bool, bool, float]:
    """Return (full_set_match, top1_match, jaccard) for one base/perturbation pair."""
    base_set = set(base_top)
    pert_set = set(pert_top)
    intersection = len(base_set & pert_set)
    union = len(base_set | pert_set)
    jaccard = float(intersection / union) if union > 0 else 1.0
    full_set_match = base_set == pert_set
    top1_match = bool(base_top and pert_top and base_top[0] == pert_top[0])
    return full_set_match, top1_match, jaccard


# --------------------------------------------------------------------------- #
# Explanation backends                                                        #
# --------------------------------------------------------------------------- #
def _kernel_shap_top_k_batch(
    model: Any,
    X: np.ndarray,
    background: np.ndarray,
    feature_names: list[str],
    top_k: int,
    nsamples: int,
) -> list[list[str]]:
    """One KernelExplainer, one batched shap_values call, return top-K per row."""
    from shap import KernelExplainer

    predict = make_positive_class_predict(model)
    explainer = KernelExplainer(predict, background)
    vals = explainer.shap_values(X, nsamples=nsamples, silent=True)
    arr = np.asarray(vals, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [_top_k_reasons(arr[i], feature_names, top_k) for i in range(arr.shape[0])]


def _lime_top_k_batch(
    model: Any,
    X: np.ndarray,
    X_train_bg: np.ndarray,
    feature_names: list[str],
    top_k: int,
    num_samples: int,
    seed: int,
) -> list[list[str]]:
    """One LimeTabularExplainer, per-row explain_instance, return top-K per row."""
    from lime.lime_tabular import LimeTabularExplainer

    predict = make_predict_proba_two_column(model)
    explainer = LimeTabularExplainer(
        X_train_bg,
        feature_names=list(feature_names),
        class_names=["negative", "positive"],
        discretize_continuous=False,
        mode="classification",
        random_state=int(seed),
    )
    out: list[list[str]] = []
    for i in range(X.shape[0]):
        exp = explainer.explain_instance(
            X[i],
            predict,
            num_features=int(top_k),
            num_samples=int(num_samples),
            labels=(1,),
        )
        contrib = exp.as_map()[1]
        idx_by_abs = sorted(contrib, key=lambda kv: -abs(kv[1]))
        out.append([feature_names[int(fi)] for fi, _ in idx_by_abs[:top_k]])
    return out


def _exact_lattice_top_k_batch(
    model: Any,
    X: np.ndarray,
    X_reference: np.ndarray,
    feature_names: list[str],
    lattice_groups: list[list[str]],
    decision_threshold: float,
    top_k: int,
) -> list[list[str]]:
    """XCreditScore's own structure-derived attribution — one report per batch."""
    from .explainability import build_explainability_report

    df = build_explainability_report(
        model=model,
        X=X,
        feature_names=list(feature_names),
        X_reference=X_reference,
        lattice_groups=lattice_groups,
        top_k=int(top_k),
        decision_threshold=float(decision_threshold),
    )
    out: list[list[str]] = []
    for _, row in df.iterrows():
        raw = str(row.get("reason_codes", "[]"))
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                parsed = []
        except Exception:
            parsed = []
        out.append([str(v) for v in parsed[:top_k]])
    return out


# --------------------------------------------------------------------------- #
# Master benchmark loop                                                       #
# --------------------------------------------------------------------------- #
def run_stability_benchmark(
    *,
    predictor_model: Any,
    predictor_lattice_groups: list[list[str]],
    baseline_models: dict[str, Any],
    X_train_ref: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    decision_threshold: float,
    output_dir: Path,
    n_rows: int = 400,
    n_perturb: int = 8,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    top_k: int = 4,
    shap_nsamples: int = 100,
    shap_bg_size: int = 50,
    lime_num_samples: int = 500,
    seed: int = 42,
    include_exact_lattice: bool = True,
    include_shap: bool = True,
    include_lime: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run KernelSHAP + LIME + exact-lattice across every (model, sigma)."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rows = int(min(int(n_rows), int(X_test.shape[0])))
    n_perturb = int(max(1, n_perturb))
    top_k = int(max(1, top_k))
    X_eval = np.asarray(X_test[:n_rows], dtype=float)

    # Fix a background subset for KernelSHAP so the explainer is deterministic
    # per model; drawn once, reused across all sigmas / perturbations.
    rng_bg = np.random.default_rng(int(seed))
    if X_train_ref.shape[0] <= int(shap_bg_size):
        shap_background = np.asarray(X_train_ref, dtype=float)
    else:
        idx = rng_bg.choice(X_train_ref.shape[0], size=int(shap_bg_size), replace=False)
        shap_background = np.asarray(X_train_ref[idx], dtype=float)

    # LIME background uses a larger subset for its density estimate, capped so
    # explain_instance stays quick; deterministic per seed.
    lime_bg_size = min(int(X_train_ref.shape[0]), 1000)
    if X_train_ref.shape[0] <= lime_bg_size:
        lime_bg = np.asarray(X_train_ref, dtype=float)
    else:
        rng_lb = np.random.default_rng(int(seed) + 1)
        idx_l = rng_lb.choice(X_train_ref.shape[0], size=lime_bg_size, replace=False)
        lime_bg = np.asarray(X_train_ref[idx_l], dtype=float)

    all_models: dict[str, Any] = {"xcreditscore": predictor_model}
    for name, model in baseline_models.items():
        if model is not None:
            all_models[name] = model

    # Verify the SHAP wrapper against native predict for every model — the
    # task-required self-audit step. Any failure aborts the benchmark loudly.
    wrapper_audit: dict[str, dict[str, Any]] = {}
    for name, model in all_models.items():
        audit_X = X_eval[: min(10, X_eval.shape[0])]
        wrapper_audit[name] = verify_shap_wrapper(model, audit_X, tol=1e-6)
        if verbose:
            r = wrapper_audit[name]
            print(
                f"[stability-bench] SHAP wrapper audit {name}: "
                f"max_abs_diff={r['max_abs_diff']:.3e} passes={r['passes_tol']}"
            )
        if not wrapper_audit[name]["passes_tol"]:
            raise AssertionError(
                f"SHAP wrapper for {name!r} diverges from native predict_proba "
                f"by {wrapper_audit[name]['max_abs_diff']:.3e} (> 1e-6)."
            )

    # Cache base (unperturbed) top-K reasons once per (method, model); they do
    # not depend on sigma. Saves n_extra_sigmas * n_rows explanations per pair.
    if verbose:
        print(
            f"[stability-bench] computing base explanations "
            f"(n_rows={n_rows}, top_k={top_k}, methods="
            f"{'shap ' if include_shap else ''}{'lime ' if include_lime else ''}"
            f"{'exact_lattice ' if include_exact_lattice else ''})"
        )

    base_top_reasons: dict[tuple[str, str], list[list[str]]] = {}
    for name, model in all_models.items():
        if include_shap:
            t0 = time.time()
            base_top_reasons[(name, "kernel_shap")] = _kernel_shap_top_k_batch(
                model, X_eval, shap_background, feature_names, top_k, shap_nsamples
            )
            if verbose:
                print(f"  base kernel_shap  {name:<24} {time.time() - t0:6.1f}s")
        if include_lime:
            t0 = time.time()
            base_top_reasons[(name, "lime")] = _lime_top_k_batch(
                model, X_eval, lime_bg, feature_names, top_k, lime_num_samples, seed
            )
            if verbose:
                print(f"  base lime         {name:<24} {time.time() - t0:6.1f}s")
        if include_exact_lattice and name == "xcreditscore":
            t0 = time.time()
            base_top_reasons[(name, "exact_lattice_logit")] = _exact_lattice_top_k_batch(
                model, X_eval, X_train_ref, feature_names, predictor_lattice_groups,
                decision_threshold, top_k,
            )
            if verbose:
                print(f"  base exact_latt   {name:<24} {time.time() - t0:6.1f}s")

    # Now iterate the sigma sweep. For every sigma we re-seed the rng exactly
    # the way the existing stability code does — that keeps the sigma=0.01
    # xcreditscore.exact_lattice_logit row byte-comparable with the pre-existing
    # explainability_reason_stability_summary.json for the sanity check.
    long_rows: list[dict[str, Any]] = []

    for sigma in sigmas:
        t_sigma = time.time()
        rng = np.random.default_rng(int(seed))
        noises = [
            rng.normal(loc=0.0, scale=float(sigma), size=X_eval.shape)
            for _ in range(n_perturb)
        ]
        # Accumulate per-(method, model) counts across the n_perturb draws.
        counters: dict[tuple[str, str], dict[str, np.ndarray]] = {
            key: {
                "full_set_match": np.zeros(n_rows, dtype=float),
                "top1_match": np.zeros(n_rows, dtype=float),
                "jaccard": np.zeros(n_rows, dtype=float),
                "n_perturb": np.zeros(n_rows, dtype=float),
            }
            for key in base_top_reasons.keys()
        }

        for pk, noise in enumerate(noises):
            X_pert = np.clip(X_eval + noise, 0.0, 1.0)
            for name, model in all_models.items():
                for method, factory in (
                    ("kernel_shap", "shap"),
                    ("lime", "lime"),
                    ("exact_lattice_logit", "exact"),
                ):
                    key = (name, method)
                    if key not in base_top_reasons:
                        continue
                    t0 = time.time()
                    if method == "kernel_shap":
                        pert_top = _kernel_shap_top_k_batch(
                            model, X_pert, shap_background, feature_names,
                            top_k, shap_nsamples,
                        )
                    elif method == "lime":
                        pert_top = _lime_top_k_batch(
                            model, X_pert, lime_bg, feature_names,
                            top_k, lime_num_samples, seed + pk + 1,
                        )
                    else:  # exact_lattice_logit
                        pert_top = _exact_lattice_top_k_batch(
                            model, X_pert, X_train_ref, feature_names,
                            predictor_lattice_groups, decision_threshold, top_k,
                        )
                    base_top = base_top_reasons[key]
                    ctr = counters[key]
                    for i in range(n_rows):
                        full_match, top1_match, jac = _pair_scores(base_top[i], pert_top[i])
                        ctr["full_set_match"][i] += float(full_match)
                        ctr["top1_match"][i] += float(top1_match)
                        ctr["jaccard"][i] += jac
                        ctr["n_perturb"][i] += 1.0
                    if verbose:
                        print(
                            f"  sigma={sigma:<6}  perturb={pk+1}/{n_perturb}  "
                            f"{method:<20} {name:<24} {time.time() - t0:6.1f}s"
                        )

        # Aggregate counters into per-(method, model, sigma) rows.
        for (name, method), ctr in counters.items():
            n = np.maximum(ctr["n_perturb"], 1.0)
            row_full = float(np.mean(ctr["full_set_match"] / n))
            row_top1 = float(np.mean(ctr["top1_match"] / n))
            row_jac = float(np.mean(ctr["jaccard"] / n))
            long_rows.append(
                {
                    "method": method,
                    "model": name,
                    "sigma": float(sigma),
                    "n_rows": int(n_rows),
                    "n_perturb": int(n_perturb),
                    "full_set_match_rate": row_full,
                    "top1_match_rate": row_top1,
                    "mean_jaccard": row_jac,
                    "top_k": int(top_k),
                    "seed": int(seed),
                }
            )
        if verbose:
            print(f"  sigma={sigma}: total {time.time() - t_sigma:.1f}s")

    long_df = pd.DataFrame(long_rows)
    long_csv = output_dir / "explanation_stability_benchmark_long.csv"
    long_df.to_csv(long_csv, index=False)

    # ---- Master comparison table: rows = method_model, cols = sigma --------
    long_df = long_df.copy()
    long_df["method_model"] = long_df["method"] + " / " + long_df["model"]
    match_wide = long_df.pivot(index="method_model", columns="sigma", values="full_set_match_rate")
    match_wide.columns = [f"sigma_{s:g}" for s in match_wide.columns]
    match_wide = match_wide.reset_index()
    master_csv = output_dir / "explanation_stability_benchmark_master.csv"
    match_wide.to_csv(master_csv, index=False)

    top1_wide = long_df.pivot(index="method_model", columns="sigma", values="top1_match_rate")
    top1_wide.columns = [f"sigma_{s:g}" for s in top1_wide.columns]
    top1_wide = top1_wide.reset_index()
    top1_wide.to_csv(
        output_dir / "explanation_stability_benchmark_top1_master.csv", index=False
    )

    jac_wide = long_df.pivot(index="method_model", columns="sigma", values="mean_jaccard")
    jac_wide.columns = [f"sigma_{s:g}" for s in jac_wide.columns]
    jac_wide = jac_wide.reset_index()
    jac_wide.to_csv(
        output_dir / "explanation_stability_benchmark_jaccard_master.csv", index=False
    )

    # ---- Plot: match rate vs sigma per method-model line ------------------
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6.5))
        colors = plt.get_cmap("tab20").colors
        for i, (name, group) in enumerate(long_df.groupby("method_model")):
            g = group.sort_values("sigma")
            ax.plot(
                g["sigma"],
                g["full_set_match_rate"],
                marker="o",
                linewidth=1.5,
                label=name,
                color=colors[i % len(colors)],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Perturbation σ (on min-max-scaled features)")
        ax.set_ylabel("Top-K reason-set exact match rate")
        ax.set_title(
            "Explanation stability vs perturbation magnitude\n"
            f"({n_rows} test instances × {n_perturb} perturbations, top-K={top_k})"
        )
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        plt.tight_layout()
        plot_path = output_dir / "explanation_stability_benchmark.png"
        plt.savefig(plot_path, dpi=400)
        plt.close(fig)
    except Exception as exc:
        plot_path = None
        if verbose:
            print(f"[stability-bench] plot skipped: {type(exc).__name__}: {exc}")

    summary = {
        "n_rows": int(n_rows),
        "n_perturb": int(n_perturb),
        "sigmas": [float(s) for s in sigmas],
        "top_k": int(top_k),
        "shap_nsamples": int(shap_nsamples),
        "shap_bg_size": int(shap_background.shape[0]),
        "lime_num_samples": int(lime_num_samples),
        "seed": int(seed),
        "models_evaluated": sorted(list(all_models.keys())),
        "methods_evaluated": [
            m for m in ("kernel_shap", "lime", "exact_lattice_logit")
            if any((name, m) in base_top_reasons for name in all_models)
        ],
        "wrapper_audit": wrapper_audit,
        "artifacts": {
            "long_csv": str(long_csv),
            "match_master_csv": str(master_csv),
            "top1_master_csv": str(output_dir / "explanation_stability_benchmark_top1_master.csv"),
            "jaccard_master_csv": str(output_dir / "explanation_stability_benchmark_jaccard_master.csv"),
            "plot_png": str(plot_path) if plot_path is not None else None,
        },
    }
    with open(
        output_dir / "explanation_stability_benchmark_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Env-var driven entrypoint for use from src.train_all                        #
# --------------------------------------------------------------------------- #
def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _sigmas_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        vals = [float(x) for x in raw.split(",") if x.strip()]
        if not vals:
            return default
        return tuple(vals)
    except ValueError:
        return default


def maybe_run_from_train_all(
    *,
    predictor_model: Any,
    predictor_lattice_groups: list[list[str]],
    baseline_models: dict[str, Any],
    X_train_ref: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    decision_threshold: float,
    output_dir: Path,
) -> Optional[dict[str, Any]]:
    """Called from train_all.main() when the benchmark is not disabled."""
    if _bool_env("SKIP_EXPLAIN_STABILITY_BENCHMARK", default=False):
        print("[stability-bench] skipped (SKIP_EXPLAIN_STABILITY_BENCHMARK=1)")
        return None
    n_rows = _int_env("EXPLAIN_STABILITY_BENCHMARK_N_ROWS", 400)
    n_perturb = _int_env("EXPLAIN_STABILITY_BENCHMARK_N_PERTURB", 8)
    sigmas = _sigmas_env("EXPLAIN_STABILITY_BENCHMARK_SIGMAS", DEFAULT_SIGMAS)
    shap_nsamples = _int_env("EXPLAIN_STABILITY_BENCHMARK_SHAP_NSAMPLES", 100)
    shap_bg_size = _int_env("EXPLAIN_STABILITY_BENCHMARK_SHAP_BG", 50)
    lime_num_samples = _int_env("EXPLAIN_STABILITY_BENCHMARK_LIME_NSAMPLES", 500)
    top_k = _int_env("EXPLAIN_STABILITY_BENCHMARK_TOPK", 4)
    seed = _int_env("EXPLAIN_STABILITY_BENCHMARK_SEED", 42)
    return run_stability_benchmark(
        predictor_model=predictor_model,
        predictor_lattice_groups=predictor_lattice_groups,
        baseline_models=baseline_models,
        X_train_ref=X_train_ref,
        X_test=X_test,
        feature_names=feature_names,
        decision_threshold=decision_threshold,
        output_dir=output_dir,
        n_rows=n_rows,
        n_perturb=n_perturb,
        sigmas=sigmas,
        top_k=top_k,
        shap_nsamples=shap_nsamples,
        shap_bg_size=shap_bg_size,
        lime_num_samples=lime_num_samples,
        seed=seed,
        include_exact_lattice=True,
        include_shap=True,
        include_lime=True,
        verbose=True,
    )
