"""Group-fairness evaluation on the Taiwan benchmark.

Taiwan (Yeh and Lien, 2009) is the one benchmark of the three that ships
protected demographic attributes: SEX, MARRIAGE, EDUCATION and AGE. The
constraint registry already declares all four immutable, so recourse never
proposes changing them, but immutability alone says nothing about whether
the scores or the offered recourse fall differently across groups. This
module measures that.

Two families of metric are reported:

  * Prediction fairness at the deployed operating threshold: approval rate,
    the four-fifths disparate-impact ratio, true/false positive rates
    (equal opportunity and equalised odds), and group AUC.
  * Recourse fairness: the share of denied applicants in each group for whom
    the mixed-integer engine returns a feasible, registry-respecting plan.
    This is read from the head-to-head artifacts, so it covers both the
    lattice and the monotone-XGBoost predictor on identical denied cases.

Group membership always comes from the raw, pre-imputation values. Taiwan
codes EDUCATION 0/5/6 and MARRIAGE 0 as unknown/other; those rows are
median-imputed for modelling but must not be silently folded into a named
group for a fairness claim, so they are reported as a separate "unknown"
row and excluded from the disparity ratios.

Every rate carries a Wilson 95% interval, matching the protocol used for
recourse feasibility elsewhere in the pipeline.

Usage:
    DATASET_NAME=taiwan .venv/bin/python -m src.fairness_evaluation
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .recourse_full_evaluation import wilson_score_ci

ROOT = Path(__file__).resolve().parents[1]

# Taiwan's documented unknown/other codes. Kept out of the disparity ratios.
UNKNOWN_CODES = {"EDUCATION": {0, 5, 6}, "MARRIAGE": {0}}
SEX_LABELS = {1: "male", 2: "female"}
EDUCATION_LABELS = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
MARRIAGE_LABELS = {1: "married", 2: "single", 3: "other"}
AGE_BANDS = [(21, 30, "21-30"), (31, 40, "31-40"), (41, 50, "41-50"), (51, 80, "51+")]

FOUR_FIFTHS = 0.80

# Disparity ratios are max/min statistics, so a single tiny cell can dominate
# them. Groups below this size are still reported in full, with their Wilson
# interval, but are excluded from the summary ratios and flagged. This is the
# same n < 30 rule the recourse stratification uses.
MIN_GROUP_N = 30


def two_proportion_test(s1: int, n1: int, s2: int, n2: int) -> dict[str, Any]:
    """Two-sided two-proportion z-test with a pooled variance estimate.

    A disparity ratio computed from two rates says nothing about whether the
    two rates are distinguishable at the sample sizes involved. Reporting a
    ratio without this test invites reading noise as a finding.
    """
    if not n1 or not n2:
        return {"z": float("nan"), "p_value": float("nan"), "note": "empty group"}
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0, "note": "zero variance"}
    z = (p1 - p2) / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"z": float(z), "p_value": float(p),
            "rate_1": p1, "rate_2": p2, "n_1": int(n1), "n_2": int(n2)}


def _test_partition_index(df: pd.DataFrame) -> pd.Index:
    """Re-derive the test partition's original row index.

    Deterministic: same frame, same seed, same stratification as
    ``preprocess_and_split``, so this selects exactly the rows the reported
    metrics were computed on. Asserted against the pipeline bundle by the
    caller.
    """
    from .config import FEATURE_COLS, RANDOM_STATE, SENTINEL_VALUES, TARGET_COL, prepare_target

    cleaned = df.copy()
    cleaned[TARGET_COL] = prepare_target(cleaned[TARGET_COL])
    X = cleaned[FEATURE_COLS].replace(SENTINEL_VALUES, np.nan)
    y = cleaned[TARGET_COL].to_numpy(dtype=int)
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    _, X_test, _, _ = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)
    return X_test.index


def _groups(raw: pd.DataFrame, idx: pd.Index) -> dict[str, pd.Series]:
    """Named group labels per protected attribute, over the test partition."""
    sub = raw.loc[idx]
    out: dict[str, pd.Series] = {}

    out["SEX"] = sub["SEX"].map(lambda v: SEX_LABELS.get(int(v), "unknown"))

    def _coded(col: str, labels: dict[int, str]) -> pd.Series:
        unknown = UNKNOWN_CODES.get(col, set())
        return sub[col].map(
            lambda v: "unknown" if int(v) in unknown else labels.get(int(v), "unknown"))

    out["EDUCATION"] = _coded("EDUCATION", EDUCATION_LABELS)
    out["MARRIAGE"] = _coded("MARRIAGE", MARRIAGE_LABELS)

    def _age(v: float) -> str:
        for lo, hi, lab in AGE_BANDS:
            if lo <= v <= hi:
                return lab
        return "unknown"

    out["AGE"] = sub["AGE"].map(_age)
    return out


def _rate(successes: int, trials: int) -> dict[str, Any]:
    ci = wilson_score_ci(successes, trials) if trials else {
        "rate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    return {"n_successes": int(successes), "n_trials": int(trials),
            "rate": ci["rate"], "ci95_low": ci["ci95_low"], "ci95_high": ci["ci95_high"]}


def group_prediction_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                             labels: pd.Series, tau: float) -> dict[str, Any]:
    """Approval rate, TPR, FPR and AUC per group, each with a Wilson interval.

    The favourable outcome in lending is approval, so the disparate-impact
    ratio is computed on approval rates, not denial rates.
    """
    y_true = np.asarray(y_true, dtype=int)
    denied = np.asarray(y_proba, dtype=float) >= tau
    lab = np.asarray(labels)

    rows: dict[str, Any] = {}
    for g in sorted(set(lab)):
        m = lab == g
        yt, dn = y_true[m], denied[m]
        pos, neg = yt == 1, yt == 0
        row = {
            "n": int(m.sum()),
            "base_default_rate": _rate(int(pos.sum()), int(m.sum())),
            "approval_rate": _rate(int((~dn).sum()), int(m.sum())),
            # among true defaulters, how many are correctly denied
            "tpr": _rate(int((dn & pos).sum()), int(pos.sum())),
            # among true non-defaulters, how many are wrongly denied
            "fpr": _rate(int((dn & neg).sum()), int(neg.sum())),
        }
        row["auc"] = (float(roc_auc_score(yt, np.asarray(y_proba)[m]))
                      if len(set(yt)) == 2 else None)
        rows[g] = row

    named = {g: r for g, r in rows.items()
             if g != "unknown" and r["n"] >= MIN_GROUP_N}
    small = {g: r["n"] for g, r in rows.items()
             if g != "unknown" and r["n"] < MIN_GROUP_N}
    summary: dict[str, Any] = {
        "groups": rows,
        "n_unknown_excluded": rows.get("unknown", {}).get("n", 0),
        "small_groups_excluded_from_ratios": small,
        "min_group_n": MIN_GROUP_N,
    }
    if len(named) >= 2:
        appr = {g: r["approval_rate"]["rate"] for g, r in named.items()}
        tpr = {g: r["tpr"]["rate"] for g, r in named.items() if r["tpr"]["n_trials"]}
        fpr = {g: r["fpr"]["rate"] for g, r in named.items() if r["fpr"]["n_trials"]}
        lo_g, hi_g = min(appr, key=appr.get), max(appr, key=appr.get)
        summary["disparate_impact_ratio"] = (appr[lo_g] / appr[hi_g]
                                             if appr[hi_g] else float("nan"))
        summary["disparate_impact_pair"] = {"lowest": lo_g, "highest": hi_g}
        summary["passes_four_fifths"] = bool(
            summary["disparate_impact_ratio"] >= FOUR_FIFTHS)
        summary["demographic_parity_difference"] = appr[hi_g] - appr[lo_g]
        if tpr:
            summary["equal_opportunity_difference"] = max(tpr.values()) - min(tpr.values())
        if tpr and fpr:
            summary["equalised_odds_difference"] = max(
                max(tpr.values()) - min(tpr.values()),
                max(fpr.values()) - min(fpr.values()))
        if len(named) == 2:
            (ga, ra), (gb, rb) = list(named.items())
            summary["approval_rate_test"] = {
                "groups": [ga, gb],
                **two_proportion_test(ra["approval_rate"]["n_successes"],
                                      ra["approval_rate"]["n_trials"],
                                      rb["approval_rate"]["n_successes"],
                                      rb["approval_rate"]["n_trials"]),
            }
    return summary


def group_recourse_metrics(feasible_by_index: dict[int, bool],
                           labels: pd.Series) -> dict[str, Any]:
    """Recourse feasibility per group among denied applicants."""
    lab = np.asarray(labels)
    rows: dict[str, Any] = {}
    for g in sorted(set(lab)):
        pos = np.where(lab == g)[0]
        vals = [feasible_by_index[i] for i in pos if i in feasible_by_index]
        if not vals:
            continue
        rows[g] = _rate(int(sum(vals)), len(vals))
    named = {g: r for g, r in rows.items()
             if g != "unknown" and r["n_trials"] >= MIN_GROUP_N}
    small = {g: r["n_trials"] for g, r in rows.items()
             if g != "unknown" and r["n_trials"] < MIN_GROUP_N}
    out: dict[str, Any] = {"groups": rows,
                           "small_groups_excluded_from_ratios": small,
                           "min_group_n": MIN_GROUP_N}
    if len(named) >= 2:
        rates = {g: r["rate"] for g, r in named.items()}
        lo_g, hi_g = min(rates, key=rates.get), max(rates, key=rates.get)
        out["feasibility_gap"] = rates[hi_g] - rates[lo_g]
        out["feasibility_ratio"] = (rates[lo_g] / rates[hi_g]
                                    if rates[hi_g] else float("nan"))
        out["lowest_group"], out["highest_group"] = lo_g, hi_g
        if len(named) == 2:
            (ga, ra), (gb, rb) = list(named.items())
            out["feasibility_test"] = {
                "groups": [ga, gb],
                **two_proportion_test(ra["n_successes"], ra["n_trials"],
                                      rb["n_successes"], rb["n_trials"]),
            }
    return out


def run_fairness_evaluation(bundle, y_proba, tau: float, out_dir: Path,
                            data_path: Path | None = None) -> dict[str, Any]:
    """Full fairness report for Taiwan; returns the summary it writes."""
    from .config import DATA_PATH

    raw = pd.read_csv(data_path or DATA_PATH)
    from .data_pipeline import load_raw_data

    idx = _test_partition_index(load_raw_data())
    if len(idx) != len(bundle.y_test):
        raise RuntimeError(
            f"test partition mismatch: re-derived {len(idx)} rows, "
            f"bundle has {len(bundle.y_test)}")

    proba = np.asarray(y_proba, dtype=float).ravel()
    if len(proba) != len(bundle.y_test):
        raise RuntimeError(
            f"score/label length mismatch: {len(proba)} vs {len(bundle.y_test)}")

    groups = _groups(raw, idx)
    report: dict[str, Any] = {
        "dataset": "taiwan",
        "tau": float(tau),
        "n_test": int(len(idx)),
        "four_fifths_threshold": FOUR_FIFTHS,
        "group_membership_source": "raw pre-imputation values",
        "prediction_fairness": {},
        "recourse_fairness": {},
    }
    for attr, lab in groups.items():
        report["prediction_fairness"][attr] = group_prediction_metrics(
            bundle.y_test, proba, lab, tau)

    # ---- recourse fairness from the head-to-head artifacts ---------------
    h2h = out_dir.parent / "counterfactuals" / "counterfactual_head_to_head.csv"
    if h2h.exists():
        cases = pd.read_csv(h2h)
        for label, name in (("xcreditscore_lattice", "lattice"),
                            ("monotone_xgboost", "monotone_xgboost")):
            sub = cases[cases["predictor_label"] == label]
            if sub.empty:
                continue
            fmap = {int(r["test_index"]): bool(r["feasible"])
                    for _, r in sub.iterrows()}
            report["recourse_fairness"][name] = {
                attr: group_recourse_metrics(fmap, lab)
                for attr, lab in groups.items()
            }
        report["recourse_fairness_note"] = (
            "feasibility among denied applicants only; group labels from raw "
            "pre-imputation values on the same test partition")

    out_dir.mkdir(parents=True, exist_ok=True)
    # json.dumps writes bare NaN, which is not valid JSON and breaks strict
    # parsers. Empty cells become null instead.
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        return o

    (out_dir / "fairness_taiwan_summary.json").write_text(
        json.dumps(_clean(report), indent=2, allow_nan=False), encoding="utf-8")

    flat = []
    for attr, blk in report["prediction_fairness"].items():
        for g, r in blk["groups"].items():
            flat.append({
                "attribute": attr, "group": g, "n": r["n"],
                "base_default_rate": r["base_default_rate"]["rate"],
                "approval_rate": r["approval_rate"]["rate"],
                "approval_ci95_low": r["approval_rate"]["ci95_low"],
                "approval_ci95_high": r["approval_rate"]["ci95_high"],
                "tpr": r["tpr"]["rate"], "fpr": r["fpr"]["rate"], "auc": r["auc"],
            })
    pd.DataFrame(flat).to_csv(out_dir / "fairness_taiwan_by_group.csv", index=False)
    return report


def main() -> None:
    import os

    os.environ.setdefault("DATASET_NAME", "taiwan")
    from .config import PACK_DIR
    from .data_pipeline import load_raw_data, preprocess_and_split

    art = Path(PACK_DIR)
    tau = json.loads((art / "metrics" / "run_summary.json").read_text())["threshold"]
    bundle, _ = preprocess_and_split(load_raw_data())

    # The deployed model's test-partition scores are already an artifact of
    # the canonical run; reading them avoids any chance of re-scoring drift.
    preds = pd.read_csv(art / "predictions" / "xcreditscore_test_predictions.csv")
    if not (preds["y_true"].to_numpy() == bundle.y_test).all():
        raise RuntimeError("saved predictions do not align with the test partition")
    rep = run_fairness_evaluation(
        bundle, preds["y_proba_bad"].to_numpy(), tau, art / "metrics")

    print(f"tau = {tau}, n_test = {rep['n_test']}")
    for attr, blk in rep["prediction_fairness"].items():
        di = blk.get("disparate_impact_ratio")
        if di is None:
            continue
        print(f"\n{attr}: DI ratio {di:.4f} "
              f"({'PASSES' if blk['passes_four_fifths'] else 'FAILS'} four-fifths), "
              f"lowest={blk['disparate_impact_pair']['lowest']}")
        for g, r in blk["groups"].items():
            print(f"   {g:16s} n={r['n']:>5d} approval={r['approval_rate']['rate']:.4f} "
                  f"[{r['approval_rate']['ci95_low']:.4f}, "
                  f"{r['approval_rate']['ci95_high']:.4f}]  "
                  f"TPR={r['tpr']['rate']:.4f}  FPR={r['fpr']['rate']:.4f}")


if __name__ == "__main__":
    main()
