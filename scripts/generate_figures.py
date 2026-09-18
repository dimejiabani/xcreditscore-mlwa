from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "artifacts" / "heloc_pack" / "metrics"
SRC_FIGS = ROOT / "artifacts" / "heloc_pack" / "figures"
OUT = ROOT / "artifacts" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(x: str, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def setup_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def copy_existing_core_curves() -> list[str]:
    copied = []
    for name in [
        "xcreditscore_roc_curve.png",
        "xcreditscore_pr_curve.png",
        "xcreditscore_reliability_curve.png",
    ]:
        src = SRC_FIGS / name
        if src.exists():
            dst = OUT / name
            copy2(src, dst)
            copied.append(dst.name)
    return copied


def plot_competitive_metrics() -> str:
    """Grouped bars: XCreditScore against all six baselines on HELOC.

    final_comparison_pack.csv keys the baselines as ``baseline:<name>`` and
    XCreditScore as a bare ``xcreditscore``, so the section lookup has to carry
    the prefix. It also records only auc, pr_auc, accuracy and f1 for the
    baselines, so those four metrics are the ones every model can be compared
    on; precision and recall are XCreditScore-only in this artifact and are
    reported per model in the Section 4.1 tables instead.
    """
    rows = read_csv(METRICS / "final_comparison_pack.csv")
    target = ["auc", "pr_auc", "accuracy", "f1"]
    # (section key in the artifact, label on the chart)
    models = [
        ("xcreditscore", "XCreditScore"),
        ("baseline:logistic_regression", "Logistic regression"),
        ("baseline:random_forest", "Random forest"),
        ("baseline:xgboost", "XGBoost"),
        ("baseline:monotone_xgboost", "Monotone XGBoost"),
        ("baseline:ebm", "EBM"),
        ("baseline:monotone_gam", "Monotone GAM"),
    ]
    data: dict[str, dict[str, float]] = {sec: {} for sec, _ in models}
    for r in rows:
        sec = r.get("section", "")
        name = r.get("name", "")
        if sec in data and name in target:
            data[sec][name] = f(r.get("value", "nan"))

    # A silently empty series is what produced a one-model "grouped" chart
    # before; refuse to emit the figure rather than mislabel it again.
    missing = [
        f"{sec}/{t}"
        for sec, _ in models
        for t in target
        if not math.isfinite(data[sec].get(t, math.nan))
    ]
    if missing:
        raise SystemExit(
            "model_comparison_bar: missing values for " + ", ".join(missing)
        )

    fig, ax = plt.subplots(figsize=(13, 6))
    x = range(len(target))
    n = len(models)
    width = 0.8 / n
    offset = (n - 1) / 2
    for i, (sec, label) in enumerate(models):
        vals = [data[sec][t] for t in target]
        ax.bar([k + (i - offset) * width for k in x], vals, width=width,
               label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["AUC", "PR-AUC", "Accuracy", "F1"])
    ax.set_ylim(0.65, 0.83)
    ax.set_title("Model comparison across core metrics (HELOC test partition)")
    ax.set_ylabel("Score")
    ax.legend(ncol=4, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    save(fig, "model_comparison_bar.png")
    return "model_comparison_bar.png"


def plot_cv_auc_distribution() -> str:
    rows = read_csv(METRICS / "cv_model_metrics.csv")
    models = ["xcreditscore", "logistic_regression", "random_forest", "xgboost"]
    values = {m: [] for m in models}
    for r in rows:
        m = r.get("model", "")
        if m in values:
            values[m].append(f(r.get("auc", "nan")))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([values[m] for m in models], tick_labels=models, showmeans=True)
    ax.set_title("Cross-Validation AUC Distribution by Model")
    ax.set_ylabel("AUC")
    save(fig, "cv_auc_boxplot.png")
    return "cv_auc_boxplot.png"


def plot_threshold_frontier() -> str:
    rows = read_csv(METRICS / "threshold_frontier.csv")
    th = [f(r["threshold"]) for r in rows]
    f1 = [f(r["f1"]) for r in rows]
    prec = [f(r["precision"]) for r in rows]
    rec = [f(r["recall"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(th, f1, label="F1")
    ax.plot(th, prec, label="Precision")
    ax.plot(th, rec, label="Recall")
    ax.set_title("Threshold Frontier: Precision, Recall, and F1")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "threshold_frontier.png")
    return "threshold_frontier.png"


def plot_threshold_objective_tradeoff() -> str:
    rows = read_csv(METRICS / "threshold_objective_comparison.csv")
    objectives = [r["objective"] for r in rows]
    f1s = [f(r["f1"]) for r in rows]
    prec = [f(r["precision"]) for r in rows]
    rec = [f(r["recall"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(objectives))
    ax.bar([i - 0.25 for i in x], prec, width=0.25, label="Precision")
    ax.bar(x, rec, width=0.25, label="Recall")
    ax.bar([i + 0.25 for i in x], f1s, width=0.25, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(objectives)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Objective-Specific Operating Trade-offs")
    ax.legend()
    save(fig, "objective_tradeoff_bar.png")
    return "objective_tradeoff_bar.png"


def plot_calibration_methods() -> str:
    rows = read_csv(METRICS / "calibration_method_comparison.csv")
    methods = [r["method"] for r in rows]
    ece = [f(r["ece"]) for r in rows]
    mce = [f(r["mce"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(methods))
    ax.bar([i - 0.2 for i in x], ece, width=0.4, label="ECE")
    ax.bar([i + 0.2 for i in x], mce, width=0.4, label="MCE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_title("Calibration Error by Method")
    ax.set_ylabel("Error")
    ax.legend()
    save(fig, "calibration_error_bar.png")
    return "calibration_error_bar.png"


def plot_explainability_stability() -> str:
    rows = read_csv(METRICS / "explainability_reason_stability.csv")
    idx = list(range(len(rows)))
    exact = [f(r["exact_match_rate"]) for r in rows]
    jac = [f(r["jaccard_mean"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(idx, exact, s=10, alpha=0.6, label="Exact-match rate")
    ax.scatter(idx, jac, s=10, alpha=0.6, label="Jaccard mean")
    ax.axhline(sum(exact) / max(1, len(exact)), color="C0", linestyle="--", linewidth=1)
    ax.axhline(sum(jac) / max(1, len(jac)), color="C1", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Local Explanation Stability Across Evaluated Cases")
    ax.set_xlabel("Evaluated case index")
    ax.set_ylabel("Stability score")
    ax.legend()
    save(fig, "explainability_stability_scatter.png")
    return "explainability_stability_scatter.png"


def plot_global_contributions() -> str:
    rows = read_csv(METRICS / "explainability_global_contributions.csv")
    rows = sorted(rows, key=lambda r: f(r["mean_abs_effect"]), reverse=True)[:12]
    feats = [r["feature"] for r in rows][::-1]
    vals = [f(r["mean_abs_effect"]) for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(feats, vals)
    ax.set_title("Top Global Feature Contributions (Mean Absolute Effect)")
    ax.set_xlabel("Mean absolute effect")
    save(fig, "global_feature_contributions.png")
    return "global_feature_contributions.png"


def plot_counterfactual_success_band() -> str:
    rows = read_csv(METRICS / "counterfactual_success_by_score_band.csv")
    bands = [r["original_score_band"] for r in rows]
    feasible = [f(r["feasible_rate"]) for r in rows]
    near = [f(r["near_feasible_rate"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(bands))
    ax.bar([i - 0.2 for i in x], feasible, width=0.4, label="Feasible")
    ax.bar([i + 0.2 for i in x], near, width=0.4, label="Near-feasible")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bands, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Recourse Success by Original Score Band")
    ax.legend()
    save(fig, "counterfactual_success_by_band.png")
    return "counterfactual_success_by_band.png"


def plot_counterfactual_top_changes() -> str:
    rows = read_csv(METRICS / "counterfactual_top_changed_features.csv")
    rows = sorted(rows, key=lambda r: f(r["count"]), reverse=True)[:12]
    feats = [r["feature"] for r in rows][::-1]
    counts = [f(r["count"]) for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(feats, counts)
    ax.set_title("Most Frequently Changed Features in Recourse")
    ax.set_xlabel("Count")
    save(fig, "counterfactual_top_changes.png")
    return "counterfactual_top_changes.png"


def plot_statistical_ci() -> str:
    rows = read_csv(METRICS / "statistical_comparison.csv")
    metrics = [r["metric"] for r in rows]
    mean = [f(r["mean_delta_xcredit_minus_baseline"]) for r in rows]
    lo = [f(r["delta_ci95_low"]) for r in rows]
    hi = [f(r["delta_ci95_high"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = list(range(len(metrics)))
    for i in range(len(metrics)):
        ax.plot([lo[i], hi[i]], [i, i], color="black", linewidth=1.5)
        ax.scatter(mean[i], i, color="C0", s=35)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_title("Paired CV Mean Delta (XCreditScore - Best Baseline) with 95% CI")
    ax.set_xlabel("Delta")
    save(fig, "statistical_delta_ci.png")
    return "statistical_delta_ci.png"


def plot_monotonic_violations() -> str:
    rows = read_csv(METRICS / "monotonic_sanity_report.csv")
    rows = rows[:15]
    feats = [r["feature"] for r in rows][::-1]
    rates = [f(r["violation_rate"]) for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(feats, rates)
    ax.set_title("Monotonic Violation Rate by Checked Feature")
    ax.set_xlabel("Violation rate")
    save(fig, "monotonic_violation_rate.png")
    return "monotonic_violation_rate.png"


def plot_onnx_parity_error() -> str:
    path = METRICS / "onnx_parity_smoke.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    names = ["mean_abs_err", "max_abs_err", "tolerance"]
    vals = [float(obj.get("mean_abs_err", 0.0)), float(obj.get("max_abs_err", 0.0)), float(obj.get("tolerance", 0.0))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, vals)
    ax.set_yscale("log")
    ax.set_title("ONNX Parity Error vs Tolerance (Log Scale)")
    ax.set_ylabel("Value (log scale)")
    save(fig, "onnx_parity_error.png")
    return "onnx_parity_error.png"


def draw_flow_diagram(filename: str, title: str, nodes: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(12, 3.3))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    x_positions = [0.08 + i * (0.84 / max(1, len(nodes) - 1)) for i in range(len(nodes))]
    y = 0.5
    for i, (x, label) in enumerate(zip(x_positions, nodes)):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eaf2ff", edgecolor="#345"),
            transform=ax.transAxes,
        )
        if i < len(nodes) - 1:
            nx = x_positions[i + 1]
            ax.annotate(
                "",
                xy=(nx - 0.05, y),
                xytext=(x + 0.05, y),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.3),
            )

    save(fig, filename)
    return filename


def write_index(generated: list[str], copied: list[str]) -> None:
    lines = [
        "# Figure Index",
        "",
        "Generated from canonical artifacts in artifacts/heloc_pack.",
        "",
        "## Core Existing Curves (copied)",
    ]
    for name in copied:
        lines.append(f"- {name}")

    lines.append("")
    lines.append("## Generated Result Figures")
    for name in generated:
        lines.append(f"- {name}")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Chapter 5 should include ROC/PR/calibration/threshold/benchmarking/recourse/stability plots.")
    lines.append("- Chapter 4 architecture diagrams are represented here as generated workflow figures and can be replaced with design-tool renderings if desired.")

    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_csv(name: str, headers: list[str], rows: list[list[object]]) -> str:
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return name


def generate_tables() -> list[str]:
    tables: list[str] = []

    # Table 5.1: Model competitiveness.
    comp = json.loads((METRICS / "model_competitiveness.json").read_text(encoding="utf-8"))
    tables.append(
        write_table_csv(
            "table_1_model_competitiveness.csv",
            ["metric", "xcreditscore", "best_baseline_model", "best_baseline_value", "delta_xcredit_minus_best_baseline"],
            [
                ["auc", comp["xcreditscore"]["auc"], comp["best_baseline"]["model"], comp["best_baseline"]["auc"], comp["delta_vs_best_baseline"]],
                ["pr_auc", comp["xcreditscore"]["pr_auc"], comp["best_baseline"]["model"], comp["best_baseline"]["pr_auc"], comp["xcreditscore"]["pr_auc"] - comp["best_baseline"]["pr_auc"]],
                ["f1", comp["xcreditscore"]["f1"], comp["best_baseline"]["model"], comp["best_baseline"]["f1"], comp["xcreditscore"]["f1"] - comp["best_baseline"]["f1"]],
                ["accuracy", comp["xcreditscore"]["accuracy"], comp["best_baseline"]["model"], comp["best_baseline"]["accuracy"], comp["xcreditscore"]["accuracy"] - comp["best_baseline"]["accuracy"]],
            ],
        )
    )

    # Table 5.2: Statistical paired comparison.
    stat_rows = read_csv(METRICS / "statistical_comparison.csv")
    tables.append(
        write_table_csv(
            "table_2_statistical_comparison.csv",
            ["metric", "mean_delta", "ci95_low", "ci95_high", "p_value", "xcredit_better_fold_share"],
            [
                [
                    r["metric"],
                    r["mean_delta_xcredit_minus_baseline"],
                    r["delta_ci95_low"],
                    r["delta_ci95_high"],
                    r["paired_sign_flip_pvalue"],
                    r["xcredit_better_on_more_folds"],
                ]
                for r in stat_rows
            ],
        )
    )

    # Table 5.3: Threshold objective outcomes.
    tobj = read_csv(METRICS / "threshold_objective_comparison.csv")
    tables.append(
        write_table_csv(
            "table_3_threshold_objectives.csv",
            ["objective", "threshold", "auc", "pr_auc", "accuracy", "precision", "recall", "f1", "denied", "cf_feasible"],
            [
                [
                    r["objective"],
                    r["selected_threshold"],
                    r["auc"],
                    r["pr_auc"],
                    r["accuracy"],
                    r["precision"],
                    r["recall"],
                    r["f1"],
                    r["denied"],
                    r["cf_feasible"],
                ]
                for r in tobj
            ],
        )
    )

    # Table 5.4: Calibration methods.
    cal = read_csv(METRICS / "calibration_method_comparison.csv")
    tables.append(
        write_table_csv(
            "table_4_calibration_methods.csv",
            ["method", "auc", "pr_auc", "brier", "log_loss", "ece", "mce"],
            [[r["method"], r["auc"], r["pr_auc"], r["brier"], r["log_loss"], r["ece"], r["mce"]] for r in cal],
        )
    )

    # Table 5.5: Counterfactual summary.
    cf = json.loads((METRICS / "counterfactual_summary.json").read_text(encoding="utf-8"))
    tables.append(
        write_table_csv(
            "table_5_counterfactual_summary.csv",
            ["cases", "feasible_rate", "median_total_cost", "median_changed_features"],
            [[cf.get("cases"), cf.get("feasible_rate"), cf.get("median_total_cost"), cf.get("median_changed_features")]],
        )
    )

    # Table 5.6: Explainability stability summary.
    ex = json.loads((METRICS / "explainability_reason_stability_summary.json").read_text(encoding="utf-8"))
    tables.append(
        write_table_csv(
            "table_6_explainability_stability_summary.csv",
            ["rows_evaluated", "perturbations", "sigma", "overall_exact_match_rate", "overall_jaccard_mean"],
            [[ex.get("rows_evaluated"), ex.get("perturbations"), ex.get("sigma"), ex.get("overall_exact_match_rate"), ex.get("overall_jaccard_mean")]],
        )
    )

    # Table 5.7: ONNX parity summary.
    onnx = json.loads((METRICS / "onnx_parity_smoke.json").read_text(encoding="utf-8"))
    tables.append(
        write_table_csv(
            "table_7_onnx_parity_summary.csv",
            ["status", "ok", "rows", "tolerance", "max_abs_err", "mean_abs_err"],
            [[onnx.get("status"), onnx.get("ok"), onnx.get("rows"), onnx.get("tolerance"), onnx.get("max_abs_err"), onnx.get("mean_abs_err")]],
        )
    )

    return tables


def main() -> None:
    setup_out()
    copied = copy_existing_core_curves()

    generated = [
        plot_competitive_metrics(),
        plot_cv_auc_distribution(),
        plot_threshold_frontier(),
        plot_threshold_objective_tradeoff(),
        plot_calibration_methods(),
        plot_explainability_stability(),
        plot_global_contributions(),
        plot_counterfactual_success_band(),
        plot_counterfactual_top_changes(),
        plot_statistical_ci(),
        plot_monotonic_violations(),
        plot_onnx_parity_error(),
        draw_flow_diagram(
            "system_implementation_flow.png",
            "Chapter 4 System Implementation Flow",
            ["Data Pipeline", "Monotonic Lattice Training", "Explainability", "Counterfactual Solver", "Artifact Validation", "ONNX Parity"],
        ),
        draw_flow_diagram(
            "training_and_validation_pipeline.png",
            "Chapter 4 Training and Validation Pipeline",
            ["Split & Preprocess", "Tune Lattice", "Tune Baselines", "Parity Comparison", "Threshold Selection", "Final Manifest"],
        ),
    ]

    table_files = generate_tables()

    write_index(generated, copied)
    with (OUT / "TABLES_INDEX.txt").open("w", encoding="utf-8") as f:
        for name in table_files:
            f.write(name + "\n")

    print(f"Generated {len(generated)} figures and copied {len(copied)} core curves.")
    print(f"Generated {len(table_files)} summary tables.")
    print(f"Output folder: {OUT}")


if __name__ == "__main__":
    main()
