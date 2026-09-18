from __future__ import annotations

from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "artifacts" / "figures"
OUT = VIS / "implementation_figures"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight")
    plt.close(fig)


def draw_flow(name: str, title: str, nodes: list[str], subtitle: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    if subtitle:
        ax.text(0.5, 0.88, subtitle, ha="center", va="center", transform=ax.transAxes, fontsize=9)

    n = len(nodes)
    xs = [0.06 + i * (0.88 / max(1, n - 1)) for i in range(n)]
    y = 0.5

    for i, (x, node) in enumerate(zip(xs, nodes)):
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#edf4ff", edgecolor="#355070", linewidth=1.2),
            transform=ax.transAxes,
        )
        if i < n - 1:
            nx = xs[i + 1]
            ax.annotate(
                "",
                xy=(nx - 0.045, y),
                xytext=(x + 0.045, y),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.3, color="#1f2937"),
            )

    save(fig, name)


def copy_if_exists(src_name: str, dst_name: str) -> bool:
    src = VIS / src_name
    if src.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        copy2(src, OUT / dst_name)
        return True
    return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Figure 4.1
    draw_flow(
        "figure_4_1_end_to_end_implementation_roadmap.png",
        "Figure 4.1: End-to-End Implementation Roadmap",
        ["Raw Data", "Validation + Preprocess", "Monotonic Lattice Training", "Baseline Parity Tuning", "Explainability + Recourse", "Artifact Contract + ONNX Parity"],
        "From data ingestion to canonical reproducible artifact pack",
    )

    # Figure 4.2
    draw_flow(
        "figure_4_2_run_final_execution_sequence.png",
        "Figure 4.2: run_final.ps1 Execution Sequence and Gate Checks",
        ["Set RUN_PROFILE=final", "Train + Tune", "Generate Explanations", "Generate Recourse", "Validate Artifacts", "Export ONNX + Parity Smoke"],
    )

    # Figure 4.3
    draw_flow(
        "figure_4_3_data_preprocessing_flow.png",
        "Figure 4.3: Data Preprocessing Flow",
        ["Load CSV", "Schema + Missing Validation", "Sentinel Handling", "Imputation + Scaling", "Feature Engineering", "Train/Validation/Test Split"],
    )

    # Figure 4.4
    draw_flow(
        "figure_4_4_predictor_graph.png",
        "Figure 4.4: Internal Predictor Graph",
        ["Input Features", "Monotonic PWL Calibrators", "Grouped Lattice Blocks", "Non-Negative Combiner", "Sigmoid Score"],
    )

    # Figure 4.5
    draw_flow(
        "figure_4_5_baseline_tuning_flow.png",
        "Figure 4.5: Baseline Tuning with Parity Gates",
        ["Define CV Splits", "Set Trial Budget", "Tune LR / RF / XGB", "Match Lattice Budget", "Holdout Eval", "Select Best Baseline"],
    )

    # Figure 4.6 and 4.7 can reuse generated analytical charts.
    copied_46 = copy_if_exists("threshold_frontier.png", "figure_4_6_threshold_frontier.png")
    copied_47 = copy_if_exists("xcreditscore_reliability_curve.png", "figure_4_7_reliability_curve.png")

    # Figure 4.8
    draw_flow(
        "figure_4_8_explainability_pipeline.png",
        "Figure 4.8: Explainability Pipeline",
        ["Prediction Request", "Mode Selection (exact/fallback)", "Contribution Extraction", "Reason Code Ranking", "Stability Diagnostics", "Audit Artifact Export"],
    )

    # Figure 4.9
    draw_flow(
        "figure_4_9_counterfactual_workflow.png",
        "Figure 4.9: Counterfactual Recourse Workflow",
        ["Denied Case", "Constraint Assembly", "Exact/Approx Solver", "Feasibility Check", "Action Plan Generation", "CSV/JSON Export"],
    )

    # Figure 4.10
    draw_flow(
        "figure_4_10_monotonic_check_workflow.png",
        "Figure 4.10: Monotonic Check + Contract Gate",
        ["Sample Instances", "Perturb Monotonic Features", "Compute Violations", "CI Estimation", "Write Report", "Contract Validator Gate"],
    )

    # Figure 4.11
    draw_flow(
        "figure_4_11_onnx_export_parity_pipeline.png",
        "Figure 4.11: ONNX Export and Parity Pipeline",
        ["Trained Native Model", "tf2onnx Export", "ONNX Runtime Inference", "Native-vs-ONNX Error", "Tolerance Check", "Parity Status Artifact"],
    )

    # Figure 4.12
    draw_flow(
        "figure_4_12_governance_stack.png",
        "Figure 4.12: Governance Stack",
        ["Config Hash", "Environment Manifest", "Run Summary", "Failure Transparency", "Final Manifest", "Experiment Registry"],
    )

    # Figure 4.13
    draw_flow(
        "figure_4_13_transition_to_results.png",
        "Figure 4.13: Transition from Implementation to Results",
        ["Implemented Pipelines", "Validated Artifacts", "Comparative Metrics", "Explainability Quality", "Recourse Outcomes", "Chapter 5 Discussion"],
    )

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Chapter 4 Implementation Figures",
                "",
                "This folder contains generated implementation-focused figures mapped to placeholders Figure 4.1 to Figure 4.13.",
                "",
                "## Notes",
                "- Figure 4.6 reuses threshold frontier chart if available.",
                "- Figure 4.7 reuses reliability curve if available.",
                f"- Figure 4.6 source chart copied: {copied_46}",
                f"- Figure 4.7 source chart copied: {copied_47}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Generated Chapter 4 implementation figures in: {OUT}")


if __name__ == "__main__":
    main()
