"""Regenerate the four figures that were unreadable or empty in the manuscript.

Figures 1 and 2 are schematic diagrams. They were imported from an earlier
draft, were sized portrait so that scaling to page width made the labels
illegible, and Figure 1 had gone stale (three baselines, 12 tuning trials,
one dataset). Both are rebuilt here in landscape at a size where the text
survives placement at 6.5 inch page width, and Figure 1 now reads its
counts from the live configuration.

Figures 17 and 18 rendered blank. Figure 17 plotted per-feature violation
rates for a model whose violation rate is zero everywhere, so every bar had
zero length. Figure 18 put values of order 1e-7 on a log axis whose limits
were 1e0 to 1e1, placing every bar far below the visible range. Both are
rebuilt to show the result rather than an empty frame.

Usage:  .venv/bin/python scripts/make_paper_diagrams.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "artifacts" / "paper_figures" / "regenerated"
VISUALS = ROOT / "artifacts" / "figures"
DIAGRAMS.mkdir(parents=True, exist_ok=True)

INK = "#1f2a37"
NAVY = "#2d3e50"
BLUE = "#2e86c1"
GREEN = "#1e8449"
TEAL = "#17a589"
ORANGE = "#e67e22"
AMBER = "#f39c12"
PURPLE = "#8e44ad"
RED_F = "#fadbd8"
RED_E = "#c0392b"
TEAL_F = "#d1f2eb"
GREY_F = "#eceff1"
GREY_E = "#7f8c8d"


def js(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p))) if p.exists() else []


def facts() -> dict:
    """Live numbers so the diagram cannot drift from the pipeline again."""
    os.environ.setdefault("DATASET_NAME", "heloc")
    import sys

    sys.path.insert(0, str(ROOT))
    from src.datasets import heloc as D

    mono = D.MONOTONIC_CONSTRAINTS
    summary = js(ROOT / "artifacts/heloc_pack/metrics/counterfactual_full.json")
    tau = js(ROOT / "artifacts/heloc_pack/counterfactuals/"
                    "counterfactual_full_evaluation_summary.json") \
        .get("recourse_validity", {}).get("tau", 0.42)
    bl = [r["model"] for r in
          rows(ROOT / "artifacts/heloc_pack/metrics/baseline_metrics.csv")]
    return {
        "n_raw": len(D.FEATURE_COLS),
        "n_total": len(mono),
        "n_eng": len(mono) - len(D.FEATURE_COLS),
        "protective": sum(1 for v in mono.values() if v == -1),
        "risk": sum(1 for v in mono.values() if v == 1),
        "free": sum(1 for v in mono.values() if v == 0),
        "blocks": len(D.LATTICE_FIXED_GROUPS),
        "immutable": len(D.IMMUTABLE_FEATURES),
        "baselines": bl,
        "tau": float(tau),
    }


def box(ax, x, y, w, h, text, *, fc="white", ec=INK, fs=11, weight="normal",
        tc=None, radius=0.012, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc or INK, weight=weight, zorder=3,
            linespacing=1.45)


def band(ax, x, y, w, h, text, fc, fs=13):
    box(ax, x, y, w, h, text, fc=fc, ec=fc, fs=fs, weight="bold", tc="white")


def arrow(ax, p0, p1, *, color=NAVY, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15,
        color=color, linewidth=lw, zorder=1,
        shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------
# Figure 1: end-to-end pipeline
# --------------------------------------------------------------------------
def figure1(F: dict) -> Path:
    fig, ax = plt.subplots(figsize=(15.0, 9.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 63)
    ax.axis("off")
    ax.set_title("XCreditScore: End-to-End Pipeline", fontsize=21,
                 weight="bold", color=NAVY, pad=14)

    nb = len(F["baselines"])
    col = [1.5, 34.5, 67.5]
    cw = 31.0

    def stage(ci, y, h, title, fc, items):
        x = col[ci]
        band(ax, x, y + h - 4.0, cw, 4.0, title, fc, fs=14.6)
        n = len(items)
        pad, gap = 0.7, 0.6
        ih = (h - 4.0 - pad - gap * (n - 1)) / n
        for k, (txt, f_, e_) in enumerate(items):
            iy = y + h - 4.0 - pad - (k + 1) * ih - k * gap
            box(ax, x + 0.8, iy, cw - 1.6, ih, txt, fc=f_, ec=e_, fs=11.4)
        return x + cw / 2

    # ---- column 1 -------------------------------------------------------
    stage(0, 42.0, 20.0, "1.  Data Input", NAVY, [
        ("Three credit benchmarks\nHELOC  10,459 x 23    Taiwan  30,000 x 23\n"
         "GMSC  150,000 x 10", "#eaf2fb", BLUE),
        ("Integrity validation\nschema, dtypes, target mapping", "#eaf2fb", BLUE),
    ])
    stage(0, 21.0, 19.0, "2.  Preprocessing", BLUE, [
        ("Sentinel handling\nper-dataset codes to NaN", "#eaf2fb", BLUE),
        ("Median imputation\nfitted on training partition only", "#eaf2fb", BLUE),
        ("Min-max scaling to [0, 1]\nfitted on training partition only", "#eaf2fb", BLUE),
        ("Stratified split  70 / 15 / 15\nrandom_state = 42", "#eaf2fb", BLUE),
    ])
    stage(0, 1.0, 18.0, "3.  Feature Engineering", GREEN, [
        ("HELOC adds four derived features\nTotalDelqEvents, InquiryToTradeRatio,\n"
         "RevolvingBurdenPerTrade, InstallBurdenPerTrade", "#e8f6ee", GREEN),
        (f"{F['n_raw']} raw  ->  {F['n_total']} modelled features on HELOC\n"
         "Taiwan and GMSC use raw features", "#e8f6ee", GREEN),
    ])

    # ---- column 2 -------------------------------------------------------
    stage(1, 42.0, 20.0, "4.  Monotonic Constraint Registry", TEAL, [
        (f"Protective  (direction -1):  {F['protective']} features", TEAL_F, TEAL),
        (f"Risk-increasing  (direction +1):  {F['risk']} features", RED_F, RED_E),
        (f"Unconstrained  (direction 0):  {F['free']} features", GREY_F, GREY_E),
        (f"Immutable for recourse:  {F['immutable']} features\n"
         "One registry governs all three components",
         "#ffffff", TEAL),
    ])
    stage(1, 21.0, 19.0, "5.  Model Training  (40 trials per model)", ORANGE, [
        ("XCreditScore: calibrated lattice ensemble\n"
         f"PWL calibration (15 keypoints), {F['blocks']} lattice blocks,\n"
         "non-negative combiner, sigmoid", "#fdf0e2", ORANGE),
        (f"{nb} baselines under an identical 40-trial budget\n"
         "logistic regression, random forest, XGBoost,\n"
         "monotone XGBoost, EBM, monotone GAM", "#f4f6f7", GREY_E),
    ])
    stage(1, 1.0, 18.0, "6.  Threshold Decision", AMBER, [
        (f"Decision threshold  tau = {F['tau']:.2f}\n"
         "selected by F1 maximisation on validation", "#fef6e6", AMBER),
        ("f(x) < tau  ->  Approved", "#e8f6ee", GREEN),
        ("f(x) >= tau  ->  Denied  ->  recourse", RED_F, RED_E),
    ])

    # ---- column 3 -------------------------------------------------------
    stage(2, 42.0, 20.0, "7.  Explanation and Recourse", PURPLE, [
        ("Structure-derived attribution\nexact PWL + lattice logit, no surrogate\n"
         "two-player Shapley per 2-D block,\ninteraction residual reported separately",
         "#f3e8f7", PURPLE),
        ("Mixed-integer recourse\nGurobi with CBC fallback, SOS2 encoding\n"
         "every denied applicant, solver status logged", "#f3e8f7", PURPLE),
    ])
    stage(2, 21.0, 19.0, "8.  Evaluation", TEAL, [
        ("Predictive performance\nAUC, PR-AUC, Brier, log-loss", TEAL_F, TEAL),
        ("Corrected inference\nHolm-Bonferroni + Nadeau-Bengio", TEAL_F, TEAL),
        ("Stability, monotonicity, calibration\nSHAP / LIME head-to-head, "
         "perturbation protocol", TEAL_F, TEAL),
        ("Recourse feasibility\nfull denied population, Wilson 95% CIs", TEAL_F, TEAL),
    ])
    stage(2, 1.0, 18.0, "9.  Outputs", NAVY, [
        ("Models and ONNX export\nparity verified over the full test partition",
         GREY_F, GREY_E),
        ("Per-instance predictions, attributions,\nrecourse plans and solver statuses",
         GREY_F, GREY_E),
        ("Environment manifest, fixed seed,\nconfiguration and run metadata", GREY_F, GREY_E),
    ])

    # Within-column flow only. Stage numbering carries the 1..9 order across
    # columns; drawing those hops as arrows made them cross through boxes.
    for ci in (0, 1, 2):
        x = col[ci] + cw / 2
        arrow(ax, (x, 42.0), (x, 40.2))
        arrow(ax, (x, 21.0), (x, 19.2))

    fig.tight_layout()
    out = DIAGRAMS / "figure1_pipeline.png"
    fig.savefig(out, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Figure 2: model architecture
# --------------------------------------------------------------------------
def figure2(F: dict) -> Path:
    fig, ax = plt.subplots(figsize=(15.0, 7.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")
    ax.set_title("XCreditScore: Monotonic Calibrated Lattice Ensemble Architecture",
                 fontsize=20, weight="bold", color=NAVY, pad=12)

    heads = ["Stage 1\nFeature Inversion", "Stage 2\nPWL Calibration",
             "Stage 3\nGrouped 2-D Lattice Blocks", "Stage 4\nOutput Combiner"]
    xs = [1.0, 24.0, 46.5, 76.5]
    ws = [21.5, 20.5, 28.5, 22.5]
    for x, w, h in zip(xs, ws, heads):
        band(ax, x, 44.0, w, 5.4, h, NAVY, fs=13)

    rowsy = [(32.5, TEAL_F, TEAL, f"Protective  (-1)\n{F['protective']} features\n"
                                  "ExternalRiskEstimate,\nMSinceOldestTradeOpen, ..."),
             (20.0, RED_F, RED_E, f"Risk-increasing  (+1)\n{F['risk']} features\n"
                                  "NumTrades60Ever2DerogPubRec,\nNumInqLast6M, ..."),
             (7.5, GREY_F, GREY_E, f"Unconstrained  (0)\n{F['free']} features\n"
                                   "MSinceMostRecentTradeOpen, ...")]
    inv = ["x~ = 1 - x\n(inverted so the\nconstraint is increasing)",
           "x~ = x", "x~ = x"]
    for (y, fc, ec, txt), iv in zip(rowsy, inv):
        box(ax, xs[0], y, ws[0], 10.0, txt, fc=fc, ec=ec, fs=11.4)
        box(ax, xs[1], y + 1.6, ws[1], 6.8, iv, fc=fc, ec=ec, fs=11.4)
        arrow(ax, (xs[0] + ws[0], y + 5.0), (xs[1], y + 5.0))
        ax.text(xs[1] + ws[1] / 2, y + 0.7,
                "phi(x~): monotone PWL, 15 keypoints",
                ha="center", va="center", fontsize=10.6, color=INK, zorder=3)

    blocks = [(35.0, TEAL_F, TEAL, "Block 1\nExternalRiskEstimate  x  AverageMInFile"),
              (27.2, TEAL_F, TEAL, "Block 2\nMSinceOldestTradeOpen\nx  MSinceMostRecentTrade"),
              (19.4, RED_F, RED_E, "Block 3\nNumTrades60Ever  x  NumTrades90Ever"),
              (11.6, RED_F, RED_E, "Block 4\nMaxDelq2PublicRec  x  MaxDelqEver"),
              (3.8, GREY_F, GREY_E, f"... {F['blocks']} grouped blocks in total\n"
                                    "lattice_size = 4 per axis (16 vertices)")]
    for y, fc, ec, txt in blocks:
        box(ax, xs[2], y, ws[2], 6.9, txt, fc=fc, ec=ec, fs=11.2)
    for (y, fc, ec, _t) in rowsy:
        arrow(ax, (xs[1] + ws[1], y + 5.0), (xs[2] - 0.4, 22.8), lw=1.3)

    box(ax, xs[3], 24.0, ws[3], 12.0,
        "Non-negative dense layer\n(single unit)\n\n"
        "l(x) = b + sum_k w_k g_k\nwith w_k >= 0\n\n"
        "P(default) = sigma(l(x))",
        fc="white", ec=NAVY, fs=11.6)
    box(ax, xs[3] + 2.0, 12.0, ws[3] - 4.0, 7.0,
        "f(x)\nDefault probability", fc="#fdebd0", ec=AMBER, fs=12, weight="bold")
    arrow(ax, (xs[2] + ws[2], 22.8), (xs[3], 30.0), lw=2.0)
    arrow(ax, (xs[3] + ws[3] / 2, 24.0), (xs[3] + ws[3] / 2, 19.0), lw=2.0)

    ax.text(50, -1.2,
            "End-to-end monotonicity is a compile-time constraint: raising a "
            "risk-increasing feature can only raise f(x), and raising a "
            "protective feature can only lower it, up to floating-point noise.",
            ha="center", va="center", fontsize=11, style="italic", color=INK)

    fig.tight_layout()
    out = DIAGRAMS / "figure2_architecture.png"
    fig.savefig(out, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Figure 17: monotonicity outcome across constrained models and datasets
# --------------------------------------------------------------------------
def figure17() -> Path:
    packs = [("HELOC", "heloc_pack"), ("Taiwan", "taiwan_pack"),
             ("GMSC", "gmsc_pack")]
    models = ["xcreditscore", "monotone_xgboost", "ebm", "monotone_gam"]
    labels = ["XCreditScore", "Monotone XGBoost", "EBM", "Monotone GAM"]
    data = {}
    for ds, pk in packs:
        src = {r["model"]: r for r in
               rows(ROOT / f"artifacts/{pk}/metrics/monotonic_sanity_comparison.csv")}
        data[ds] = [(int(src[m]["total_violations"]), int(src[m]["total_checks"]))
                    if m in src else (0, 0) for m in models]

    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    width = 0.2
    xpos = range(len(packs))
    colours = ["#17a589", "#2e86c1", "#8e44ad", "#e67e22"]
    for i, (lab, c) in enumerate(zip(labels, colours)):
        vals = [data[ds][i][0] for ds, _ in packs]
        offs = [x + (i - 1.5) * width for x in xpos]
        ax.bar(offs, vals, width * 0.92, label=lab, color=c,
               edgecolor=INK, linewidth=0.7, zorder=3)
        for xo, v in zip(offs, vals):
            ax.annotate(str(v), (xo, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=10,
                        weight="bold" if v else "normal",
                        color=RED_E if v else INK, zorder=4)

    ax.set_xticks(list(xpos))
    ax.set_xticklabels([f"{ds}\n{data[ds][0][1]:,} checks per model"
                        for ds, _ in packs], fontsize=11)
    ax.set_ylabel("Monotonicity violations (count)", fontsize=12)
    ax.set_title("Monotonicity violations under the perturbation protocol\n"
                 "Constrained models, all three benchmarks",
                 fontsize=14, weight="bold", color=NAVY)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=10.5, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("Zero is the expected outcome for structurally monotone models;\n"
                "the monotone GAM applies a soft spline penalty, not a hard constraint",
                xy=(0.015, 0.90), xycoords="axes fraction", fontsize=10,
                style="italic", color=INK)
    fig.tight_layout()
    out = VISUALS / "monotonic_violation_rate.png"
    fig.savefig(out, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Figure 18: ONNX parity error against tolerance
# --------------------------------------------------------------------------
def figure18() -> Path:
    o = js(ROOT / "artifacts/heloc_pack/metrics/onnx_parity_smoke.json")
    mean_e = float(o["mean_abs_err"])
    max_e = float(o["max_abs_err"])
    tol = float(o.get("tolerance", 1e-3))

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    names = ["Mean absolute\ndifference", "Maximum absolute\ndifference",
             "Configured\ntolerance"]
    vals = [mean_e, max_e, tol]
    cols = ["#17a589", "#2e86c1", "#c0392b"]
    bars = ax.bar(names, vals, 0.52, color=cols, edgecolor=INK,
                  linewidth=0.9, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(min(vals) / 12, tol * 22)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3e}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=11.5, weight="bold", color=INK, zorder=4)
    ax.axhline(tol, color=RED_E, linestyle="--", linewidth=1.4, zorder=2)
    ax.set_ylabel("Absolute prediction difference (log scale)", fontsize=12)
    ax.set_title(f"ONNX conversion parity over the full HELOC test partition "
                 f"({int(o['rows']):,} rows)\n"
                 f"Maximum error is {tol / max_e:,.0f}x below the configured tolerance; "
                 f"{int(o['decision_flips_at_tau'])} rows change decision",
                 fontsize=13.5, weight="bold", color=NAVY)
    ax.grid(axis="y", alpha=0.25, which="both", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = VISUALS / "onnx_parity_error.png"
    fig.savefig(out, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    F = facts()
    print("live facts:", {k: v for k, v in F.items() if k != "baselines"})
    print("baselines:", ", ".join(F["baselines"]))
    for fn in (lambda: figure1(F), lambda: figure2(F), figure17, figure18):
        p = fn()
        print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
