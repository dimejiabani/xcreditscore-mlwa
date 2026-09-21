"""Export the complete constraint registry for each dataset as JSON.

The registry the paper describes declares four things per feature: the
monotonic direction, whether the feature is immutable, the per-feature
step-size fraction recourse may move it by, and the justification for the
direction. Those live in two places in the source: the direction and its
justification in artifacts/<ds>_pack/config/monotonic_constraints_documented.json,
and the immutability set and step fractions in src/datasets/<ds>.py, with
src/config.py supplying the step fraction used where a feature declares none.

This script joins them into one file per dataset,
artifacts/<ds>_pack/config/constraint_registry.json, so the machine-readable
release matches the registry tables in the paper column for column rather
than carrying only half of it.

Usage:
    python scripts/export_constraint_registry.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("heloc", "gmsc", "taiwan")


def _module_constant(dataset: str, name: str):
    """Read a literal module-level constant without importing the package."""
    source = (ROOT / "src" / "datasets" / f"{dataset}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and node.targets:
            target = getattr(node.targets[0], "id", None)
        if target == name:
            return ast.literal_eval(node.value)
    raise LookupError(f"{name} not found in src/datasets/{dataset}.py")


def build(dataset: str) -> Path:
    config = ROOT / "artifacts" / f"{dataset}_pack" / "config"
    documented = json.loads((config / "monotonic_constraints_documented.json").read_text(encoding="utf-8"))
    features = documented["features"]

    immutable = set(_module_constant(dataset, "IMMUTABLE_FEATURES"))
    steps = dict(_module_constant(dataset, "COUNTERFACTUAL_MAX_STEP_FRACTIONS"))

    # Features without an explicit bound fall back to the default the recourse
    # engine applies, so the exported value is the bound actually enforced.
    config_src = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
    default_step = float(
        next(l for l in config_src.splitlines()
             if l.startswith("COUNTERFACTUAL_DEFAULT_MAX_STEP_FRACTION")).split("=")[1]
    )

    unknown = (immutable | set(steps)) - set(features)
    if unknown:
        raise ValueError(f"{dataset}: names absent from the documented registry: {sorted(unknown)}")

    out = {
        "dataset": dataset,
        "default_step_fraction": default_step,
        "description": (
            "Per-feature constraint registry. direction: -1 protective, "
            "+1 risk-increasing, 0 unconstrained. immutable: recourse may not "
            "move the feature. step_fraction: the largest fraction of the "
            "feature's observed range a single recourse plan may move it by, "
            "null where the feature is immutable."
        ),
        "features": {
            name: {
                "direction": entry["direction"],
                "immutable": name in immutable,
                "step_fraction": (
                    None if name in immutable else steps.get(name, default_step)
                ),
                "step_fraction_source": (
                    None if name in immutable
                    else ("declared" if name in steps else "default")
                ),
                "justification": entry["justification"],
            }
            for name, entry in features.items()
        },
    }
    path = config / "constraint_registry.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    for ds in DATASETS:
        p = build(ds)
        data = json.loads(p.read_text(encoding="utf-8"))["features"]
        n_imm = sum(1 for v in data.values() if v["immutable"])
        n_decl = sum(1 for v in data.values() if v["step_fraction_source"] == "declared")
        n_def = sum(1 for v in data.values() if v["step_fraction_source"] == "default")
        print(f"{ds:7} {len(data):>2} features  {n_imm:>2} immutable  "
              f"{n_decl:>2} declared step  {n_def:>2} default step  -> {p.relative_to(ROOT)}")
