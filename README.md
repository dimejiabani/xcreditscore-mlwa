# Governing Prediction, Explanation and Recourse from One Constraint Registry

Reproducible pipeline and artifacts for a credit-scoring architecture in which
a single shared constraint registry governs prediction, explanation and
recourse together. The registry declares four things per feature, a monotonic
direction, an immutability flag, a step-size bound and the justification for
the direction, and that one declaration drives the predictor, the
structure-derived attribution layer and the mixed-integer counterfactual
recourse engine alike, so the three cannot diverge.

Around the registry the pipeline implements a parity-controlled evaluation
protocol: a common per-model tuning budget, a fixed seed and split held
constant across every reported number, paired cross-validated inference under
both Holm-Bonferroni and Nadeau-Bengio corrections, full-population recourse
with Wilson confidence intervals, and a post-solve validity check.

## Predictors

Seven predictors are implemented and evaluated under the same registry, the
same budget and the same splits on three public benchmarks (HELOC, Taiwan
Default of Credit Card Clients, Give-Me-Some-Credit):

| Predictor | Shape constraint | Implementation |
|---|---|---|
| Monotonic calibrated lattice ensemble | structural | TensorFlow Lattice |
| Logistic regression | none | scikit-learn |
| Random forest | none | scikit-learn |
| XGBoost | none | XGBoost |
| Monotone-constrained XGBoost | structural | XGBoost |
| Explainable Boosting Machine | structural | interpret |
| Monotone generalised additive model | penalty | pyGAM |

The registry is architecture-agnostic: any predictor exposing a monotone
scoring function can be placed under it, and the recourse engine accepts any
of them unchanged. Comparing the constrained predictors against their
unconstrained counterparts under an identical budget is what the protocol is
for, so the baselines are part of the result rather than a backdrop to it.

## What this repository contains

```
src/                     pipeline (datasets, predictor, baselines, recourse, stats)
scripts/                 dataset fetcher, figure generation
tests/                   unit tests
artifacts/
  heloc_pack/            HELOC results
  taiwan_pack/           Taiwan results
  gmsc_pack/             Give-Me-Some-Credit results
  figures/               generated figures
```

Each `<dataset>_pack/config/constraint_registry.json` is the machine-readable
registry for that dataset: direction, immutability, step-size bound and
justification for every feature. Regenerate them with
`python scripts/export_constraint_registry.py`.

Every reported number is produced by the pipeline and written under
`artifacts/`, so results can be traced back to the run that produced them.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt

python scripts/fetch_datasets.py          # see data/README.md for HELOC
python run_project.py --dataset heloc
python run_project.py --dataset taiwan
python run_project.py --dataset gmsc
```

Optional stages, off by default because they are expensive:

```bash
ENABLE_HEAD_TO_HEAD_RECOURSE=1  python run_project.py --dataset heloc
SKIP_EXPLAIN_STABILITY_BENCHMARK=0 python run_project.py --dataset heloc
```

Regenerate the figures from the artifact set:

```bash
python scripts/generate_figures.py
```

`xgboost` on macOS needs OpenMP (`brew install libomp`).

## Reproducibility notes

* Fixed seed 42 and a fixed stratified 70/15/15 split throughout.
* Threshold-independent metrics for the non-lattice models are bit-stable
  across runs. TensorFlow Lattice training is not bit-deterministic on all
  hardware, so the lattice model's AUC can move in the third decimal between
  runs; this is smaller than the between-model spread but is disclosed
  rather than hidden.
* `artifacts/*/metrics/environment_manifest.json` records the exact package
  versions used for the published run.

## Data

Benchmark datasets are governed by their own licences and are not
redistributed here. `data/README.md` documents how to obtain each one, and
records which HELOC per-instance files are withheld under FICO's licence.

## Licence

Code and generated artifacts: MIT (see `LICENSE`).
