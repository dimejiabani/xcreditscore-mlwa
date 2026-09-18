# XCreditScore

Reproducible pipeline and artifacts for a monotonic calibrated lattice
ensemble for credit scoring, evaluated against six baselines on three public
benchmarks under a single shared constraint registry that drives the
predictor, the attribution layer and the counterfactual recourse engine
together.

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
