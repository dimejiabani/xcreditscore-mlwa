# Benchmark data

Raw data is not committed to this repository.

| Dataset | Licence position | How to obtain |
|---|---|---|
| **HELOC** (FICO Explainable ML Challenge) | Requires accepting FICO's data licence; redistribution not permitted | Register at the FICO community site, accept the licence, download `heloc_dataset_v1.csv`, and place it in the repository root as `heloc_dataset_v1 (1).csv` |
| **Give-Me-Some-Credit** | Kaggle competition terms restrict redistribution | `python scripts/fetch_datasets.py` (public mirror), or download `cs-training.csv` from Kaggle and save as `data/gmsc/gmsc.csv` with the index column dropped |
| **Taiwan Default of Credit Card Clients** | UCI, openly redistributable | `python scripts/fetch_datasets.py` |

After fetching, verify you have the exact bytes used for the published
results:

```bash
python scripts/fetch_datasets.py --verify
```

## Withheld per-instance files

Three HELOC artifacts are **not** included in this repository:

```
artifacts/heloc_pack/predictions/explainability_report.csv
artifacts/heloc_pack/predictions/explainability_shapley_2d_report.csv
artifacts/heloc_pack/predictions/xcreditscore_test_predictions.csv
```

Between them they record the min-max-scaled values of 26 of the 27 HELOC
predictors for every one of the 1,569 test rows, together with the true
labels. That is substantially the licensed test partition in a reversible
transform, and FICO's licence does not permit redistribution.

Consequences for a clean clone:

* Every **aggregate** result in the manuscript reproduces from the files that
  are included.
* `paper/verify_numbers.py` and `paper/regenerate_paper_figures.py` need the
  three files above, so obtain HELOC under licence and run
  `python run_project.py --dataset heloc` first. They regenerate byte-for-byte
  under the fixed seed.
* The corresponding author can supply them directly to anyone who has accepted
  FICO's licence.

The equivalent GMSC and Taiwan files are included: Taiwan is openly
redistributable via UCI, and the GMSC per-block file exposes 6 of 10
predictors rather than a near-complete feature vector.
