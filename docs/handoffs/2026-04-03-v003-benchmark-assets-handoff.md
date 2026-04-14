# Benchmark Assets Handoff

- Date: 2026-04-03
- Version: v003
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Session focus: bootstrap real starter benchmark assets from the reorganized CEMD data workspace

## What changed this session

- Added `scripts/bootstrap_benchmarks.py`.
- Generated deterministic starter benchmark assets under `benchmarks/`.
- Added a benchmark manifest at `benchmarks/manifest.json`.
- Updated `benchmarks/README.md` with the bootstrap workflow.

## Generated assets

### Gold

- `benchmarks/gold/starter_gold_100.csv`
  - 100 rows
  - source: `CEMD curriculum matching sample dataset.csv`
  - includes `selection_identifier` and `expected_product_identifier`

### Weak labels

- `benchmarks/weak_labels/starter_weak_labels_250.csv`
  - 250 rows
  - source: `matched_training_sample dataset.csv`

### Slices

- `benchmarks/slices/slice_missing_publisher_50.csv`
- `benchmarks/slices/slice_publisher_variants_50.csv`
- `benchmarks/slices/slice_ela_core_50.csv`

### Metadata

- `benchmarks/manifest.json`

## Source files used

- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/raw-input/samples/CEMD curriculum matching sample dataset.csv`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 002/matched_training_sample dataset.csv`
- `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/catalog/snapshots/CEMD Product Catalog - 05212025.csv`

## Verification completed

- `python3 scripts/bootstrap_benchmarks.py`
  - succeeded
- benchmark files are present in the expected folders

## Important blocker

The first full end-to-end evaluation run is still blocked by the local Python environment.

What happened:

- `PYTHONPATH=src python3 scripts/evaluate_matcher.py ...`
  failed with `ModuleNotFoundError: No module named 'sklearn'`
- the repo-local `venv` is a Windows virtual environment
  - `./venv/Scripts/python.exe`
  - it cannot run in the current shell environment and returns `exec format error`

## Implication

The code path for evaluation is implemented, but a real benchmark run needs one of these next:

1. create a fresh local macOS/Linux virtual environment in this repo and install dependencies
2. use another working Python environment that already has:
   - pandas
   - numpy
   - scikit-learn
   - rapidfuzz
   - rank-bm25
   - sentence-transformers
   - torch
   - transformers

## Recommended next step

- rebuild the repo environment with a local native venv
- rerun:
  - `python scripts/evaluate_matcher.py --benchmark-file benchmarks/gold/starter_gold_100.csv --catalog-file <catalog> --profile fast ...`
- save the first summary outputs into `benchmarks/outputs/`

After that, the next useful step will be comparing:

- starter gold set
- starter weak-label set
- each benchmark slice

to establish the current matcher baseline before redesigning retrieval or reranking.
