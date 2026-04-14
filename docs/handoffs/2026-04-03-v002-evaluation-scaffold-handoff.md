# Matcher Evaluation Scaffold Handoff

- Date: 2026-04-03
- Version: v002
- Repo: `/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/VS Code/curriculum-matcher`
- Session focus: implement the first evaluation-first foundation for future matcher work

## What changed this session

- Added a reusable evaluation module at `src/curriculum_matcher/evaluation.py`.
- Added a package CLI entry at `src/curriculum_matcher/evaluation_cli.py`.
- Added a script runner at `scripts/evaluate_matcher.py`.
- Added benchmark folder scaffolding under `benchmarks/`.
- Added unit tests in:
  - `tests/test_matcher_core.py`
  - `tests/test_evaluation.py`
- Updated the repo README to document evaluation usage.
- Added a package script target:
  - `curriculum-matcher-evaluate`

## New capabilities

The repo can now evaluate matcher performance against a benchmark CSV plus catalog CSV and output:

- top-1 accuracy
- top-3 recall
- prediction rate
- mean top-1 score
- confidence-band metrics
- slice metrics for common fields such as subject, product type, state, and publisher
- optional per-record CSV output for analysis

## Key files

- `src/curriculum_matcher/evaluation.py`
  main evaluation logic and metric helpers
- `src/curriculum_matcher/evaluation_cli.py`
  package CLI entry
- `scripts/evaluate_matcher.py`
  repo-local runner
- `benchmarks/README.md`
  benchmark layout and usage notes
- `tests/test_matcher_core.py`
  baseline unit tests for normalization and parsing helpers
- `tests/test_evaluation.py`
  baseline unit tests for evaluation helpers

## Verification completed

- `PYTHONPATH=src python3 scripts/evaluate_matcher.py --help`
- `PYTHONPATH=src python3 -m unittest discover -s tests`

Results:

- evaluation CLI help succeeded
- 10 unit tests passed

## Benchmark file expectations

Benchmark CSVs should include normal matcher input columns plus one expected-id column.

Supported expected-id column names:

- `expected_product_identifier`
- `expected_catalog_id`
- `expected_match_id`
- `product_identifier`

## What this does not do yet

- no gold benchmark data has been added yet
- no calibration report exists yet
- no confusion matrix or richer error taxonomy exists yet
- no learned reranker exists yet
- no regression-comparison workflow exists yet between matcher versions

## Recommended next step

Create the first real benchmark assets:

1. a small trusted gold set
2. a weak-label benchmark set
3. a few named edge-case slices

Then run the current matcher as baseline and save the first summary outputs under `benchmarks/outputs/`.
