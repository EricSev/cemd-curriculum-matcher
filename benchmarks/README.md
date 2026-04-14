# Benchmarks

This folder is the canonical home for matcher evaluation assets.

## Suggested layout

```text
benchmarks/
  gold/
  weak_labels/
  slices/
  corruptions/
  outputs/
```

## Benchmark input expectations

Benchmark CSV files should contain the normal matcher input columns plus one expected-id column.

Supported expected-id column names:

- `expected_product_identifier`
- `expected_catalog_id`
- `expected_match_id`
- `product_identifier`

## Evaluation command

```bash
python scripts/evaluate_matcher.py \
  --benchmark-file path/to/benchmark.csv \
  --catalog-file path/to/catalog.csv \
  --profile accurate \
  --output-json benchmarks/outputs/latest-summary.json \
  --output-csv benchmarks/outputs/latest-records.csv
```

## Metrics produced

- top-1 accuracy
- top-3 recall
- prediction rate
- confidence-band accuracy
- slice metrics for common columns such as subject, product type, state, and publisher

## Starter assets

Use the bootstrap script to create deterministic starter benchmark files from the reorganized `Consulting/Meg/02-data` source files:

```bash
python scripts/bootstrap_benchmarks.py
```

This generates:

- `gold/starter_gold_100.csv`
- `weak_labels/starter_weak_labels_250.csv`
- `slices/slice_missing_publisher_50.csv`
- `slices/slice_publisher_variants_50.csv`
- `slices/slice_ela_core_50.csv`
- `manifest.json`

## Field-corruption evaluation

Use the corruption builder to create deterministic bad-input variants from the gold set:

```bash
./.venv/bin/python scripts/build_corruption_benchmarks.py \
  --source-benchmark benchmarks/gold/starter_gold_100.csv
```

This writes `benchmarks/corruptions/starter_gold_100/` with:

- `clean_reference.csv`
- `title_publisher_swapped.csv`
- `title_in_publisher_title_blank.csv`
- `publisher_in_title_publisher_blank.csv`
- `grade_corrupted_or_missing.csv`

Then evaluate the full corruption suite and generate a comparison report:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/evaluate_corruption_benchmarks.py \
  --benchmark-dir benchmarks/corruptions/starter_gold_100 \
  --catalog-file /absolute/path/to/catalog.csv \
  --profile fast
```

This writes per-slice outputs under `benchmarks/outputs/corruptions/` plus a markdown report in `docs/analysis/`.

## Human-match QA benchmark

Use saved evaluation outputs to synthesize a QA benchmark with both correct and incorrect human-selected matches:

```bash
./.venv/bin/python scripts/build_human_qa_benchmark.py \
  --benchmark-file benchmarks/gold/starter_gold_100.csv \
  --records-csv benchmarks/outputs/starter-gold-fast-records.csv \
  --output-csv benchmarks/qa/starter_gold_human_qa.csv
```

Then evaluate challenge precision and recall:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/evaluate_human_match_qa.py \
  --input-file benchmarks/qa/starter_gold_human_qa.csv \
  --catalog-file /absolute/path/to/catalog.csv \
  --profile fast
```
