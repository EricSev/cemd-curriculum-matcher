# Field Corruption Benchmark Report

- profile: `fast`
- topn_final: `3`
- slice count: 5

## Summary

| Slice | Top-1 | Delta | Top-3 | Delta | Prediction | Delta | Gate Reject | Blank Title | Blank Publisher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean_reference | 0.410 | +0.000 | 0.520 | +0.000 | 0.740 | +0.000 | 0.420 | 0.000 | 0.370 |
| grade_corrupted_or_missing | 0.430 | +0.020 | 0.540 | +0.020 | 0.740 | +0.000 | 0.420 | 0.000 | 0.370 |
| publisher_in_title_publisher_blank | 0.020 | -0.390 | 0.020 | -0.500 | 0.310 | -0.430 | 0.960 | 0.370 | 1.000 |
| title_in_publisher_title_blank | 0.000 | -0.410 | 0.000 | -0.520 | 0.000 | -0.740 | 1.000 | 1.000 | 0.000 |
| title_publisher_swapped | 0.020 | -0.390 | 0.020 | -0.500 | 0.310 | -0.430 | 0.960 | 0.370 | 0.000 |

## Takeaways

- Worst top-3 recall drop: `title_in_publisher_title_blank` moved from 0.520 to 0.000.
- Worst prediction-rate drop: `title_in_publisher_title_blank` moved from 0.740 to 0.000.
- Highest expected-row gate rejection rate: `title_in_publisher_title_blank` at 1.000.
- High gate rejection suggests the right catalog row is being filtered out before publisher and grade can help.

## Reproduce

```bash
./.venv/bin/python scripts/build_corruption_benchmarks.py \
  --source-benchmark benchmarks/gold/starter_gold_100.csv

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/evaluate_corruption_benchmarks.py \
  --benchmark-dir benchmarks/corruptions/starter_gold_100 \
  --catalog-file '/absolute/path/to/catalog.csv' \
  --profile fast
```
