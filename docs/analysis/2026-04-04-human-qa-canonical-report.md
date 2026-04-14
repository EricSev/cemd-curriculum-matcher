# Human QA Canonical Benchmark Report

This report captures the first QA evaluation against the richer canonical silver benchmark.

Source artifacts:

- `benchmarks/qa/starter_gold_human_qa_canonical.csv`
- `benchmarks/qa/starter_gold_human_qa_canonical_summary.json`
- `benchmarks/outputs/starter-gold-human-qa-canonical-summary.json`

## Benchmark Composition

- row count: `261`
- label counts:
  - correct human match: `100`
  - incorrect human match: `161`
- label strength:
  - gold positive: `100`
  - silver strong negative: `53`
  - silver negative: `108`
- slice counts:
  - fallback selected: `105`
  - has adopted url: `156`

## QA Metrics

- challenge rate: `0.2567`
- precision: `0.8209`
- recall: `0.3416`
- incorrect row rate: `0.6169`

## Current Interpretation

- The richer canonical benchmark continues to produce a stronger precision signal than the earlier synthetic-only benchmark.
- The current working operating point is now narrower and more operational: it challenges about one quarter of rows instead of more than half, while keeping precision above `0.82`.
- Recall is intentionally lower than the previous broad challenge setting, so this scorer should be treated as a reviewer triage layer rather than an exhaustive error-catch layer.
- The canonical benchmark is now the working benchmark for Milestone 2 threshold tuning and reviewer workflow design.
