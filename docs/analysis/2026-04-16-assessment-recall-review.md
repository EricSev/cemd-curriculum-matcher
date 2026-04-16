# H2 Assessment-Aware Candidate Recall Review

- Date: `2026-04-16`
- Task: `H2` assessment-aware candidate recall experiment
- Decision: `rejected`
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h2_assessment_recall_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h2_assessment_recall_records.csv`
- Comparison artifact: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h2_assessment_recall_comparison.json`

## Single Variable

- enabled an opt-in retrieval experiment named `assessment_recall`
- for rows where `product_type_usage == Assessment`, added extra stage-1 candidates from assessment-like catalog rows before the existing scorer produced the final top-10

Held fixed:

- matcher baseline `C1 + C2 + C5`
- accepted stage-1 blend formula `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- no LLM rerank batch
- rerank prompt baseline `F2` no-score candidate payload

## Benchmark Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --retrieval-experiment assessment_recall --topn-final 10 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h2_assessment_recall_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h2_assessment_recall_records.csv
```

## Overall Delta

| Metric | Baseline | H2 Candidate | Delta |
| --- | ---: | ---: | ---: |
| hit@1 / top-1 | 0.4880 | 0.4880 | +0.0000 |
| hit@3 | 0.5740 | 0.5740 | +0.0000 |
| hit@10 | 0.6280 | 0.6280 | +0.0000 |
| MRR | 0.5374 | 0.5374 | +0.0000 |
| nDCG@10 | 0.5596 | 0.5596 | +0.0000 |

Row movement:

- changed shortlist rows: `51`
- changed metric rows: `16`
- top-1 gains / losses: `0 / 0`
- top-3 gains / losses: `0 / 0`
- top-10 gains / losses: `0 / 0`

## Required Slice Readout

| Slice | Baseline top-1 | H2 top-1 | Delta | Baseline hit@10 | H2 hit@10 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Assessment | 0.3654 | 0.3654 | +0.0000 | 0.4538 | 0.4538 | +0.0000 |
| assessment_short_or_acronym_title | 0.4286 | 0.4286 | +0.0000 | 0.4720 | 0.4720 | +0.0000 |
| assessment_state_specific_expected | 0.0000 | 0.0000 | +0.0000 | 0.8571 | 0.8571 | +0.0000 |
| assessment_publisher_missing | 0.3532 | 0.3532 | +0.0000 | 0.4450 | 0.4450 | +0.0000 |

Movement detail:

- `Assessment`: `51` changed shortlist rows, but `0` top-10 gains
- `assessment_short_or_acronym_title`: `44` changed shortlist rows, but `0` top-10 gains
- `assessment_state_specific_expected`: `1` changed shortlist row, but `0` top-10 gains
- `catalog_unspecified`: `0` changed shortlist rows
- `adoption_state_high_risk`: `6` changed shortlist rows, but `0` top-10 gains

## Decision

- reject `H2`
- do not send an LLM rerank batch
- restore `src/curriculum_matcher/app.py` and `tests/test_matcher_core.py` to the accepted baseline after measurement

Why:

- the candidate produced no overall top-1, top-3, top-10, MRR, or nDCG lift
- the assessment target slices did not improve
- shortlist movement without gold-row recovery is not enough to justify an LLM batch

## Recommended Next Step

Proceed to `H3` only if continuing the `H` series.

`H3` should focus on `catalog_unspecified` recovery, because H1 showed that slice is strongly recall-limited and H2 did not move it at all.
