# I4 Taxonomy-Informed Retrieval Candidate Review

- Date: `2026-04-17`
- Task: `I4` taxonomy-informed retrieval candidate
- Decision: `rejected`
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_i4_taxonomy_recall_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_i4_taxonomy_recall_records.csv`
- Comparison artifact: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_i4_taxonomy_recall_comparison.json`

## Single Variable

- enabled an opt-in retrieval experiment named `i4_taxonomy_recall`
- preserved the accepted char n-gram blend
- appended a tiny set of assessment-like catalog candidates for the clean `I1` assessment retrieval-miss patterns before the existing scorer/cutoff produced the final top-10

Held fixed:

- matcher baseline `C1 + C2 + C5`
- accepted stage-1 blend formula `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- no LLM rerank batch
- rerank prompt baseline `F2` no-score candidate payload

## Benchmark Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --retrieval-experiment i4_taxonomy_recall --topn-final 10 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_i4_taxonomy_recall_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_i4_taxonomy_recall_records.csv
```

## Overall Delta

| Metric | Baseline | I4 Candidate | Delta |
| --- | ---: | ---: | ---: |
| hit@1 / top-1 | 0.4880 | 0.4880 | +0.0000 |
| hit@3 | 0.5740 | 0.5740 | +0.0000 |
| hit@10 | 0.6280 | 0.6280 | +0.0000 |
| MRR | 0.5374 | 0.5374 | +0.0000 |
| nDCG@10 | 0.5596 | 0.5596 | +0.0000 |

Row movement:

- changed shortlist rows: `0`
- top-1 gains / losses: `0 / 0`
- top-3 gains / losses: `0 / 0`
- top-10 gains / losses: `0 / 0`

## Required Slice Readout

| Slice | Count | Top-1 Delta | Hit@10 Delta | Changed Shortlists |
| --- | ---: | ---: | ---: | ---: |
| Assessment | 260 | +0.0000 | +0.0000 | 0 |
| assessment_other | 11 | +0.0000 | +0.0000 | 0 |
| catalog_unspecified proxy | 56 | +0.0000 | +0.0000 | 0 |

## Decision

- reject `I4`
- do not send an LLM rerank batch
- restore `src/curriculum_matcher/app.py` and `tests/test_matcher_core.py` to the accepted baseline after measurement

Why:

- the opt-in retrieval candidate produced no shortlist movement
- no top-10 recovery gains appeared
- no target slice improved
- the result confirms the `I1` / `I2` readout: most remaining issues are taxonomy, catalog structure, or historical-label ambiguity rather than a simple candidate-recall rule

## I-Series Closeout

The `I` series is now closed:

- `I1` accepted as a taxonomy diagnostic
- `I2` accepted as a catalog-label diagnostic
- `I3` accepted as a diagnostic label-normalization candidate
- `I4` rejected as a retrieval candidate

The accepted runtime baseline remains unchanged:

- matcher `C1 + C2 + C5`
- retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- rerank prompt `F2` no-score candidate payload
- model `gpt-5.4-mini`
- reasoning `medium`

## Recommended Next Step

Do not start another retrieval expansion immediately.

The next phase should turn the accepted `I3` diagnostic split into a durable evaluation/reporting contract, then separately decide whether ambiguous assessment gold labels need benchmark review before new behavior changes.
