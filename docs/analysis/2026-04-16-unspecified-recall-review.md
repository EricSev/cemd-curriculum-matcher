# H3 Catalog-Unspecified Recovery Review

- Date: `2026-04-16`
- Task: `H3` catalog-unspecified recovery experiment
- Decision: `rejected`
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h3_unspecified_recall_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h3_unspecified_recall_records.csv`
- Comparison artifact: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h3_unspecified_recall_comparison.json`

## Single Variable

- enabled an opt-in retrieval experiment named `unspecified_recall`
- added extra stage-1 candidates from catalog rows whose product or series looked `Unspecified`, no-information, or district-created before the existing scorer produced the final top-10

Held fixed:

- matcher baseline `C1 + C2 + C5`
- accepted stage-1 blend formula `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- no LLM rerank batch
- rerank prompt baseline `F2` no-score candidate payload

## Benchmark Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --retrieval-experiment unspecified_recall --topn-final 10 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h3_unspecified_recall_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h3_unspecified_recall_records.csv
```

## Overall Delta

| Metric | Baseline | H3 Candidate | Delta |
| --- | ---: | ---: | ---: |
| hit@1 / top-1 | 0.4880 | 0.4850 | -0.0030 |
| hit@3 | 0.5740 | 0.5740 | +0.0000 |
| hit@10 | 0.6280 | 0.6280 | +0.0000 |
| MRR | 0.5374 | 0.5358 | -0.0016 |
| nDCG@10 | 0.5596 | 0.5584 | -0.0012 |

Row movement:

- changed shortlist rows: `35`
- changed metric rows: `11`
- top-1 gains / losses: `2 / 5`
- top-3 gains / losses: `0 / 0`
- top-10 gains / losses: `0 / 0`

## Required Slice Readout

| Slice | Top-1 Delta | Top-3 Delta | Hit@10 Delta | Notes |
| --- | ---: | ---: | ---: | --- |
| catalog_unspecified | +0.0000 | +0.0000 | +0.0000 | only `1` changed shortlist row |
| sparse | -0.0076 | +0.0000 | +0.0000 | `1 / 4` top-1 gains / losses |
| medium | +0.0000 | +0.0000 | +0.0000 | `1 / 1` top-1 gains / losses |
| Assessment | +0.0000 | +0.0000 | +0.0000 | no target recovery |
| adoption_state_high_risk | -0.0028 | +0.0000 | +0.0000 | one top-1 loss |

## Decision

- reject `H3`
- do not send an LLM rerank batch
- restore `src/curriculum_matcher/app.py` and `tests/test_matcher_core.py` to the accepted baseline after measurement

Why:

- the target `catalog_unspecified` slice did not improve
- overall top-1, MRR, and nDCG regressed
- top-1 losses outnumbered gains
- there were no top-10 recovery gains to justify downstream LLM reranking

## Recommended Next Step

Proceed to `H4` only if continuing the `H` series.

`H4` should focus on state-specific candidate balancing, but it should avoid broad candidate expansion patterns like `H2` and `H3` because those moved shortlists without recovering gold rows.
