# H4 State-Specific Candidate Balancing Review

- Date: `2026-04-16`
- Task: `H4` state-specific candidate balancing experiment
- Decision: `rejected`
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h4_state_balance_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h4_state_balance_records.csv`
- Comparison artifact: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h4_state_balance_comparison.json`

## Single Variable

- enabled an opt-in retrieval experiment named `state_balance`
- for `CA`, `FL`, and `TX` rows, added extra stage-1 candidates from catalog rows where:
  - `state_specific_version` is true
  - product or series text matches the row state
- let the existing scorer produce the final top-10

Held fixed:

- matcher baseline `C1 + C2 + C5`
- accepted stage-1 blend formula `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- no LLM rerank batch
- rerank prompt baseline `F2` no-score candidate payload

## Benchmark Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --retrieval-experiment state_balance --topn-final 10 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h4_state_balance_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_h4_state_balance_records.csv
```

## Overall Delta

| Metric | Baseline | H4 Candidate | Delta |
| --- | ---: | ---: | ---: |
| hit@1 / top-1 | 0.4880 | 0.4870 | -0.0010 |
| hit@3 | 0.5740 | 0.5730 | -0.0010 |
| hit@10 | 0.6280 | 0.6280 | +0.0000 |
| MRR | 0.5374 | 0.5368 | -0.0006 |
| nDCG@10 | 0.5596 | 0.5591 | -0.0005 |

Row movement:

- changed shortlist rows: `13`
- changed metric rows: `4`
- top-1 gains / losses: `0 / 1`
- top-3 gains / losses: `0 / 1`
- top-10 gains / losses: `0 / 0`

## Required Slice Readout

| Slice | Top-1 Delta | Top-3 Delta | Hit@10 Delta | Notes |
| --- | ---: | ---: | ---: | --- |
| adoption_state_high_risk | -0.0028 | -0.0029 | +0.0000 | `0 / 1` top-1 gains / losses |
| catalog_state_specific_expected | +0.0000 | +0.0000 | +0.0000 | `6` changed shortlist rows, no metric movement |
| assessment rows inside adoption states | n/a | n/a | n/a | `4` changed shortlist rows, no top-10 gains |
| CA | +0.0000 | +0.0000 | +0.0000 | no metric gain |
| FL | -0.0143 | +0.0000 | +0.0000 | one top-1 loss |
| TX | +0.0000 | -0.0175 | +0.0000 | one top-3 loss |

## Decision

- reject `H4`
- do not send an LLM rerank batch
- restore `src/curriculum_matcher/app.py` and `tests/test_matcher_core.py` to the accepted baseline after measurement

Why:

- the candidate produced no top-10 recovery gains
- target state-specific slices did not improve
- overall top-1, top-3, MRR, and nDCG regressed slightly
- shortlist movement remained too small and too weak to justify downstream LLM reranking

## H-Series Closeout

The `H` series is now closed:

- `H1` accepted as a diagnostic checkpoint
- `H2` rejected
- `H3` rejected
- `H4` rejected

The accepted runtime baseline remains unchanged:

- matcher `C1 + C2 + C5`
- retrieval `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- shortlist `10`
- rerank prompt `F2` no-score candidate payload
- model `gpt-5.4-mini`
- reasoning `medium`

## Recommended Next Step

Define a new experiment set before making any further behavior change.

The H-series result suggests that simple metadata-gated stage-1 expansion is not enough. A next phase should probably focus on error taxonomy or catalog-label structure before another retrieval tweak.
