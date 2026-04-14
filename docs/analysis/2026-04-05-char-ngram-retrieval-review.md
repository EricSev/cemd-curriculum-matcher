# Character N-Gram Retrieval Review

- Date: `2026-04-05`
- Task: `D3` character n-gram retrieval experiment
- Variable changed: stage-1 retrieval only
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_state_normalization_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_records.csv`

## Code Change

Candidate retrieval path:

- kept the accepted `C1 + C2 + C5` matcher baseline unchanged
- kept final candidate scoring unchanged
- kept repair policy unchanged
- kept rerank baseline unchanged
- changed only stage-1 recall to blend `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`

The rerun-backed source now keeps this retrieval path as the default baseline while preserving env override control for future experiments.

## Benchmark Run Note

This environment still needed offline model flags:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

Benchmark command used for the acceptance rerun:

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --topn-final 10 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_records.csv`

Validation command:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_matcher_core tests.test_evaluation -q`

## Overall Benchmark Delta

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4760 | 0.4880 | +0.0120 |
| top-3 recall | 0.5740 | 0.5740 | +0.0000 |
| prediction rate | 0.9370 | 0.9430 | +0.0060 |
| shortlist hit@10 | 0.6190 | 0.6280 | +0.0090 |
| shortlist MRR | 0.5285 | 0.5374 | +0.0089 |
| shortlist nDCG@10 | 0.5508 | 0.5596 | +0.0088 |

Interpretation:

- this is a real shortlist-quality win, not a flat or noise-only result
- top-1 accuracy improved while top-3 recall held flat
- shortlist hit@10, MRR, and nDCG@10 all improved together
- prediction rate also improved, which reduces no-prediction churn risk

## Required Hard-Slice Delta

| Slice | Metric | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| `product_type_usage / Assessment` | top-1 | 0.3385 | 0.3654 | +0.0269 |
| `product_type_usage / Assessment` | top-3 | 0.4077 | 0.4038 | -0.0039 |
| `product_type_usage / Assessment` | top-10 | 0.4462 | 0.4538 | +0.0076 |
| `product_type_usage / Assessment` | MRR | 0.3760 | 0.3920 | +0.0160 |
| `placeholder_mapping / catalog_unspecified` | top-1 | 0.0893 | 0.0893 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-3 | 0.2143 | 0.1786 | -0.0357 |
| `placeholder_mapping / catalog_unspecified` | top-10 | 0.3393 | 0.3393 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | MRR | 0.1607 | 0.1563 | -0.0044 |
| `state_specific_risk / adoption_state_high_risk` | top-1 | 0.4728 | 0.4842 | +0.0114 |
| `state_specific_risk / adoption_state_high_risk` | top-3 | 0.5473 | 0.5530 | +0.0057 |
| `state_specific_risk / adoption_state_high_risk` | top-10 | 0.5673 | 0.5759 | +0.0086 |
| `state_specific_risk / adoption_state_high_risk` | MRR | 0.5104 | 0.5203 | +0.0099 |
| `state_specific_risk / catalog_state_specific_expected` | top-1 | 0.4717 | 0.4717 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-3 | 0.5283 | 0.5283 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-10 | 0.6981 | 0.6981 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | MRR | 0.5253 | 0.5253 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-1 | 0.3851 | 0.4286 | +0.0435 |
| `assessment_slice / assessment_short_or_acronym_title` | top-3 | 0.4658 | 0.4658 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-10 | 0.4720 | 0.4720 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | MRR | 0.4221 | 0.4469 | +0.0248 |
| `assessment_slice / assessment_publisher_missing` | top-1 | 0.2840 | 0.2840 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-3 | 0.3333 | 0.3210 | -0.0123 |
| `assessment_slice / assessment_publisher_missing` | top-10 | 0.3704 | 0.3951 | +0.0247 |
| `assessment_slice / assessment_publisher_missing` | MRR | 0.3152 | 0.3171 | +0.0019 |
| `evidence_richness / sparse` | top-1 | 0.4887 | 0.5063 | +0.0176 |
| `evidence_richness / sparse` | top-3 | 0.6171 | 0.6146 | -0.0025 |
| `evidence_richness / sparse` | top-10 | 0.6499 | 0.6574 | +0.0075 |
| `evidence_richness / sparse` | MRR | 0.5515 | 0.5646 | +0.0131 |

Interpretation:

- the most important retrieval-facing slices improved where this experiment was supposed to help, especially assessment acronym rows and high-risk state rows
- `catalog_unspecified` stayed weak and lost top-3 recall, so that slice remains a known retrieval risk
- even with that weakness, there were no protected top-10 regressions on `catalog_unspecified` or state-specific rows
- the shortlist-quality gains are broad enough to accept while carrying the weak-slice risk forward into `D4`

## Row-Level Churn Check

- `42` rows changed expected shortlist rank
- top-1 improved on `15` rows and regressed on `3`
- top-3 improved on `4` rows and regressed on `4`
- top-10 improved on `9` rows and regressed on `0`

Representative wins:

- `WIDA`: rank `3` -> rank `1`
- `ReadTheory`: rank `5` -> rank `3`
- `Read Works`: rank `6` -> rank `4`
- `MAP Growth`: rank `2` -> rank `1`

Representative regressions:

- `ACCESS for English Language Learners`: rank `3` -> rank `6`
- `Go Math!`: rank `3` -> rank `5`
- `Rocket Math`: rank `2` -> rank `4`
- `ST Math`: rank `1` -> rank `2`

Interpretation:

- churn was limited rather than broad
- row movement skewed positive at top-1 and top-10
- the lack of any top-10 regressions is the strongest reason this candidate is safe enough to adopt

## Decision

Decision: `accepted`

Why:

- topline shortlist metrics all improved except top-3 recall, which stayed flat
- key target slices improved on top-1, top-10, and MRR
- row-level movement added `9` new top-10 hits with `0` top-10 losses
- this is the first retrieval experiment in the `D` series to produce a clean enough benchmark win to replace the direct blend

## Baseline Update

The active matcher baseline remains:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

The active retrieval baseline is now:

- `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram` stage-1 recall blend

The rerank baseline remains:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## Next Task

Move to `D4`:

- retrieval baseline decision checkpoint
- compare accepted retrieval candidates only
- confirm the accepted retrieval default and whether shortlist stays `10`
- keep the representative historical benchmark and rerank settings fixed
