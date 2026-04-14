# Field-Aware Lexical Weighting Review

- Date: 2026-04-05
- Task: `D2` field-aware lexical weighting experiment
- Variable changed: stage-1 lexical weighting only
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_state_normalization_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_field_lexical_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_field_lexical_records.csv`

## Code Change

Candidate retrieval path:

- kept semantic recall unchanged
- kept final candidate scoring unchanged
- kept repair policy unchanged
- kept rerank baseline unchanged
- changed only the lexical recall component to split title and publisher BM25 signals before blending with semantic recall

This candidate was measured and then reverted because the lift did not clearly beat the accepted baseline.

## Benchmark Run Note

This environment still needed offline model flags:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

Benchmark command used:

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m curriculum_matcher.evaluation_cli --benchmark-file benchmarks/gold/historical_07122025_representative_1000.csv --catalog-file "/Users/ericseverson/Library/CloudStorage/OneDrive-Personal/Consulting/Meg/02-data/evaluation/historical-runs/Test Data 004/CEMD Product Catalog - 07122025.csv" --profile fast --topn-final 10 --output-json benchmarks/outputs/historical_07122025_representative_1000_fast_top10_field_lexical_summary.json --output-csv benchmarks/outputs/historical_07122025_representative_1000_fast_top10_field_lexical_records.csv`

## Overall Benchmark Delta

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4760 | 0.4760 | +0.0000 |
| top-3 recall | 0.5740 | 0.5740 | +0.0000 |
| prediction rate | 0.9370 | 0.9370 | +0.0000 |
| shortlist hit@10 | 0.6190 | 0.6190 | +0.0000 |
| shortlist MRR | 0.5285 | 0.5285 | +0.0000 |
| shortlist nDCG@10 | 0.5508 | 0.5508 | +0.0000 |

Interpretation:

- the saved benchmark summary is flat across all topline metrics
- the candidate did not improve shortlist quality on the representative set
- the experiment does not justify changing the accepted retrieval baseline

## Required Hard-Slice Delta

| Slice | Metric | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| `product_type_usage / Assessment` | top-1 | 0.3385 | 0.3385 | +0.0000 |
| `product_type_usage / Assessment` | top-3 | 0.4077 | 0.4077 | +0.0000 |
| `product_type_usage / Assessment` | top-10 | 0.4462 | 0.4462 | +0.0000 |
| `product_type_usage / Assessment` | MRR | 0.3760 | 0.3760 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-1 | 0.0893 | 0.0893 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-3 | 0.2143 | 0.2143 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-10 | 0.3393 | 0.3393 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | MRR | 0.1607 | 0.1607 | +0.0000 |
| `state_specific_risk / adoption_state_high_risk` | top-1 | 0.4728 | 0.4728 | +0.0000 |
| `state_specific_risk / adoption_state_high_risk` | top-3 | 0.5473 | 0.5473 | +0.0000 |
| `state_specific_risk / adoption_state_high_risk` | top-10 | 0.5673 | 0.5673 | +0.0000 |
| `state_specific_risk / adoption_state_high_risk` | MRR | 0.5104 | 0.5104 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-1 | 0.4717 | 0.4717 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-3 | 0.5283 | 0.5283 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-10 | 0.6981 | 0.6981 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | MRR | 0.5253 | 0.5253 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-1 | 0.3851 | 0.3851 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-3 | 0.4658 | 0.4658 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-10 | 0.4720 | 0.4720 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | MRR | 0.4221 | 0.4221 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-1 | 0.2840 | 0.2840 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-3 | 0.3333 | 0.3333 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-10 | 0.3704 | 0.3704 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | MRR | 0.3152 | 0.3152 | +0.0000 |
| `evidence_richness / sparse` | top-1 | 0.4887 | 0.4887 | +0.0000 |
| `evidence_richness / sparse` | top-3 | 0.6171 | 0.6171 | +0.0000 |
| `evidence_richness / sparse` | top-10 | 0.6499 | 0.6499 | +0.0000 |
| `evidence_richness / sparse` | MRR | 0.5515 | 0.5515 | +0.0000 |

Interpretation:

- every required review slice stayed flat at the summary level
- the candidate did not produce measurable lift where `D2` was supposed to help

## Row-Level Churn Check

The candidate was not a literal no-op:

- `51` rows changed expected shortlist rank
- top-1 improved on `25` rows and regressed on `9`
- top-3 improved on `19` rows and regressed on `6`
- top-10 improved on `24` rows and regressed on `4`

Representative wins:

- `ELPAC`: miss -> rank `1`
- `TS GOLD`: miss -> rank `1`
- `DRA: K-5 | Pearson Education`: miss -> rank `1`
- `STAAR`: miss -> rank `8`

Representative regressions:

- `NWEA: MAP Growth: ELA: 3-11 | NWEA`: rank `1` -> miss
- `WIDA Access`: rank `1` -> miss
- `EnVision Algebra 1, SC 1st Edition | Savvas`: rank `5` -> miss
- `Precalculus, TX ed | McGraw Hill`: rank `1` -> miss

Interpretation:

- the experiment moved individual rows in both directions
- gains and regressions netted out to zero on the representative benchmark
- that churn creates risk without producing a measurable benchmark win

## Decision

Decision: `rejected`

Why:

- the candidate did not clearly beat the accepted baseline
- topline and hard-slice summaries were flat
- row-level churn introduced regressions without benchmark lift
- review defaults for this phase say non-winning retrieval changes should stay rejected

## Baseline Update

No retrieval baseline change.

The active matcher baseline remains:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

The active retrieval path remains:

- direct `0.5 * BM25 + 0.5 * semantic` stage-1 recall blend

The rerank baseline remains:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## Next Task

Move to `D3`:

- character n-gram retrieval experiment
- keep the representative historical benchmark fixed
- keep the accepted matcher and rerank baselines fixed
- continue reporting assessment slices, `catalog_unspecified`, state-risk slices, and sparse-evidence rows explicitly
