# Grade Normalization Review

- Date: 2026-04-05
- Task: `C3` grade normalization tightening
- Variable changed: grade parsing / canonicalization only
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_acronym_alias_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_grade_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_grade_records.csv`

## Code Change Tested

The experiment corrected grade parsing so range-like values such as `K-5`, `K-12`, `PK-12`, `PK-K`, and `TK` were treated as actual spans instead of collapsing to the final numeric token.

This was a reasonable experiment because the catalog contains many `K-*` and `PK-*` spans, and the old parser materially misread them.

## Overall Benchmark Delta

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4750 | 0.4850 | +0.0100 |
| top-3 recall | 0.5720 | 0.5650 | -0.0070 |
| prediction rate | 0.9400 | 0.9320 | -0.0080 |
| shortlist hit@10 | 0.6130 | 0.5990 | -0.0140 |
| shortlist MRR | 0.5265 | 0.5289 | +0.0024 |
| shortlist nDCG@10 | 0.5479 | 0.5464 | -0.0015 |

Interpretation:

- top-1 improves
- top-3, prediction rate, and hit@10 regress
- the result is mixed rather than clearly positive against the accepted `C2` baseline

## Required Hard-Slice Delta

| Slice | Metric | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| `product_type_usage / Assessment` | top-1 | 0.3346 | 0.2577 | -0.0769 |
| `product_type_usage / Assessment` | top-3 | 0.4038 | 0.3462 | -0.0576 |
| `placeholder_mapping / catalog_unspecified` | top-1 | 0.0893 | 0.1607 | +0.0714 |
| `placeholder_mapping / catalog_unspecified` | top-3 | 0.2143 | 0.2500 | +0.0357 |
| `state_specific_risk / adoption_state_high_risk` | top-1 | 0.4699 | 0.4728 | +0.0029 |
| `state_specific_risk / adoption_state_high_risk` | top-3 | 0.5444 | 0.5387 | -0.0057 |
| `state_specific_risk / catalog_state_specific_expected` | top-1 | 0.4340 | 0.4340 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-3 | 0.5094 | 0.5283 | +0.0189 |
| `assessment_slice / assessment_short_or_acronym_title` | top-1 | 0.3789 | 0.2795 | -0.0994 |
| `assessment_slice / assessment_short_or_acronym_title` | top-3 | 0.4596 | 0.3727 | -0.0869 |
| `assessment_slice / assessment_publisher_missing` | top-1 | 0.2840 | 0.2346 | -0.0494 |
| `assessment_slice / assessment_publisher_missing` | top-3 | 0.3333 | 0.3210 | -0.0123 |

Interpretation:

- the grade parser helps some placeholder and state-sensitive rows
- the assessment regressions are too large to ignore
- the accepted `C2` assessment gains do not survive this parser change

## Grade-Sensitive Subset Check

On the subset whose expected catalog grades contain `K` / `PK` / `TK` style spans:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.5035 | 0.5718 | +0.0683 |
| top-3 recall | 0.6376 | 0.6518 | +0.0142 |
| shortlist hit@10 | 0.6941 | 0.6847 | -0.0094 |
| shortlist MRR | 0.5733 | 0.6146 | +0.0413 |

Interpretation:

- this is a real gain on the intended grade-span families
- but that improvement is not broad enough to offset the accepted assessment regressions

## Representative Wins

- `Benchmark Advance` / `Benchmark Education Company` / grade `2` moved `rank 2 -> rank 1` for `Benchmark Advance` (`K-6`)
- `Grammar for Writing` / `McDougal Littell` / grade `K` moved from miss to `rank 1` for `McDougal Littell Grammar for Writing` (`K-12`)
- `MyView Literacy` / `Savvas Learning Company` / grade `1` moved `rank 2 -> rank 1` for `myView Literacy` (`K-5`)
- `i-Ready` / blank publisher / grade `2` moved `rank 2 -> rank 1` for `i-Ready, ELA: Unspecified` (`K-8`)

## Representative Regressions

- `Smarter Balanced Summative Assessments` / grade `3` dropped `rank 1 -> rank 2`
- `Standardized Testing and Reporting (STAR) Program` / grade `2` dropped `rank 1 -> rank 3`
- `NWEA: MAP Growth: ELA: 3-11` / `NWEA` / grade `9` dropped `rank 1 -> rank 2`
- several `Smarter Balanced` assessment rows dropped from `rank 1 -> rank 2`

## Decision

Decision: `rejected`

Why:

- the experiment improves the intended grade-span subset
- but compared to the accepted `C2` baseline it regresses top-3, prediction rate, hit@10, and several key assessment slices
- under the current review rule, this is too mixed to promote

## Baseline Update

No baseline change.

The current matcher baseline remains:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion

The rerank baseline remains:

- shortlist size `10`
- model `gpt-5.4-mini`
- reasoning effort `medium`

## Follow-Up Note

The underlying parser issue is real, but the current grade fix needs a narrower follow-up if it is revisited. Any later retry should explicitly protect the accepted assessment gains from `C2`.
