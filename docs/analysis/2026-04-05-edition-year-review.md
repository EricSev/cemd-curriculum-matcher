# Edition And Year Extraction Review

- Date: 2026-04-05
- Task: `C4` edition / year extraction
- Variable changed: product-title cleanup for edition/year boilerplate only
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_acronym_alias_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_edition_year_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_edition_year_records.csv`

## Code Change

Updated `_normalize_product_title(...)` in `src/curriculum_matcher/app.py` to strip narrow edition/year boilerplate before title matching:

- leading year prefixes such as `2019/ 2024`
- copyright markers such as `©2014` and `Copyright 2012`
- ordinal edition phrases such as `4th Ed.` and `2nd Edition`
- trailing standalone years
- generic `edition` / `student edition` boilerplate

This change only affects product-title normalization paths. It does not change:

- the structured year score logic
- scoring weights
- publisher normalization
- rerank settings

## Overall Benchmark Delta

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4750 | 0.4670 | -0.0080 |
| top-3 recall | 0.5720 | 0.5710 | -0.0010 |
| prediction rate | 0.9400 | 0.9480 | +0.0080 |
| shortlist hit@10 | 0.6130 | 0.6190 | +0.0060 |
| shortlist MRR | 0.5265 | 0.5232 | -0.0033 |
| shortlist nDCG@10 | 0.5479 | 0.5469 | -0.0010 |

Interpretation:

- prediction rate and hit@10 improved
- top-1, MRR, and nDCG regressed
- overall the result is mixed rather than promotable

## Required Hard-Slice Delta

| Slice | Metric | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| `product_type_usage / Assessment` | top-1 | 0.3346 | 0.3308 | -0.0038 |
| `product_type_usage / Assessment` | top-3 | 0.4038 | 0.4000 | -0.0038 |
| `product_type_usage / Assessment` | top-10 | 0.4192 | 0.4115 | -0.0077 |
| `product_type_usage / Assessment` | MRR | 0.3686 | 0.3640 | -0.0046 |
| `placeholder_mapping / catalog_unspecified` | top-1 | 0.0893 | 0.0893 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-3 | 0.2143 | 0.1786 | -0.0357 |
| `placeholder_mapping / catalog_unspecified` | top-10 | 0.3393 | 0.3036 | -0.0357 |
| `placeholder_mapping / catalog_unspecified` | MRR | 0.1603 | 0.1471 | -0.0132 |
| `state_specific_risk / adoption_state_high_risk` | top-1 | 0.4699 | 0.4556 | -0.0143 |
| `state_specific_risk / adoption_state_high_risk` | top-3 | 0.5444 | 0.5415 | -0.0029 |
| `state_specific_risk / adoption_state_high_risk` | top-10 | 0.5645 | 0.5759 | +0.0114 |
| `state_specific_risk / adoption_state_high_risk` | MRR | 0.5076 | 0.5030 | -0.0046 |
| `state_specific_risk / catalog_state_specific_expected` | top-1 | 0.4340 | 0.4151 | -0.0189 |
| `state_specific_risk / catalog_state_specific_expected` | top-3 | 0.5094 | 0.5094 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-10 | 0.5849 | 0.5849 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | MRR | 0.4839 | 0.4723 | -0.0116 |
| `assessment_slice / assessment_short_or_acronym_title` | top-1 | 0.3789 | 0.3727 | -0.0062 |
| `assessment_slice / assessment_short_or_acronym_title` | top-3 | 0.4596 | 0.4596 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-10 | 0.4658 | 0.4658 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | MRR | 0.4159 | 0.4127 | -0.0032 |
| `assessment_slice / assessment_publisher_missing` | top-1 | 0.2840 | 0.2840 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-3 | 0.3333 | 0.3210 | -0.0123 |
| `assessment_slice / assessment_publisher_missing` | top-10 | 0.3704 | 0.3457 | -0.0247 |
| `assessment_slice / assessment_publisher_missing` | MRR | 0.3152 | 0.3071 | -0.0081 |

Interpretation:

- the protected slices do not hold up well enough
- `catalog_unspecified` regresses materially
- assessment slices slip modestly, which is enough to matter after the accepted `C2` gain

## Year / Edition Heavy Subset Check

On the subset of 43 rows whose district title or publisher contains explicit year / edition markers:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.5581 | 0.5814 | +0.0233 |
| top-3 recall | 0.6744 | 0.7442 | +0.0698 |
| shortlist hit@10 | 0.7674 | 0.8140 | +0.0466 |
| shortlist MRR | 0.6326 | 0.6733 | +0.0407 |

Interpretation:

- this is a real targeted improvement
- the intended year/edition-heavy subset benefits clearly
- but those gains do not survive the full benchmark comparison cleanly enough

## Representative Wins

- `2019/ 2024 Cengage Learning, Calculus of a Single Variable`: miss -> rank `1`
- `EnVision Algebra 1, SC 1st Edition`: rank `5 -> 1`
- `Modern Chemistry GA Ed`: miss -> rank `1`
- `Biology in Focus AP Edition, 2014, Campbell`: rank `2 -> 1`
- `2024 BFW, Myer's Psychology, 4th Ed.`: miss -> rank `2`

## Representative Regressions

- `Pearson Chemistry Foundation Edition (Copyright 2012) ...`: rank `3 -> 0`
- `The Cultural Landscape 2020`: rank `1 -> 2`
- `The Cultural Landscape: An Introduction to Human Geography, 12th ed.`: rank `1 -> 2`
- `World History, Florida Edition`: rank `1 -> 3`
- `MAGRUDERS AMERICAN GOVERNMENT 2018 VIRGINIA STUDENT EDITION`: rank `4 -> 5`

## Decision

Decision: `rejected`

Why:

- the intended subset improved clearly
- but the accepted benchmark baseline regressed on top-1, MRR, and multiple protected slices
- under the current one-variable-at-a-time review rule, this is too mixed to promote

## Baseline Update

No baseline change.

The current matcher baseline remains:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion

The rerank baseline remains:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## Follow-Up Note

The year/edition cleanup is directionally useful, but it likely needs a narrower retry later. A future pass should protect:

- `catalog_unspecified`
- state-specific variants
- Florida / Virginia edition-sensitive rows

instead of broadly stripping edition/year boilerplate everywhere.
