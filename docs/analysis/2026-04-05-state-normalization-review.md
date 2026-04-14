# State-Specific Token Normalization Review

- Date: 2026-04-05
- Task: `C5` state-specific token normalization
- Variable changed: safe state-specific product-title normalization only
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_acronym_alias_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_state_normalization_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_state_normalization_records.csv`

## Code Change

Extended `product_title_alias_patterns` in `src/curriculum_matcher/app.py` with exact state-specific expansions only:

- `CAASPP`
- `ELPAC`
- `Alternate ELPAC`
- `STAAR`
- `TELPAS`
- `TEKS`
- `GA Ed`
- `CA Studies`
- `SC 1st Edition`
- `Florida's B.E.S.T.`

The change is intentionally narrow:

- product-title normalization only
- no scoring-weight changes
- no retrieval architecture changes
- no rerank changes
- ambiguous short tokens such as `FAST` remain unexpanded

## Overall Benchmark Delta

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4750 | 0.4760 | +0.0010 |
| top-3 recall | 0.5720 | 0.5740 | +0.0020 |
| prediction rate | 0.9400 | 0.9370 | -0.0030 |
| shortlist hit@10 | 0.6130 | 0.6190 | +0.0060 |
| shortlist MRR | 0.5265 | 0.5285 | +0.0020 |
| shortlist nDCG@10 | 0.5479 | 0.5508 | +0.0029 |

Interpretation:

- overall movement is modest but positive
- prediction rate dips slightly
- hit@10, MRR, and nDCG all improve

## Required Hard-Slice Delta

| Slice | Metric | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| `product_type_usage / Assessment` | top-1 | 0.3346 | 0.3385 | +0.0039 |
| `product_type_usage / Assessment` | top-3 | 0.4038 | 0.4077 | +0.0039 |
| `product_type_usage / Assessment` | top-10 | 0.4192 | 0.4462 | +0.0270 |
| `product_type_usage / Assessment` | MRR | 0.3686 | 0.3760 | +0.0074 |
| `placeholder_mapping / catalog_unspecified` | top-1 | 0.0893 | 0.0893 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-3 | 0.2143 | 0.2143 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-10 | 0.3393 | 0.3393 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | MRR | 0.1603 | 0.1607 | +0.0004 |
| `state_specific_risk / adoption_state_high_risk` | top-1 | 0.4699 | 0.4728 | +0.0029 |
| `state_specific_risk / adoption_state_high_risk` | top-3 | 0.5444 | 0.5473 | +0.0029 |
| `state_specific_risk / adoption_state_high_risk` | top-10 | 0.5645 | 0.5673 | +0.0028 |
| `state_specific_risk / adoption_state_high_risk` | MRR | 0.5076 | 0.5104 | +0.0028 |
| `state_specific_risk / catalog_state_specific_expected` | top-1 | 0.4340 | 0.4717 | +0.0377 |
| `state_specific_risk / catalog_state_specific_expected` | top-3 | 0.5094 | 0.5283 | +0.0189 |
| `state_specific_risk / catalog_state_specific_expected` | top-10 | 0.5849 | 0.6981 | +0.1132 |
| `state_specific_risk / catalog_state_specific_expected` | MRR | 0.4839 | 0.5253 | +0.0414 |
| `assessment_slice / assessment_short_or_acronym_title` | top-1 | 0.3789 | 0.3851 | +0.0062 |
| `assessment_slice / assessment_short_or_acronym_title` | top-3 | 0.4596 | 0.4658 | +0.0062 |
| `assessment_slice / assessment_short_or_acronym_title` | top-10 | 0.4658 | 0.4720 | +0.0062 |
| `assessment_slice / assessment_short_or_acronym_title` | MRR | 0.4159 | 0.4221 | +0.0062 |
| `assessment_slice / assessment_publisher_missing` | top-1 | 0.2840 | 0.2840 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-3 | 0.3333 | 0.3333 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-10 | 0.3704 | 0.3704 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | MRR | 0.3152 | 0.3152 | +0.0000 |

Interpretation:

- the protected slices held or improved
- `catalog_state_specific_expected` improved materially
- `catalog_unspecified` stayed stable
- assessment slices stayed safe and slightly improved

## State-Token Subset Check

On the targeted subset of rows containing explicit state-program tokens such as `CAASPP`, `ELPAC`, `STAAR`, `TELPAS`, `TEKS`, `B.E.S.T.`, `GA Ed`, `SC 1st Edition`, or `CA Studies`:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.0000 | 0.1000 | +0.1000 |
| top-3 recall | 0.0250 | 0.1000 | +0.0750 |
| shortlist hit@10 | 0.0750 | 0.2500 | +0.1750 |
| shortlist MRR | 0.0238 | 0.1237 | +0.0999 |

Interpretation:

- the intended subset improved clearly
- the gain is concentrated in the exact state-program normalization targets

## Representative Wins

- `ELPAC`: miss -> rank `1`
- several `STAAR` rows: miss -> shortlist rank `7` or `8`
- `STAAR | Texas Education Agency`: miss -> rank `4`
- `Florida B.E.S.T. Math Savvas Envision`: rank `2 -> 1`
- `Florida's B.E.S.T. Math`: miss -> rank `1`
- `Eureka Maths TEKS`: rank `4 -> 1`

## Representative Neutral / Weak Cases

- most `CAASPP` spellings still miss the shortlist
- `TELPAS` remains largely unchanged
- `EnVision Algebra 1, SC 1st Edition` regressed from rank `5` into a miss
- `Modern Chemistry GA Ed` remained a miss

## Decision

Decision: `accepted`

Why:

- overall benchmark movement is modest but positive
- the intended state-risk slices improved meaningfully
- the protected assessment and placeholder slices did not regress
- the state-token subset improved enough to justify promotion

## Baseline Update

The matcher baseline now includes:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

The rerank baseline remains:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## Next Task

Move to `D1`:

- hybrid retrieval with RRF
- keep the representative benchmark fixed
- compare against the accepted `C1 + C2 + C5` matcher baseline
