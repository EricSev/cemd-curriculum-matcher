# Acronym And Abbreviation Expansion Review

- Date: 2026-04-05
- Task: `C2` acronym and abbreviation expansion
- Variable changed: safe product-title acronym expansion only
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_acronym_alias_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_acronym_alias_records.csv`

## Code Change

Added conservative acronym expansion in `src/curriculum_matcher/app.py` through `_normalize_product_title(...)`, and used it only in product-title paths:

- catalog `search_text`
- product-name fuzzy scoring
- stage-1 retrieval query text

The expansion intentionally stays narrow:

- expands safe assessment acronyms such as `NAEP`, `ELPA21`, `DRA`, `TS GOLD`, `TSG`, `WIDA ACCESS`, and `NWEA MAP`
- leaves ambiguous short titles like `MAP`, `ACCESS`, and `FAST` unexpanded unless a safer surrounding phrase is present

Added matcher-core coverage in `tests/test_matcher_core.py` to verify:

- safe acronym expansion happens
- ambiguous short tokens remain unexpanded
- fuzzy matching improves on a safe acronym pair

## Overall Benchmark Delta

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4650 | 0.4750 | +0.0100 |
| top-3 recall | 0.5590 | 0.5720 | +0.0130 |
| prediction rate | 0.9320 | 0.9400 | +0.0080 |
| shortlist hit@10 | 0.6000 | 0.6130 | +0.0130 |
| shortlist MRR | 0.5148 | 0.5265 | +0.0117 |
| shortlist nDCG@10 | 0.5359 | 0.5479 | +0.0120 |

Interpretation:

- this is a meaningful overall improvement
- the gains are not limited to one metric family
- the intended acronym-heavy assessment slice improved directly

## Required Hard-Slice Delta

| Slice | Metric | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| `assessment_slice / assessment_short_or_acronym_title` | top-1 | 0.3106 | 0.3789 | +0.0683 |
| `assessment_slice / assessment_short_or_acronym_title` | top-3 | 0.3851 | 0.4596 | +0.0745 |
| `assessment_slice / assessment_short_or_acronym_title` | top-10 | 0.3913 | 0.4658 | +0.0745 |
| `assessment_slice / assessment_short_or_acronym_title` | MRR | 0.3445 | 0.4159 | +0.0714 |
| `assessment_slice / assessment_publisher_missing` | top-1 | 0.2840 | 0.2840 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-3 | 0.3210 | 0.3333 | +0.0123 |
| `assessment_slice / assessment_publisher_missing` | top-10 | 0.3580 | 0.3704 | +0.0124 |
| `assessment_slice / assessment_publisher_missing` | MRR | 0.3091 | 0.3152 | +0.0061 |
| `evidence_richness / sparse` | top-1 | 0.4761 | 0.4912 | +0.0151 |
| `evidence_richness / sparse` | top-3 | 0.5945 | 0.6146 | +0.0201 |
| `evidence_richness / sparse` | top-10 | 0.6272 | 0.6474 | +0.0202 |
| `evidence_richness / sparse` | MRR | 0.5334 | 0.5514 | +0.0180 |
| `product_type_usage / Assessment` | top-1 | 0.2962 | 0.3346 | +0.0384 |
| `product_type_usage / Assessment` | top-3 | 0.3577 | 0.4038 | +0.0461 |
| `product_type_usage / Assessment` | top-10 | 0.3731 | 0.4192 | +0.0461 |
| `product_type_usage / Assessment` | MRR | 0.3262 | 0.3686 | +0.0424 |
| `placeholder_mapping / catalog_unspecified` | top-1 | 0.0893 | 0.0893 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-3 | 0.2143 | 0.2143 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-10 | 0.3393 | 0.3393 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | MRR | 0.1603 | 0.1603 | +0.0000 |

Interpretation:

- the intended acronym-heavy assessment slice improved strongly
- sparse rows improved
- `Assessment` improved materially
- `catalog_unspecified` stayed flat rather than regressing

## Changed-Row Check

Observed improvements in the candidate records:

- `TS GOLD`: expected rank `0 -> 1`, top-1 `False -> True`
- `DRA: K-5`: expected rank `0 -> 1`, top-1 `False -> True`
- `WIDA ACCESS`: expected rank `0 -> 1`, top-1 `False -> True`
- `DRA`: expected rank `0 -> 1`, top-1 `False -> True`
- multiple `NAEP` rows: expected rank `0 -> 1` or `2`, top-10 `False -> True`

Interpretation:

- the changed rows are directly aligned with the exact phrase expansions
- the gain is coming from the intended assessment-acronym targets, not from unrelated side effects

## Decision

Decision: `accepted`

Why:

- the task acceptance rule for `C2` is improvement on acronym-heavy rows without broader regression
- the overall benchmark improved clearly
- the intended acronym-heavy assessment slice improved strongly
- the result is strong enough to promote into the matcher baseline

## Baseline Update

The matcher baseline now includes:

- safe publisher alias normalization accepted in `C1`
- safe product-title acronym expansion accepted in `C2`
- rerank baseline unchanged: shortlist `10`, model `gpt-5.4-mini`, reasoning `medium`

## Next Task

Move to `C3`:

- grade normalization tightening
- keep the representative benchmark fixed
- keep the accepted publisher alias and acronym baseline
