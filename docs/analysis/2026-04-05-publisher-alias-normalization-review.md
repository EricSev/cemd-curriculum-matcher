# Publisher Alias Normalization Review

- Date: 2026-04-05
- Task: `C1` publisher alias normalization
- Variable changed: safe publisher alias normalization only
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_publisher_alias_records.csv`

## Code Change

Updated publisher canonicalization in `src/curriculum_matcher/app.py` to add a narrow set of exact aliases that map common publisher variants onto already-observed catalog forms without collapsing the risky `Pearson` <-> `Savvas` distinction.

Added test coverage in `tests/test_matcher_core.py` for:

- `Benchmark Education Co.` -> `benchmark education company`
- `Pearson/Prentice Hall` -> `pearson education`
- `Holt, Rinehart and Winston` -> `hmh`
- exact publisher-score matches for safe alias pairs

## Overall Benchmark Delta

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4600 | 0.4650 | +0.0050 |
| top-3 recall | 0.5610 | 0.5590 | -0.0020 |
| prediction rate | 0.9320 | 0.9320 | +0.0000 |
| shortlist hit@10 | 0.5990 | 0.6000 | +0.0010 |
| shortlist MRR | 0.5126 | 0.5148 | +0.0022 |
| shortlist nDCG@10 | 0.5341 | 0.5359 | +0.0018 |

Interpretation:

- overall movement is small
- the experiment does not materially regress the benchmark
- the gain is not broad, but it is directionally positive on top-1, hit@10, MRR, and nDCG@10

## Required Hard-Slice Delta

| Slice | Metric | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| `product_type_usage / Assessment` | top-1 | 0.2962 | 0.2962 | +0.0000 |
| `product_type_usage / Assessment` | top-3 | 0.3577 | 0.3577 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-1 | 0.0893 | 0.0893 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-3 | 0.2321 | 0.2143 | -0.0178 |
| `state_specific_risk / adoption_state_high_risk` | top-1 | 0.4470 | 0.4527 | +0.0057 |
| `state_specific_risk / adoption_state_high_risk` | top-3 | 0.5272 | 0.5244 | -0.0028 |
| `state_specific_risk / catalog_state_specific_expected` | top-1 | 0.4151 | 0.4340 | +0.0189 |
| `state_specific_risk / catalog_state_specific_expected` | top-3 | 0.5094 | 0.5094 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-1 | 0.3106 | 0.3106 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-3 | 0.3851 | 0.3851 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-1 | 0.2840 | 0.2840 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-3 | 0.3210 | 0.3210 | +0.0000 |

Interpretation:

- no improvement on `Assessment`
- no improvement on assessment-specific missing-publisher rows
- a small regression on `catalog_unspecified` top-3 recall
- the main lift is concentrated in publisher/state-sensitive non-assessment rows

## Target Publisher-Variant Slice Check

On a targeted publisher-variant subset covering `McGraw`, `Houghton`, `Savvas`, `Pearson`, `Benchmark`, `HMH`, and `Holt` rows:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4627 | 0.4925 | +0.0298 |
| top-3 recall | 0.5821 | 0.5746 | -0.0075 |
| shortlist hit@10 | 0.6269 | 0.6418 | +0.0149 |
| shortlist MRR | 0.5260 | 0.5436 | +0.0176 |

Interpretation:

- this is the strongest evidence for keeping the change
- the experiment improves the intended publisher-variant subset on top-1, top-10, and MRR
- the loss on top-3 is small relative to the intended top-1 and shortlist gains

## Decision

Decision: `accepted`

Why:

- the experiment changed only one variable
- the representative benchmark shows no material overall regression
- the intended publisher-variant slice improved meaningfully
- the alias set stays conservative and does not introduce a risky `Pearson` <-> `Savvas` collapse

Scope of acceptance:

- accept this as a narrow normalization improvement
- do not describe it as a general fix for `Assessment` or `catalog_unspecified`
- keep watching `catalog_unspecified` in later normalization and retrieval experiments

## Baseline Update

The locked rerank baseline remains:

- shortlist size `10`
- model `gpt-5.4-mini`
- reasoning effort `medium`

The matcher baseline now includes the accepted safe publisher alias normalization in `src/curriculum_matcher/app.py`.

## Next Task

Move to `C2`:

- acronym and abbreviation expansion experiment
- keep the representative benchmark fixed
- keep rerank settings fixed
- change only acronym handling
