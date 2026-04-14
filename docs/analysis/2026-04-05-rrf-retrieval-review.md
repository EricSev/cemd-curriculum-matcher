# Hybrid Retrieval With RRF Review

- Date: 2026-04-05
- Task: `D1` hybrid retrieval with RRF
- Variable changed: stage-1 retrieval fusion only
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_state_normalization_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_rrf_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_rrf_records.csv`

## Code Change

Replaced the stage-1 recall fusion in `src/curriculum_matcher/app.py`:

- before: direct `0.5 * BM25 + 0.5 * semantic` score blend
- candidate: Reciprocal Rank Fusion over BM25 rank and semantic rank only

The change stayed intentionally narrow:

- no Tkinter operator-surface changes
- no final-score weight changes
- no repair-policy changes
- no rerank baseline changes
- recall depth stayed unchanged

## Benchmark Run Note

This environment required offline model flags for the benchmark command:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

Without them, SentenceTransformer attempted a Hugging Face network request and the benchmark failed before model load.

## Overall Benchmark Delta

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, shortlist `10`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.4760 | 0.4800 | +0.0040 |
| top-3 recall | 0.5740 | 0.5680 | -0.0060 |
| prediction rate | 0.9370 | 0.9410 | +0.0040 |
| shortlist hit@10 | 0.6190 | 0.6200 | +0.0010 |
| shortlist MRR | 0.5285 | 0.5292 | +0.0007 |
| shortlist nDCG@10 | 0.5508 | 0.5514 | +0.0006 |

Interpretation:

- topline movement is mixed
- top-1 and prediction rate improved modestly
- shortlist quality improved only slightly
- top-3 recall regressed

## Required Hard-Slice Delta

| Slice | Metric | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| `product_type_usage / Assessment` | top-1 | 0.3385 | 0.3577 | +0.0192 |
| `product_type_usage / Assessment` | top-3 | 0.4077 | 0.4000 | -0.0077 |
| `product_type_usage / Assessment` | top-10 | 0.4462 | 0.4423 | -0.0039 |
| `product_type_usage / Assessment` | MRR | 0.3760 | 0.3836 | +0.0076 |
| `placeholder_mapping / catalog_unspecified` | top-1 | 0.0893 | 0.0893 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | top-3 | 0.2143 | 0.1786 | -0.0357 |
| `placeholder_mapping / catalog_unspecified` | top-10 | 0.3393 | 0.3393 | +0.0000 |
| `placeholder_mapping / catalog_unspecified` | MRR | 0.1607 | 0.1539 | -0.0068 |
| `state_specific_risk / adoption_state_high_risk` | top-1 | 0.4728 | 0.4814 | +0.0086 |
| `state_specific_risk / adoption_state_high_risk` | top-3 | 0.5473 | 0.5473 | +0.0000 |
| `state_specific_risk / adoption_state_high_risk` | top-10 | 0.5673 | 0.5731 | +0.0058 |
| `state_specific_risk / adoption_state_high_risk` | MRR | 0.5104 | 0.5163 | +0.0059 |
| `state_specific_risk / catalog_state_specific_expected` | top-1 | 0.4717 | 0.4717 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-3 | 0.5283 | 0.5283 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | top-10 | 0.6981 | 0.6981 | +0.0000 |
| `state_specific_risk / catalog_state_specific_expected` | MRR | 0.5253 | 0.5253 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-1 | 0.3851 | 0.4161 | +0.0310 |
| `assessment_slice / assessment_short_or_acronym_title` | top-3 | 0.4658 | 0.4658 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | top-10 | 0.4720 | 0.4720 | +0.0000 |
| `assessment_slice / assessment_short_or_acronym_title` | MRR | 0.4221 | 0.4375 | +0.0154 |
| `assessment_slice / assessment_publisher_missing` | top-1 | 0.2840 | 0.2840 | +0.0000 |
| `assessment_slice / assessment_publisher_missing` | top-3 | 0.3333 | 0.3210 | -0.0123 |
| `assessment_slice / assessment_publisher_missing` | top-10 | 0.3704 | 0.3580 | -0.0124 |
| `assessment_slice / assessment_publisher_missing` | MRR | 0.3152 | 0.3107 | -0.0045 |
| `evidence_richness / sparse` | top-1 | 0.5048 | 0.5098 | +0.0050 |
| `evidence_richness / sparse` | top-3 | 0.6164 | 0.6114 | -0.0050 |
| `evidence_richness / sparse` | top-10 | 0.6743 | 0.6743 | +0.0000 |
| `evidence_richness / sparse` | MRR | 0.5824 | 0.5834 | +0.0010 |

Interpretation:

- acronym-heavy assessment rows improved at top-1 and MRR
- state-risk slices were neutral to slightly better
- sparse-evidence rows did not gain shortlist recall
- `catalog_unspecified` and `assessment_publisher_missing` regressed on protected recall metrics

## Decision

Decision: `rejected`

Why:

- the experiment produced only marginal overall shortlist gains
- the topline result is mixed because top-3 recall fell
- protected slices regressed where the review defaults require caution
- the retrieval lift is too small to justify changing the accepted baseline

## Baseline Update

No retrieval baseline change.

The active matcher baseline remains:

- `C1` safe publisher alias normalization
- `C2` safe product-title acronym expansion
- `C5` safe state-specific token normalization

The rerank baseline remains:

- shortlist `10`
- model `gpt-5.4-mini`
- reasoning `medium`

## Next Task

Move to `D2`:

- field-aware lexical weighting experiment
- keep the representative benchmark fixed
- compare against the accepted `C1 + C2 + C5` matcher baseline
- keep the benchmark command offline-safe in this environment
