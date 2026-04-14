# E1 Cross-Encoder Pre-LLM Reranker Review

- Date: `2026-04-06`
- Task: `E1` cross-encoder pre-LLM reranker experiment
- Decision: `rejected`
- Baseline summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_char_ngram_summary.json`
- Candidate summary: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_cross_encoder_summary.json`
- Candidate records: `benchmarks/outputs/historical_07122025_representative_1000_fast_top10_cross_encoder_records.csv`

Representative benchmark: `historical_07122025_representative_1000`, profile `fast`, retrieval baseline `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`

Single variable:

- rerank experiment `default` -> `cross_encoder`
- cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- shortlist held fixed at `10`

## Overall Delta

| Metric | Baseline | Cross-Encoder | Delta |
| --- | --- | --- | --- |
| top-1 accuracy | 0.4880 | 0.4860 | -0.0020 |
| top-3 recall | 0.5740 | 0.5860 | +0.0120 |
| shortlist hit@10 | 0.6280 | 0.6250 | -0.0030 |
| shortlist MRR | 0.5374 | 0.5391 | +0.0017 |
| shortlist nDCG@10 | 0.5596 | 0.5604 | +0.0008 |

Interpretation:

- the cross-encoder improved mid-list quality, but not enough to justify a baseline change
- the protected top-1 metric regressed slightly
- shortlist hit@10 also regressed, which weakens the pre-LLM shortlist contract rather than strengthening it
- cost stayed stable because shortlist size remained `10`, but quality did not improve cleanly enough to accept the cascade

## Row-Level Movement

- top-1 gains: `51`
- top-1 losses: `53`
- top-3 gains: `27`
- top-3 losses: `15`
- top-10 gains: `5`
- top-10 losses: `8`

The candidate helped more rows inside the top-3 than it hurt, but it still lost net ground at top-1 and top-10.

## Required Slice Readout

- `Assessment`: top-1 `+0.0308`, top-10 `+0.0039`, MRR `+0.0221`
- `assessment_short_or_acronym_title`: top-1 `+0.0186`, top-3 `-0.0062`, top-10 `+0.0000`, MRR `+0.0081`
- `adoption_state_high_risk`: top-1 `-0.0143`, top-10 `+0.0058`, MRR `-0.0048`
- `sparse`: top-1 `+0.0126`, top-10 `-0.0025`, MRR `+0.0092`
- `catalog_state_specific_expected`: top-1 `-0.0377`, top-10 `+0.0000`, MRR `-0.0118`

Known mixed-risk movement:

- `catalog_unspecified`: top-1 `+0.0714`, top-3 `+0.0714`, top-10 `-0.0357`
- `assessment_publisher_missing`: top-1 `+0.0367`, top-10 `+0.0045`, MRR `+0.0261`

## Keep / Reject Decision

- reject `E1`
- keep the current rerank baseline

Why:

- roadmap default says mixed results should be rejected
- the experiment did not deliver a clean top-1 lift
- the experiment also reduced top-10 hit rate, which is too costly for a pre-LLM shortlist reranker
- improvements were real in some difficult slices, but they were not broad or safe enough to replace the locked baseline

## Baseline Update

- no baseline change
- retrieval baseline remains `0.25 * BM25 + 0.5 * semantic + 0.25 * char n-gram`
- rerank baseline remains:
  - shortlist `10`
  - model `gpt-5.4-mini`
  - reasoning `medium`

